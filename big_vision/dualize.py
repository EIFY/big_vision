import functools
import math
from typing import NamedTuple, Optional, Union

import chex
import jax
from jax import nn
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax.transforms import _accumulation
from optax.transforms import _adding

import modula.compound

import big_vision.models.modula_vit


def orthogonalize(M, eps: float = 1e-8):
    # six step Newton-Schulz by @YouJiacheng
    # coefficients from: https://twitter.com/YouJiacheng/status/1893704552689303901
    # found by optimization: https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae
    # the idea of stability loss was from @leloykun

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / (jnp.linalg.norm(M) + eps)
    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M


def scale_by_linear_dualize(column_axis: int = -1, target_norm: float = 1.0) -> base.GradientTransformation:

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(lambda g: jnp.moveaxis(g, column_axis, -1), updates)
    shapes = jax.tree.map(lambda g: g.shape, updates)
    updates = jax.tree.map(lambda g: orthogonalize(jnp.reshape(g, (fanin := math.prod(g.shape[:-1]), fanout := g.shape[-1]))) * jnp.sqrt(fanout / fanin).astype(g.dtype) * target_norm, updates)
    updates = jax.tree.map(lambda g, s: jnp.reshape(g, s), updates, shapes)
    updates = jax.tree.map(lambda g: jnp.moveaxis(g, -1, column_axis), updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def scale_by_embed_dualize(target_norm: float = 1.0, eps: float = 1e-8) -> base.GradientTransformation:
  # For both embedding and bias since the latter is a special case of embedding 
  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(lambda g: g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + eps) * jnp.sqrt(g.shape[-1]).astype(g.dtype) * target_norm, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def scale_by_scale_dualize(target_norm: float = 1.0) -> base.GradientTransformation:
  # It's jnp.sign() since the scale parameter results in linear transformation of a diagonal matrix.
  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree.map(lambda g: jnp.sign(g) * target_norm, updates)
    return updates, state

  return base.GradientTransformation(base.init_empty_state, update_fn)


def scale_by_momentum_dualize(config, image_shape):
  d = dict(config.model)
  if 'variant' in d:
    d |= big_vision.models.modula_vit.decode_variant(d.pop('variant'))
  _, p1, p2, c = image_shape
  vit = modula.compound.ViT(
    num_classes=config.num_classes,
    image_size=(p1, p2),
    patch_size=d["patch_size"],
    num_heads=d["num_heads"],
    d_embed=d["width"],
    d_mlp=d["mlp_dim"],
    d_query=d["width"] // d["num_heads"],
    d_value=d["width"] // d["num_heads"],
    num_blocks=d["depth"],
    channels=c,
    blocks_mass=d["depth"] * 2,)
  vit.dualize_norm()
  d = modula.compound.extract_target_norm(vit)
  param_labels = {}
  transforms = {}
  for label, (module, target_norm) in d.items():
    t = param_labels
    l = label.split('/')
    last = l.pop()
    for k in l:
        if k not in t:
            t[k] = {}
        t = t[k]
    t[last] = label
    if module == 'Linear':
        # Only QKV matrices need special handling since they have shape (d_embed, num_heads, d_query)
        if l[-1] in ('key', 'query', 'value'):
            t = scale_by_linear_dualize(column_axis=0, target_norm=target_norm)
        else:
            # Everything else happens to be fine with the last axis as the fan-out:
            # l[-1] == 'out': shape (num_heads, d_query, d_embed)
            # l[-1] == 'embedding': shape (p1, p2, c, d_embed)
            # others: standard shape (fanin, fanout)
            t = scale_by_linear_dualize(column_axis=-1, target_norm=target_norm)
    elif module == 'Bias':
        t = scale_by_embed_dualize(target_norm=target_norm)
    elif module == 'Scale':
        t = scale_by_scale_dualize(target_norm=target_norm)
    else:
        raise ValueError(f"Unsupported module: {module}")
    transforms[label] = t
  momentum = config.get("optax", {}).get('momentum', 0.95)
  dualize = optax.partition(transforms, param_labels)
  return [optax.ema(decay=momentum), dualize]
