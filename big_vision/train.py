# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training loop example.

This is a basic variant of a training loop, good starting point for fancy ones.
"""
# pylint: disable=consider-using-from-import
# pylint: disable=logging-fstring-interpolation

import functools
import importlib
import multiprocessing.pool
import math
import os

from absl import app
from absl import flags
from absl import logging
import big_vision.evaluators.common as eval_common
import big_vision.input_pipeline as input_pipeline
import big_vision.optax as bv_optax
import big_vision.sharding as bv_sharding
import big_vision.utils as u
from clu import parameter_overview
import flax.linen as nn
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.experimental.array_serialization import serialization as array_serial
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from ml_collections import config_flags
import numpy as np
import optax
import tensorflow as tf

from tensorflow.io import gfile

import torch
import dlpack
from big_vision.simple_vit import SimpleVisionTransformer

import wandb

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")
flags.DEFINE_string("name", default=None, help="Name of the run.")


# Adds jax flags to the program.
jax.config.parse_flags_with_absl()
# Transfer guard will fail the program whenever that data between a host and
# a device is transferred implicitly. This often catches subtle bugs that
# cause slowdowns and memory fragmentation. Explicit transfers are done
# with jax.device_put and jax.device_get.
# jax.config.update("jax_transfer_guard", "disallow")
# Fixes design flaw in jax.random that may cause unnecessary d2d comms.
jax.config.update("jax_threefry_partitionable", True)


# "(...)/python3.10/site-packages/torch/_inductor/compile_fx.py:140: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance."
torch.set_float32_matmul_precision('high')


NamedSharding = jax.sharding.NamedSharding
P = jax.sharding.PartitionSpec


def main(argv):
  del argv

  try:
    jax.distributed.initialize()
  except ValueError as e:
    logging.warning('Could not initialize distributed environment: %s', e)

  # Make sure TF does not touch GPUs.
  tf.config.set_visible_devices([], "GPU")

  config = flags.FLAGS.config

################################################################################
#                                                                              #
#                                Set up logging                                #
#                                                                              #
################################################################################

  # Set up work directory and print welcome message.
  workdir = flags.FLAGS.workdir
  logging.info(
      f"\u001b[33mHello from process {jax.process_index()} holding "
      f"{jax.local_device_count()}/{jax.device_count()} devices and "
      f"writing to workdir {workdir}.\u001b[0m")

  if workdir:  # Always create if requested, even if we may not write into it.
    gfile.makedirs(workdir)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # Here we register preprocessing ops from modules listed on `pp_modules`.
  for m in config.get("pp_modules", ["ops_general", "ops_image", "ops_text"]):
    importlib.import_module(f"big_vision.pp.{m}")

  # Setup up logging and experiment manager.
  xid, wid = -1, -1
  fillin = lambda s: s
  def info(s, *a):
    logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)
  def write_note(note):
    if jax.process_index() == 0:
      info("%s", note)

  mw = u.BigVisionMetricWriter(xid, wid, workdir, config)

  # Allow for things like timings as early as possible!
  u.chrono.inform(measure=mw.measure, write_note=write_note)

  if flags.FLAGS.name:
    wandb.init(
      project="mup-vit",
      name=flags.FLAGS.name,
      id=flags.FLAGS.name,
      tags=[],
      resume='auto',
      config=vars(config),
  )

################################################################################
#                                                                              #
#                                Set up Mesh                                   #
#                                                                              #
################################################################################

  # We rely on jax mesh_utils to organize devices, such that communication
  # speed is the fastest for the last dimension, second fastest for the
  # penultimate dimension, etc.
  config_mesh = config.get("mesh", [("data", jax.device_count())])

  # Sharding rules with default
  sharding_rules = config.get("sharding_rules", [("act_batch", "data")])

  mesh_axes, mesh_size = tuple(zip(*config_mesh))

  # Because jax.utils do not support `-1` shape size.
  mesh_size = np.array(jax.devices()).reshape(mesh_size).shape

  device_mesh = mesh_utils.create_device_mesh(mesh_size)

  # Consistent device order is important to ensure correctness of various train
  # loop components, such as input pipeline, update step, evaluators. The
  # order presribed by the `devices_flat` variable should be used throughout
  # the program.
  devices_flat = device_mesh.flatten()

################################################################################
#                                                                              #
#                                Input Pipeline                                #
#                                                                              #
################################################################################

  write_note("Initializing train dataset...")
  batch_size = config.input.batch_size
  if batch_size % jax.device_count() != 0:
    raise ValueError(f"Batch size ({batch_size}) must "
                     f"be divisible by device number ({jax.device_count()})")
  info("Global batch size %d on %d hosts results in %d local batch size. With "
       "%d dev per host (%d dev total), that's a %d per-device batch size.",
       batch_size, jax.process_count(), batch_size // jax.process_count(),
       jax.local_device_count(), jax.device_count(),
       batch_size // jax.device_count())

  train_ds, ntrain_img = input_pipeline.training(config.input)

  total_steps = u.steps("total", config, ntrain_img, batch_size)
  def get_steps(name, default=ValueError, cfg=config):
    return u.steps(name, cfg, ntrain_img, batch_size, total_steps, default)

  u.chrono.inform(total_steps=total_steps, global_bs=batch_size,
                  steps_per_epoch=ntrain_img / batch_size)

  info("Running for %d steps, that means %f epochs",
       total_steps, total_steps * batch_size / ntrain_img)

  # Start input pipeline as early as possible.
  n_prefetch = config.get("prefetch_to_device", 1)
  train_iter = input_pipeline.start_global(train_ds, devices_flat, n_prefetch)

################################################################################
#                                                                              #
#                           Create Model & Optimizer                           #
#                                                                              #
################################################################################

  write_note("Creating model...")

  model = SimpleVisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=6,
        hidden_dim=384,
        mlp_dim=1536,
    ).cuda()

  def weight_decay_param(n, p):
    if p.ndim >= 2 and n.endswith('weight'):
      print('Weight decay for:', n)
      return True
    else:
      return False

  wd_params = [p for n, p in model.named_parameters() if weight_decay_param(n, p) and p.requires_grad]
  non_wd_params = [p for n, p in model.named_parameters() if not weight_decay_param(n, p) and p.requires_grad]

  criterion = torch.nn.CrossEntropyLoss().cuda()
  optimizer = torch.optim.AdamW(
      [
          {"params": wd_params, "weight_decay": config.wd / config.lr},
          {"params": non_wd_params, "weight_decay": 0.},
      ],
      lr=config.lr,
  )

  warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step / config.schedule['warmup_steps'])
  cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - config.schedule['warmup_steps'])
  scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], [config.schedule['warmup_steps']])
  first_step = 0

################################################################################
#                                                                              #
#                               Load Checkpoint                                #
#                                                                              #
################################################################################

  if workdir:
    filename = os.path.join(workdir, "checkpoint.pth.tar")
    if os.path.isfile(filename):
      write_note(f"Resuming training from checkpoint {filename}...")
      checkpoint = torch.load(filename, weights_only=True)
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      first_step = checkpoint['step']

  original_model = model
  model = torch.compile(original_model)

  rng = jax.random.PRNGKey(u.put_cpu(config.get("seed", 0)))
  rng, rng_init = jax.random.split(rng)  # rng_init unused here but we want to keep the pseudorandom numbers the same as the main branch.

  rng, rng_loop = jax.random.split(rng, 2)
  del rng  # not used anymore, so delete it.

################################################################################
#                                                                              #
#                                 Mix-up Step                                  #
#                                                                              #
################################################################################

  def mixup_fn(step, rng, batch):
    images, labels = batch["image"], batch["labels"]
    rng = jax.random.fold_in(rng, step)
    f = u.get_mixup(rng, config.mixup.p)
    rng, (images, labels), _ = f(images, labels)
    return images, labels

  eval_fns = {
      "predict": None,
      "loss": None,
  }

  # Only initialize evaluators when they are first needed.
  @functools.lru_cache(maxsize=None)
  def evaluators():
    return eval_common.from_config(
        config, eval_fns,
        lambda s: write_note(f"Init evaluator: {s}…\n{u.chrono.note}"),
        lambda key, cfg: get_steps(key, default=None, cfg=cfg),
        devices_flat,
    )

  u.chrono.inform(first_step=first_step)

  # Note that training can be pre-empted during the final evaluation (i.e.
  # just after the final checkpoint has been written to disc), in which case we
  # want to run the evals.
  if first_step in (total_steps, 0):
    write_note("Running initial or final evals...")
    mw.step_start(first_step)
    for (name, evaluator, _, prefix) in evaluators():
      if config.evals[name].get("skip_first") and first_step != total_steps:
        continue
      write_note(f"{name} evaluation...\n{u.chrono.note}")
      with u.chrono.log_timing(f"z/secs/eval/{name}"):
        for key, value in evaluator.run(model, criterion, config.input.accum_freq):
          mw.measure(f"{prefix}{key}", jax.device_get(value))

################################################################################
#                                                                              #
#                                  Train Loop                                  #
#                                                                              #
################################################################################

  prof = None  # Keeps track of start/stop of profiler state.

  def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
      filename = os.path.join(path, filename)
      torch.save(state, filename)
      if is_best:
          shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))

  dev = jax.devices('gpu')[0]

  write_note("Starting training loop, compiling the first step...")
  for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
    mw.step_start(step)
    images, target = mixup_fn(step, jax.device_put(rng_loop, dev), batch)

    images, target = torch.from_dlpack(dlpack.asdlpack(images)), torch.from_dlpack(dlpack.asdlpack(target))
    images = images.transpose(1, 3)

    minibatch = zip(images.chunk(config.input.accum_freq), target.chunk(config.input.accum_freq))
    step_loss = 0.0

    for img, trt in minibatch:
        # compute output
        output = model(img)
        loss = criterion(output, trt)
        step_loss += loss.item()

        # compute gradient
        (loss / config.input.accum_freq).backward()

    step_loss /= config.input.accum_freq
    l2_grads = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()
    optimizer.zero_grad()

    # Report training progress
    if (u.itstime(step, get_steps("log_training"), total_steps, host=0)
        or u.chrono.warmup and jax.process_index() == 0):

      with torch.no_grad():
        l2_params = sum(p.square().sum().item() for _, p in model.named_parameters())
        mw.measure("l2_params", math.sqrt(l2_params))

      mw.measure("train/loss", step_loss)
      mw.measure("l2_grads", l2_grads.item())
      mw.measure("lr", scheduler.get_last_lr()[0])
      u.chrono.tick(step)

    scheduler.step()

    # Checkpoint saving
    keep_ckpt_steps = get_steps("keep_ckpt", None) or total_steps
    if workdir and (
        (keep := u.itstime(step, keep_ckpt_steps, total_steps, first=False))
        or u.itstime(step, get_steps("ckpt", None), total_steps, first=True)
    ):
      save_checkpoint({
          'step': step,
          'state_dict': original_model.state_dict(),
          'optimizer' : optimizer.state_dict(),
          'scheduler' : scheduler.state_dict()
      }, False, workdir)

    for (name, evaluator, log_steps, prefix) in evaluators():
      if u.itstime(step, log_steps, total_steps, first=False, last=True):
        u.chrono.tick(step)  # Record things like epoch number, core hours etc.
        write_note(f"{name} evaluation...\n{u.chrono.note}")
        with u.chrono.log_timing(f"z/secs/eval/{name}"):
          for key, value in evaluator.run(model, criterion, config.input.accum_freq):
            mw.measure(f"{prefix}{key}", jax.device_get(value))
    mw.step_end()

  # Always give a chance to stop the profiler, no matter how things ended.
  # TODO: can we also do this when dying of an exception like OOM?
  if jax.process_index() == 0 and prof is not None:
    u.startstop_prof(prof)

  # Last note needs to happen before the pool's closed =)
  write_note(f"Done!\n{u.chrono.note}")

  pool.close()
  pool.join()
  mw.close()

  # Make sure all hosts stay up until the end of main.
  u.sync()

  u.maybe_cleanup_workdir(workdir, flags.FLAGS.cleanup, info)


if __name__ == "__main__":
  app.run(main)
