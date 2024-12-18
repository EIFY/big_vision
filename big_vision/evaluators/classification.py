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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import

import functools

import big_vision.datasets.core as ds_core
import big_vision.input_pipeline as input_pipeline
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import jax
import jax.numpy as jnp

import torch
import dlpack


def number_correct(output, target, topk=(1,)):
    """Computes the top-k correct predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, target = target.topk(1, 1, True, True)
        target = target.squeeze(dim=1)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).sum(0, keepdim=True)
            res.append(correct_k)
        return res


class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size, loss_name,
               cache_final=True, cache_raw=False, prefetch=1,
               label_key='labels', *, devices):
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), pp_fn, batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(self.ds, devices, prefetch)
    self.label_key = label_key

  def run(self, model, criterion, accum_freq):
    """Computes all metrics."""
    ncorrect, loss, nseen = 0, 0, 0
    with torch.no_grad():
      torch.cuda.empty_cache()
      for _, batch in zip(range(self.steps), self.data_iter):
        target, mask = batch.pop(self.label_key), batch.pop('_mask')
        images = batch["image"]
        images, target = torch.from_dlpack(dlpack.asdlpack(images)), torch.from_dlpack(dlpack.asdlpack(target))
        images = images.permute(0, 3, 1, 2)
        mask = torch.from_dlpack(dlpack.asdlpack(mask))
        # Ignore the entries with all zero labels for evaluation.
        mask = mask.float() * target.max(dim=1)[0]
        images = images[mask > 0.]
        target = target[mask > 0.]

        for img, trt in zip(images.chunk(accum_freq), target.chunk(accum_freq)):
            # compute output
            output = model(img)
            size = img.size(0)
            loss += criterion(output, trt).item() * size
            ncorr = number_correct(output, trt)
            ncorrect += ncorr[0][0].item()
            nseen += size

    yield ('acc@1', ncorrect / nseen)
    yield ('loss', loss / nseen)
