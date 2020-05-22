#!/usr/bin/env python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Run sotabench.com ImageNet benchmark."""

import os
import urllib.request

import numpy as np
from sotabencheval.image_classification import ImageNetEvaluator
from sotabencheval.utils import is_server
import torch
import torchvision as tv

import bit_pytorch.models as models
import bit_hyperrule


PAPER_RESULTS = {
    'BiT-M-R152x4': {
        'Top 1 Accuracy': 0.8539,
    },
}


def make_model(variant):
  print(f'Downloading and loading {variant}...', flush=True)

  # Create and load the model.
  model = models.KNOWN_MODELS[variant](head_size=1000)
  url = f'https://storage.googleapis.com/bit_models/{variant}-ILSVRC2012.npz'
  urllib.request.urlretrieve(url, f'{variant}.npz')
  model.load_from(np.load(f'{variant}.npz'))

  model = torch.nn.DataParallel(model)
  model = model.to(device='cuda')
  model.eval()
  return model


def make_data(batch_size):
  print('Preparing data...', flush=True)

  if is_server():
    datadir = './.data/vision/imagenet'
  else:  # local settings
    datadir = '/fastwork/data/ilsvrc2012'

  # Setup the input pipeline
  _, crop = bit_hyperrule.get_resolution_from_dataset('imagenet2012')
  input_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  # valid_set = tv.datasets.ImageFolder(os.path.join(datadir, 'val'), input_tx)
  valid_set = tv.datasets.ImageNet(datadir, split='val', transform=input_tx)

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=batch_size, shuffle=False,
      num_workers=8, pin_memory=True, drop_last=False)
  return valid_set, valid_loader


def get_img_id(data, i):
  image_name, _ = data.imgs[i]
  return image_name.split('/')[-1].replace('.JPEG', '')


def run_eval(model, data, loader, name):
  evaluator = ImageNetEvaluator(
      model_name=name,
      paper_arxiv_id='1912.11370',
      paper_results=PAPER_RESULTS.get(name),
  )

  with torch.no_grad():
    for i, (x, _) in enumerate(loader):
      print(f'\rEvaluating batch {i}/{len(loader)}', flush=True, end='')
      x = x.to(device='cuda', non_blocking=True)
      y = model(x).cpu().numpy()

      bs = loader.batch_size
      evaluator.add({
          get_img_id(data, i*bs + j): list(logits) for j, logits in enumerate(y)
      })

      if evaluator.cache_exists:
        break

  for k, v in evaluator.save().to_dict().items():
    print(f"{k}: {v}")


def main():
  for name, bs in [
      ('BiT-M-R152x4', 16),  # 7.9G ram, could try 32 on V100
      ('BiT-M-R152x2', 64),  # 9.xG ram, could try 96 on V100
      ('BiT-M-R50x1', 128),  # 8.3G ram, could try 192 on V100
  ]:
    data, loader = make_data(bs)
    model = make_model(name)
    run_eval(model, data, loader, name)


if __name__ == '__main__':
  os.system("nvidia-smi")  # Just to have it in the logs.
  main()
