import argparse
import json
from glob import glob
import copy
from pathlib import Path
import os

import paddle as P
import numpy as np
import tqdm

from external.labeling_segment.modules.model import Model
from external.labeling_segment.dataset import Dataset
from utils.build_dataloader import build_dataloader

# import optimizer as opt

for k in filter(lambda k: k.startswith('FLAGS_'), os.environ.keys()):
    print(k, os.environ[k])

parser = argparse.ArgumentParser('Trainer')
parser.add_argument('-c', '--config_file', type=str, required=True)
parser.add_argument('-d', '--dataset_path', type=str, required=True)
parser.add_argument('-g', '--gpu_id', type=int, default=0)
parser.add_argument('-o', '--output_file', default='a.pdparams')
parser.add_argument('-bs', '--batch_size', type=int, default=4)
args = parser.parse_args()

label_path = f'{args.dataset_path}/train/label/'
image_path = f'{args.dataset_path}/train/images/'

n_inst = len(glob(f"{image_path}/*"))

reload = False

valid_label_path = f'{args.dataset_path}/valid/label/'
valid_image_path = f'{args.dataset_path}/valid/images/'

with open(args.config_file, "r") as fd:
    config = json.load(fd)

eval_config = config['eval']
model_config = config['architecture']

eval_config['dataset']['data_path'] = label_path
eval_config['dataset']['image_path'] = image_path
eval_config['dataset']['max_seqlen'] = model_config['embedding']['max_seqlen']

train_place = valid_place = P.set_device(
    'cpu' if args.gpu_id < 0 else f'gpu:{args.gpu_id}'
)

train_dataset = Dataset(
    eval_config['dataset'],
    eval_config['feed_names'],
    False)
train_loader = build_dataloader(
    config['eval'], train_dataset, 'Train',
    train_place, args.batch_size, False)

valid_config_ds = copy.deepcopy(eval_config)
valid_config_ds['dataset']['data_path'] = valid_label_path
valid_config_ds['dataset']['image_path'] = valid_image_path
valid_config_ds['dataset']['max_seqlen'] = model_config['embedding']['max_seqlen']

valid_dataset = Dataset(
    valid_config_ds['dataset'],
    eval_config['feed_names'],
    False)

valid_loader = build_dataloader(
    config['eval'], valid_dataset, 'Eval',
    valid_place, args.batch_size, False)

max_epochs = 100
tolerance = 40
min_epochs = 50

since_best = 0
best_loss = np.inf

model = Model(model_config, eval_config['feed_names'])

loss = P.nn.CrossEntropyLoss()
opt = P.optimizer.Adamax(learning_rate=5e-5, parameters=model.parameters())

if reload and Path(args.output_file).exists():
    # TODO: load state dict of optimizer
    state_dict = P.load(args.output_file)
    model.state_dict = state_dict

# print(len(train_dataset))
for epoch_idx in range(max_epochs):
    # print(f'memory allocated (MB): {P.device.cuda.memory_allocated(train_place) / (1024*1024)}')
    model.train()
    for x in tqdm.tqdm(train_loader):
        # Clear grad.
        opt.clear_grad()

        # Train model.
        # print(f'train bs: {[y.shape for y in x]}')
        res = model(*x, feed_names=eval_config['feed_names'])

        # Update loss.
        step_loss = loss(res['scores'], res['label_prim'])
        step_loss.backward()
        opt.step()

    # Validate.
    losses = []
    model.eval()
    for x in tqdm.tqdm(valid_loader):
        # print(f'valid bs: {[y.shape for y in x]}')
        out = model(*x, feed_names=eval_config['feed_names'])
        losses.append(loss(out['scores'], out['label_prim']))

    cur_loss = sum(losses) / len(losses)
    print("Valid loss: ", cur_loss)

    # Early stopping tests.
    if cur_loss < best_loss:
        since_best = 0
        best_loss = cur_loss
        print("Saved!")
        P.save(model.state_dict(), args.output_file)
    else:
        since_best += 1
    if epoch_idx >= min_epochs and since_best >= tolerance:
        print('leaving')
        break
