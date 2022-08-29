""" build_dataloader"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import signal
import random

import numpy as np
import paddle

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import copy
import paddle as P
from paddle.io import DataLoader, BatchSampler, DistributedBatchSampler
import paddle.distributed as dist

__all__ = ['build_dataloader']


def term_mp(sig_num, frame):
    """ kill all child processes
    """
    pid = os.getpid()
    pgid = os.getpgid(os.getpid())
    print("main proc {} exit, kill process group " "{}".format(pid, pgid))
    os.killpg(pgid, signal.SIGKILL)


signal.signal(signal.SIGINT, term_mp)
signal.signal(signal.SIGTERM, term_mp)

def build_dataloader(config, dataset, mode, device, batch_size=1, distributed=False):
    """ build_dataloader """
    config = copy.deepcopy(config)

    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."

    loader_config = config['loader']
    collect_batch = loader_config['collect_batch']
    num_workers = loader_config['num_workers']
    shuffle = loader_config.get('shuffle', False)
    drop_last = loader_config.get('drop_last', True)
    batch_sampler = None

    def collate_fn(x):
        return x[0]

    if collect_batch:
        # batch_size = loader_config.get('batch_size_per_card', batch_size)
        if distributed:
            #Distribute data to multiple cards
            batch_sampler = DistributedBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
        else:
            #Distribute data to single card
            batch_sampler = BatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last)
    elif distributed:
        batch_sampler = DistributedBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        places=device,
        timeout=60,
        num_workers=num_workers,
        return_list=True,
        shuffle=shuffle,
        drop_last=drop_last
    )

    # print(dir(data_loader))
    # print(data_loader.batch_size)
    # print(dir(data_loader.batch_sampler))
    # for bsi in data_loader.batch_sampler:
    #     print(bsi)
    #     break
    # for bs in data_loader:
    #     print(len(bs[0]))
    #     break
    # exit(-1)

    return data_loader
