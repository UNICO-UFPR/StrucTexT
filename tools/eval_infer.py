""" eval_infer.py """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import json
import logging
import argparse
import functools
import importlib
import numpy as np
import paddle as P
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from utils.utility import add_arguments, print_arguments
from utils.build_dataloader import build_dataloader
from utils.metrics import build_metric
from external.evaler import Evaler

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)

# sysconf
# base
parser = argparse.ArgumentParser('launch for eval')
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--task_type', type=str, required=True)
parser.add_argument('--label_path', type=str, required=True)
parser.add_argument('--image_path', type=str, required=True)
parser.add_argument('--weights_path', type=str, required=True)
parser.add_argument('--out_path', type=str, required=False)

args = parser.parse_args()
print_arguments(args)
config = json.loads(open(args.config_file).read())

ALL_MODULES = ['labeling_segment', 'labeling_token', 'linking']
if args.task_type not in ALL_MODULES:
    raise ValueError('Not valid task_type %s in %s' % (args.task_type, str(ALL_MODULES)))

# modules
model = importlib.import_module('external.' + args.task_type + '.modules.model')
dataset = importlib.import_module('external.' + args.task_type + '.dataset')

# package_name
Model = model.Model
Dataset = dataset.Dataset

def eval(config):
    """ eval """
    # program
    eval_config = config['eval']
    model_config = config['architecture']

    label_path = args.label_path
    image_path = args.image_path
    weights_path = args.weights_path
    out_path = Path(args.out_path)

    out_path.mkdir(exist_ok=True, parents=True)

    assert weights_path.endswith('.pdparams') and \
            os.path.isfile(weights_path), \
            'the weights_path %s is not existed!' % weights_path
    assert os.path.isdir(label_path), 'the label_dir %s is not existed!' % label_path
    assert os.path.isdir(image_path), 'the image_dir %s is not existed!' % image_path

    config['init_model'] = weights_path
    eval_config['dataset']['data_path'] = label_path
    eval_config['dataset']['image_path'] = image_path
    eval_config['dataset']['max_seqlen'] = model_config['embedding']['max_seqlen']
    config['eval'] = eval_config


    place = P.set_device('gpu:0')

    eval_dataset = Dataset(
        eval_config['dataset'],
        eval_config['feed_names'],
        False)

    eval_loader = build_dataloader(
        config['eval'],
        eval_dataset,
        'Eval',
        place, 1, False)
    #model
    model = Model(model_config, eval_config['feed_names'])
    para_dict = P.load(weights_path)
    model.set_dict(para_dict)

    model.eval()
    dct = {0: 'other', 1: 'header', 2: 'question', 3: 'answer'}
    dct_full = {0: "unknown", 1: "meta", 2: "info-nome", 3:"nome", 4:"info-filiacao", 5:"filiacao", 6: "info-datanasc",
            7: "data-nascimento", 8: "info-org", 9: "org", 10: "info-rh", 11: "fator-rh", 12: "info-naturalidade",
            13: "naturalidade", 14: "info-obs", 15: "info-asstitular", 16: "ass-titular", 17: "5-code", 18: "info-rg",
            19: "rg", 20: "info-cpf", 21: "cpf", 22: "info-doc", 23: "doc-origem", 24: "comarca", 25: "obs", 26: "info-dataexp",
            27: "data-expedicao", 28: "info-assdiretor", 29: "ass-diretor"}
    dct_full = {0: "header-nome", 1: "nomeMae", 2: "naturalidade", 3: "header-obs", 4: "serial?", 5: "header-datanasc", 6: "header-orgaoexp", 7: "assin", 8: "tag", 9: "header-rh", 10: "nomePai", 11: "orgaoEmissor", 12: "cod-sec", 13: "header-naturalidade", 14: "header-filiacao", 15: "dataNascimento", 16: "nome", 17: "header-assin"}


    i = 0
    right = wrong = 0
    for idx,X in enumerate(eval_loader):
        f = eval_dataset.data_list[idx][0].split('/')[-1] + '.txt'
        new_f = (out_path / f).as_posix()
        #print(X[6].tolist()[0])
        out = model(*X, feed_names=eval_config['feed_names'])#['logit'].tolist()
        labels = out['label'].tolist()
        logits = out['logit_prim'].tolist()[0]
        gts = out['label_prim'].tolist()[0]
        f = open(new_f, "w")
        f.writelines([dct_full[x] + " " + dct_full[y] + " " + dct_full[z] + "\n" for x,y,z in zip(labels, gts, logits)])
        f.close()
    return
    #metric
    eval_classes = build_metric(eval_config['metric'])

    #start
    logging.info('eval start...')
    eval = Evaler(config=config,
                  model=model,
                  data_loader=eval_loader,
                  eval_classes=eval_classes)

    eval.run()
    logging.info('eval end...')

#start to eval
eval(config)
