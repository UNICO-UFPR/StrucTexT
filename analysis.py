import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from glob import glob

def draw_conf(mtr, its, fname='conf.png'):
    n_its = len(its)
    conf_img = np.zeros(((n_its + 1)*50, (n_its + 1)*50, 3), dtype=np.uint8)
    bp = Image.new('RGB', conf_img.shape[:-1])
    drawer = ImageDraw.Draw(bp)

    for i in range(n_its):
        idx_x = i*50
        total = sum(mtr[i])
        if total != 0:
            for j in range(n_its):
                idx_y = j*50
                if i == j:
                    conf_img[idx_x:50 + idx_x, idx_y:50 + idx_y, 2] += int((mtr[i][j]*255)/(total + mtr[i][j]))
                else:
                    conf_img[idx_x:50 + idx_x, idx_y:50 + idx_y, 0] += int((mtr[i][j]*255)/(total + mtr[i][j]))

    img = Image.fromarray(conf_img,mode='RGB')
    bp.paste(img, box=(50, 50))
    for i in range(n_its):
        idx_x = i*50 + 65
        drawer.multiline_text((0, idx_x),its[i].replace('-', '\n'))

        for j in range(n_its):
            idx_y = j*50 + 65
            drawer.text((idx_y, idx_x),str(int(mtr[i][j])))

    for j in range(n_its):
        idx_y = j*50 + 65
        drawer.multiline_text((idx_y, 0),its[j].replace('-', '\n'))

    bp.save(fname)

#f = "./preds.txt"
input_path = sys.argv[1]
fs = glob(f"{input_path}/*")

#with open(f, "r") as fd:
#    lines = fd.readlines()
dct = {'other': 0, 'header': 1, 'question': 2, 'answer': 3}
dct = {'header-nome': 0, 'nomeMae': 1, 'naturalidade': 2, 'header-obs': 3, 'serial?': 4, 'header-datanasc': 5, 'header-orgaoexp': 6, 'assin': 7, 'tag': 8, 'header-rh': 9, 'nomePai': 10, 'orgaoEmissor': 11, 'cod-sec': 12, 'header-naturalidade': 13, 'header-filiacao': 14, 'dataNascimento': 15, 'nome': 16, 'header-assin': 17}

matrix = np.zeros((18, 18))
hit = miss = 0
ls = {}

for f in fs:
    with open(f, "r") as fd:
        ls[f] = [x.strip().split(' ') for x in fd.readlines()]

num_erratic_samples = 0
for f in fs:
    is_erratic = False
    for l in ls[f]:
        if l[0] == l[2]:
            hit += 1
        else:
            miss += 1
            is_erratic = True
        matrix[dct[l[0]]][dct[l[2]]] += 1
    num_erratic_samples += 1 if is_erratic else 0


#for i in range(0, len(lines)//2):
#    for j in range(0, (len(lines[2*i]) - 1)//3):
#        if lines[2*i + 1][3*j + 1] == lines[2*i][3*j + 1]:
#            hit += 1
#        else:
#            miss += 1
#        matrix[int(lines[2*i][3*j + 1])][int(lines[2*i + 1][3*j + 1])] += 1
# print(matrix)

acc = 100 * hit / (miss + hit)
num_errs = 100 * num_erratic_samples / len(fs)
model_name = os.path.basename(input_path)
draw_conf(matrix, list(dct.keys()), f'./{model_name}.png')

print(f'{model_name}', f'{acc}', f'{num_errs}')
