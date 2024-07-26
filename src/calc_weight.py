import json
import os
import pandas as pd
import numpy as np
from collections import Counter
from scipy.ndimage import convolve1d
from utils_lds import get_lds_kernel_window
import os
import numpy as np
import scipy.io as scio
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def padding(idxs, Ns):
    min_idx, max_idx = min(idxs), max(idxs)
    idxs_pad = range(min_idx, max_idx + 1)
    retD = list(set(idxs_pad).difference(set(idxs)))
    retD_idxs = [i - min_idx for i in retD]
    [Ns.insert(i, 0) for i in retD_idxs]
    return idxs_pad, Ns

def get_bin_idx(labels):
    Ns = []
    idxs = []
    b = Counter(np.array(labels))
    for k,v in b.items():
        Ns.append(v)
        idxs.append(k)
    Ns = [i for _, i in sorted(zip(idxs, Ns))]
    idxs = sorted(idxs)
    return idxs, Ns

if __name__=='__main__':
    f = open('./data_json.json', 'r')
    content = f.read()
    data_dict = json.loads(content)
    keys = data_dict.keys()
    whole_data = {}
    whole_weights = {}
    # bin size = 20
    for target in ['BUAA', 'UBFC', 'V4V', 'PURE', 'VIPL']:
        temp_lst = []
        for key in keys:
            if key == target:
                continue
            data_lst = data_dict[key]
            data_lst = [i for i in data_lst if i>0]
            temp_lst.extend(data_lst)
        whole_data[target] = temp_lst
    f.close()
    for key in whole_data.keys():
        print('='*50 + key + '='*50)
        labels = whole_data[key]
        print(max(labels))

        bin_index_per_label, emp_label_dist = get_bin_idx(labels)
        bin_index_per_label, emp_label_dist = padding(bin_index_per_label, emp_label_dist)
        emp_label_dist  = [i+1 for i in emp_label_dist]
        print(bin_index_per_label)
        #print(emp_label_dist)
        lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
        eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')
        weights = [1/x for x in eff_label_dist]
        #print(weights)
        whole_weights[key] = weights

with open('./weights.json', 'w') as json_file:
    json.dump(whole_weights, json_file)
