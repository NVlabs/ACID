import os
import sys
import glob
import tqdm
import random
import argparse
import numpy as np
import os.path as osp
import time
from multiprocessing import Pool
ACID_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
sys.path.insert(0,ACID_dir)

parser = argparse.ArgumentParser("Training Contrastive Pair Data Generation")

data_plush_default = osp.join(ACID_dir, "data_plush")
meta_default = osp.join(ACID_dir, "data_plush", "metadata")
flow_default = osp.join(ACID_dir, "train_data", "flow")
pair_default = osp.join(ACID_dir, "train_data", "pair")
parser.add_argument("--data_root", type=str, default=data_plush_default)
parser.add_argument("--meta_root", type=str, default=meta_default)
parser.add_argument("--flow_root", type=str, default=flow_default)
parser.add_argument("--save_root", type=str, default=pair_default)
args = parser.parse_args()

data_root = args.data_root
flow_root = args.flow_root
save_root = args.save_root
meta_root = args.meta_root
os.makedirs(save_root, exist_ok=True)


def using_complex(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind = np.unique(b, return_index=True)
    b = np.zeros_like(a) + 256
    np.put(b, ind, a.flat[ind])
    return b

def process(pair, num_samples=320, keep=80):
    split_id, model_name, f,p = pair
    src_file = np.load(f"{flow_root}/{split_id}/{model_name}/{f}")
    tgt_file = np.load(f"{flow_root}/{split_id}/{model_name}/{p}")

    src_inds = src_file['ind']
    tgt_inds = tgt_file['ind']
    src_inds = np.tile(src_inds, (num_samples,1)).T            
    tgt_samples = np.random.randint(0, high=len(tgt_inds) - 1, size=(len(src_inds), num_samples))
    tgt_samples_inds = tgt_inds[tgt_samples]

    dists = dist_matrix[src_inds.reshape(-1), tgt_samples_inds.reshape(-1)].reshape(*src_inds.shape)
    dists_unique = using_complex(dists)
    idx = np.argsort(dists_unique, axis=-1)
    dists_sorted = np.take_along_axis(dists, idx, axis=-1).astype(np.uint8)[:,:keep]

    tgt_samples_sorted = np.take_along_axis(tgt_samples, idx, axis=-1)[:,:keep]

    if tgt_samples_sorted.max() <= np.iinfo(np.uint16).max:
        tgt_samples_sorted = tgt_samples_sorted.astype(np.uint16)
    else:
        tgt_samples_sorted = tgt_samples_sorted.astype(np.uint32)

    results = {"target_file":p, "dists":dists_sorted, "inds":tgt_samples_sorted}
    np.savez_compressed(os.path.join(save_dir, f"pair_{f}"), **results)

def export_pair_data(data_id):
    split_id, model_name = data_id
    all_files = all_geoms[data_id]
    print(split_id, model_name)
    global dist_matrix 
    dist_matrix = np.load(f'{meta_root}/{split_id}/{model_name}_dist.npz')['arr_0']
    global save_dir
    save_dir = os.path.join(save_root, split_id, model_name)
    os.makedirs(save_dir, exist_ok=True)
    pairs = [ (split_id, model_name, f,random.choice(all_files)) for f in all_files ]

    start_time = time.time()
    with Pool(10) as p:
        for _ in tqdm.tqdm(p.imap_unordered(process, pairs), total=len(all_files)):
            pass

    end_time = time.time()
    from datetime import timedelta
    time_str = str(timedelta(seconds=end_time - start_time))
    print(f'Total processing takes: {time_str}')

if __name__ == '__main__':
    from collections import defaultdict
    global all_geoms
    all_geoms = defaultdict(lambda: [])

    for g in glob.glob(f"{flow_root}/*/*/*"):
        split_id, model_name, file_name = g.split('/')[-3:]
        all_geoms[(split_id, model_name)].append(file_name) 

    for k in all_geoms.keys():
        export_pair_data(k)
