import os
import sys
import os.path as osp
ACID_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
sys.path.insert(0,ACID_dir)

import glob

import argparse
flow_default = osp.join(ACID_dir, "train_data", "flow")
parser = argparse.ArgumentParser("Making training / testing splits...")
parser.add_argument("--flow_root", type=str, default=flow_default)
parser.add_argument("--no_split", action="store_true", default=False)
args = parser.parse_args()

flow_root = args.flow_root

all_npz = glob.glob(f"{flow_root}/*/*/*.npz")

print(f"In total {len(all_npz)} data points...")

def filename_to_id(fname):
    split_id, model_name, f = fname.split("/")[-3:]
    reset_id, frame_id = (int(x) for x in os.path.splitext(f)[0].split('_'))
    return split_id, model_name, reset_id, frame_id

from collections import defaultdict

total_files = defaultdict(lambda : defaultdict(lambda : []))
for fname in all_npz:
    split_id, model_name, reset_id, frame_id = filename_to_id(fname)
    total_files[(split_id, model_name)][reset_id].append(frame_id)

total_files = dict(total_files)
for k,v in total_files.items():
    total_files[k] = dict(v)
import pickle
if args.no_split:
    train = total_files
    test = total_files
else:
    train = {}
    test = {}
    for k,v in total_files.items():
        split_id, model_name = k
        if "teddy" in model_name:
            test[k] = v
        else:
            train[k] = v

train_total = []
for k,v in train.items():
    for x, u in v.items():            
        for y in u:
            train_total.append((*k, x, y))
print(f"training data points: {len(train_total)}")
test_total = []
for k,v in test.items():
    for x, u in v.items():            
        for y in u:
            test_total.append((*k, x, y))
print(f"testing data points: {len(test_total)}")
    
with open(f"{flow_root}/train.pkl", "wb") as fp:
    pickle.dump(train_total, fp)
with open(f"{flow_root}/test.pkl", "wb") as fp:
    pickle.dump(test_total, fp)