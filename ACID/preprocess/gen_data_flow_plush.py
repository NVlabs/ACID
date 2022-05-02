import numpy as np
import os
import time, datetime
import sys
import os.path as osp
ACID_dir = osp.dirname(osp.dirname(osp.realpath(__file__)))
sys.path.insert(0,ACID_dir)

import json

from src.utils import plushsim_util 
from src.utils import common_util 
import glob
import tqdm
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser("Training Flow Data Generation")
data_plush_default = osp.join(ACID_dir, "data_plush")
flow_default = osp.join(ACID_dir, "train_data", "flow")
parser.add_argument("--data_root", type=str, default=data_plush_default)
parser.add_argument("--save_root", type=str, default=flow_default)
args = parser.parse_args()

data_root = args.data_root
save_root = args.save_root

scene_range = plushsim_util.SCENE_RANGE.copy()
to_range = np.array([[-1.1,-1.1,-1.1],[1.1,1.1,1.1]]) * 0.5
class_to_std = {
    'teddy':0.12,
    'elephant':0.15,
    'octopus':0.12,
    'rabbit':0.08,
    'dog':0.08,
    'snake':0.04,
}
def export_train_data(data_id):
    # try:
    # load action info
    split_id, model_category, model_name, reset_id, interaction_id = data_id
    grasp_loc, target_loc, f1, _, f2 = plushsim_util.get_action_info(model_category, model_name, split_id, reset_id, interaction_id, data_root)
    # get observations 
    obj_pts1, env_pts1 = plushsim_util.get_scene_partial_pointcloud(
                    model_category, model_name, split_id, reset_id, f1, data_root)
    obj_pts1=common_util.subsample_points(
        common_util.transform_points(obj_pts1, scene_range, to_range), resolution=0.005, return_index=False)
    env_pts1=common_util.subsample_points(
        common_util.transform_points(env_pts1, scene_range, to_range), resolution=0.020, return_index=False)
    # calculate flow
    sim_pts1, _, loc,_,_= plushsim_util.get_object_full_points(
                    model_category, model_name, split_id, reset_id, f1, data_root)
    sim_pts2, _,_,_,_= plushsim_util.get_object_full_points(
                    model_category, model_name, split_id, reset_id, f2, data_root)
    sim_pts1=common_util.transform_points(sim_pts1, scene_range, to_range)
    sim_pts2=common_util.transform_points(sim_pts2, scene_range, to_range)
    sim_pts_flow = sim_pts2 - sim_pts1

    # sample occupancy
    center =common_util.transform_points(loc, scene_range, to_range)[0]
    pts, occ, pt_class = plushsim_util.sample_occupancies(sim_pts1, center, 
                            std=class_to_std[model_category],sample_scheme='object')
    # get implicit flows
    flow = sim_pts_flow[pt_class]
    # save
    kwargs = {'sim_pts':sim_pts1.astype(np.float16), 
            'obj_pcloud_obs':obj_pts1.astype(np.float16),
            'env_pcloud':env_pts1.astype(np.float16), 
            'pts':pts.astype(np.float16), 
            'occ':np.packbits(occ), 
            'ind':pt_class.astype(np.uint16),
            'flow':flow.astype(np.float16),
            'start_frame':f1,
            'end_frame':f2,
            'grasp_loc':grasp_loc,
            'target_loc': target_loc}
    model_dir = os.path.join(save_root, f"{split_id}", f"{model_name}")
    save_path = os.path.join(model_dir, f"{reset_id:03d}_{interaction_id:03d}.npz")
    np.savez_compressed(save_path, **kwargs)

def get_all_data_points_flow(data_root):
    good_interactions = glob.glob(f"{data_root}/*/*/*/info/good_interactions.json")
    good_ints = []
    for g in tqdm.tqdm(good_interactions):
        split_id, model_category, model_name = g.split('/')[-5:-2]
        model_dir = os.path.join(save_root, f"{split_id}", f"{model_name}")
        os.makedirs(model_dir, exist_ok=True)
        model_dir = plushsim_util.get_model_dir(data_root, split_id, model_category, model_name)
        with open(g, 'r') as fp:
            good_ones = json.load(fp)
        for k,v in good_ones.items():
            reset_id = int(k)
            for int_id in v:
                good_ints.append((split_id, model_category, model_name, reset_id, int_id))
    return good_ints

good_ints = get_all_data_points_flow(data_root)#[:100]

start_time = time.time()
with Pool(40) as p:
    for _ in tqdm.tqdm(p.imap_unordered(export_train_data, good_ints), total=len(good_ints)):
        pass

end_time = time.time()
from datetime import timedelta
time_str = str(timedelta(seconds=end_time - start_time))
print(f'Total processing takes: {time_str}')