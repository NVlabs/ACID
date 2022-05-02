import os
import glob
import json
import scipy
import itertools
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

from .common_util import *

########################################################################
# Some file getters
########################################################################
def get_model_dir(data_root, split_id, model_category, model_name):
    return f"{data_root}/{split_id}/{model_category}/{model_name}"

def get_interaction_info_file(data_root, split_id, model_category, model_name, reset_id):
    model_dir = get_model_dir(data_root, split_id, model_category, model_name)
    return f"{model_dir}/info/interaction_info_{reset_id:04d}.npz"

def get_geom_file(data_root, split_id, model_category, model_name, reset_id, frame_id):
    model_dir = get_model_dir(data_root, split_id, model_category, model_name)
    return f"{model_dir}/geom/{reset_id:04d}_{frame_id:06d}.npz"

def get_image_file_template(data_root, split_id, model_category, model_name, reset_id, frame_id):
    model_dir = get_model_dir(data_root, split_id, model_category, model_name)
    return f"{model_dir}/img/{{}}_{reset_id:04d}_{frame_id:06d}.{{}}"

def get_rgb(data_root, split_id, model_category, model_name, reset_id, frame_id):
    temp = get_image_file_template(data_root, split_id, model_category, model_name, reset_id, frame_id)
    return temp.format('rgb', 'jpg')

def get_depth(data_root, split_id, model_category, model_name, reset_id, frame_id):
    temp = get_image_file_template(data_root, split_id, model_category, model_name, reset_id, frame_id)
    return temp.format('depth', 'png')

def get_seg(data_root, split_id, model_category, model_name, reset_id, frame_id):
    temp = get_image_file_template(data_root, split_id, model_category, model_name, reset_id, frame_id)
    return temp.format('seg', 'jpg')

def get_flow_data_file(flow_root,split_id, model_id, reset_id, int_id):
    return f"{flow_root}/{split_id}/{model_id}/{reset_id:03d}_{int_id:03d}.npz"

def get_flow_pair_data_file(pair_root,split_id, model_id, reset_id, int_id):
    return f"{pair_root}/{split_id}/{model_id}/pair_{reset_id:03d}_{int_id:03d}.npz"

def get_geom_data_file(geom_root,split_id, model_id, reset_id, frame_id):
    return f"{geom_root}/{split_id}/{model_id}/{reset_id:03d}_{frame_id:06d}.npz"

def get_pair_data_file(pair_root,split_id, model_id, reset_id, frame_id):
    return f"{pair_root}/{split_id}/{model_id}/pair_{reset_id:03d}_{frame_id:06d}.npz"

# Getters for plan data
def get_plan_geom_file(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id):
    if sequence_id == 'gt':
        seq_str = sequence_id
    else:
        seq_str = f"{sequence_id:04d}"
    model_dir = get_model_dir(data_root, split_id, model_category, model_name)
    return f"{model_dir}/geom/{scenario_id:04d}_{seq_str}_{frame_id}.npz"

def get_plan_interaction_info_file(data_root, split_id, model_category, model_name, scenario_id, sequence_id):
    if sequence_id == 'gt':
        seq_str = sequence_id
    else:
        seq_str = f"{sequence_id:04d}"
    model_dir = get_model_dir(data_root, split_id, model_category, model_name)
    return f"{model_dir}/info/interaction_info_{scenario_id:04d}_{seq_str}.npz"

def get_plan_image_file_template(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id):
    if sequence_id == 'gt':
        seq_str = sequence_id
    else:
        seq_str = f"{sequence_id:04d}"
    model_dir = get_model_dir(data_root, split_id, model_category, model_name)
    return f"{model_dir}/img/{{}}_{scenario_id:04d}_{seq_str}_{frame_id}.{{}}"

def get_plan_rgb(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id):
    temp = get_plan_image_file_template(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id)
    return temp.format('rgb', 'jpg')

def get_plan_depth(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id):
    temp = get_plan_image_file_template(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id)
    return temp.format('depth', 'png')

def get_plan_seg(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id):
    temp = get_plan_image_file_template(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id)
    return temp.format('seg', 'jpg')

def get_plan_perf_file(data_root, split_id, model_category, model_name, scenario_id):
    model_dir = get_model_dir(data_root, split_id, model_category, model_name)
    return f"{model_dir}/info/perf_{scenario_id:04d}.npz"


########################################################################
# partial observation getter for full experiment
########################################################################
CAM_EXTR = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.6427898318479135, -0.766043895201295, -565.0], 
                     [0.0, 0.766047091387779, 0.6427871499290135, 550.0], [0.0, 0.0, 0.0, 1.0]])
CAM_INTR = np.array([[687.1868314210544, 0.0, 360.0], [0.0, 687.1868314210544, 360.0], [0.0, 0.0, 1.0]])
SCENE_RANGE = np.array([[-600, -600, -20], [600, 600, 380]])

def get_plan_scene_partial_pointcloud(
        model_category, model_name, split_id, scenario_id, sequence_id, frame_id, data_root):
    depth_img = get_plan_depth(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id)
    depth_img = np.array(Image.open(depth_img).convert(mode='I'))
    depth_vals = -np.array(depth_img).astype(float) / 1000.

    rgb_img = get_plan_rgb(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id)
    rgb_img = np.array(Image.open(rgb_img).convert(mode="RGB")).astype(float) / 255

    seg_img = get_plan_seg(data_root, split_id, model_category, model_name, scenario_id, sequence_id, frame_id)
    seg_img = np.array(Image.open(seg_img).convert('L')).squeeze()
    non_env = np.where(seg_img != 0)
    env = np.where(seg_img == 0)

    partial_points = project_depth_world_space(depth_vals, CAM_INTR, CAM_EXTR, keep_dim=True, project_factor=100.)
    partial_points_rgb = np.concatenate([partial_points, rgb_img], axis=-1)
    obj_pts = partial_points_rgb[non_env]
    env_pts = partial_points_rgb[env]
    return obj_pts, env_pts

def get_scene_partial_pointcloud(model_category, model_name, split_id, reset_id, frame_id, data_root):
    depth_img = get_depth(data_root, split_id, model_category, model_name, reset_id, frame_id)
    depth_img = np.array(Image.open(depth_img).convert(mode='I'))
    depth_vals = -np.array(depth_img).astype(float) / 1000.

    rgb_img = get_rgb(data_root, split_id, model_category, model_name, reset_id, frame_id)
    rgb_img = np.array(Image.open(rgb_img).convert(mode="RGB")).astype(float) / 255

    seg_img = get_seg(data_root, split_id, model_category, model_name, reset_id, frame_id)
    seg_img = np.array(Image.open(seg_img).convert('L')).squeeze()
    non_env = np.where(seg_img != 0)
    env = np.where(seg_img == 0)

    partial_points = project_depth_world_space(depth_vals, CAM_INTR, CAM_EXTR, keep_dim=True, project_factor=100.)
    partial_points_rgb = np.concatenate([partial_points, rgb_img], axis=-1)
    obj_pts = partial_points_rgb[non_env]
    env_pts = partial_points_rgb[env]
    return obj_pts, env_pts

def render_points(world_points, cam_extr=None, cam_intr=None, return_index=False, filter_in_cam=True):
    if cam_extr is None:
        cam_extr = CAM_EXTR
    if cam_intr is None:
        cam_intr = CAM_INTR
    cam_points = transform_points_world_to_cam(world_points, cam_extr) / 100.
    cam_pts_x = cam_points[:,0]
    cam_pts_y = cam_points[:,1]
    cam_pts_z = cam_points[:,2]
    cam_pts_x = -cam_pts_x / cam_pts_z * cam_intr[0,0] + cam_intr[1,2]
    cam_pts_y = cam_pts_y / cam_pts_z * cam_intr[1,1] + cam_intr[0,2]
    idx = np.rint(cam_pts_y / 6) * 1000 + np.rint(cam_pts_x / 6)
    val = np.stack([cam_pts_z, np.arange(len(cam_pts_x))]).T
    order = idx.argsort()
    idx = idx[order]
    val = val[order]
    grouped_pts = np.split(val, np.unique(idx, return_index=True)[1][1:])
    min_depth = np.array([p[p[:,0].argsort()][-1] for p in grouped_pts])
    min_idx = min_depth[:,-1].astype(int)
    if filter_in_cam:
        in_cam = np.where(np.logical_and(cam_pts_x > 0, cam_pts_y > 0))[0]
        min_idx = np.intersect1d(in_cam, min_idx, assume_unique=True)
    if return_index:
        return min_idx
    return world_points[min_idx]

########################################################################
# Get geometric state (full experiment)
########################################################################
def extract_full_points(path):
    geom_data = np.load(path)
    loc = geom_data['loc']
    w,x,y,z= geom_data['rot']
    rot = Rotation.from_quat(np.array([x,y,z,w]))
    scale = geom_data['scale']
    sim_pts = (rot.apply(geom_data['sim'] * scale)) + loc
    vis_pts = (rot.apply(geom_data['vis'] * scale)) + loc
    return sim_pts, vis_pts, loc, rot, scale

def get_object_full_points(model_category, model_name, split_id, reset_id, frame_id, data_root):
    path = get_geom_file(data_root, split_id, model_category, model_name, reset_id, frame_id)
    return extract_full_points(path)


def get_action_info(model_category, model_name, split_id, reset_id, interaction_id, data_root):
    obj_info = get_interaction_info_file(data_root, split_id, model_category, model_name, reset_id)
    int_info = np.load(obj_info)
    grasp_loc = np.array(int_info['grasp_points'][interaction_id])
    target_loc = np.array(int_info['target_points'][interaction_id])
    start_frame = int_info['start_frames'][interaction_id]
    release_frame = int_info['release_frames'][interaction_id]
    static_frame = int_info['static_frames'][interaction_id]
    return grasp_loc, target_loc, start_frame, release_frame, static_frame

########################################################################
# Get point-based supervision data for implicit functions (teddy toy example)
########################################################################
def sample_occupancies(full_pts, center, 
                        sample_scheme='gaussian',
                        num_pts = 100000, bound=0.55, 
                        std=0.1):
    if sample_scheme not in ['uniform', 'gaussian', 'object']:
        raise ValueError('Unsupported sampling scheme for occupancy')
    if sample_scheme == 'uniform':
        pts = np.random.rand(num_pts, 3)
        pts = 1.1 * (pts - 0.5)
    elif sample_scheme == 'object':
        displace = full_pts[np.random.randint(full_pts.shape[0], size=num_pts)]
        x_min,y_min,z_min = full_pts.min(axis=0)
        x_max,y_max,z_max = full_pts.max(axis=0)
        a, b = -bound, bound
        xs = scipy.stats.truncnorm.rvs(*get_trunc_ab_range(x_min, x_max, std, a, b), loc=0, scale=std, size=num_pts)
        ys = scipy.stats.truncnorm.rvs(*get_trunc_ab_range(y_min, y_max, std, a, b), loc=0, scale=std, size=num_pts)
        zs = scipy.stats.truncnorm.rvs(*get_trunc_ab_range(z_min, z_max, std, a, b), loc=0, scale=std, size=num_pts)
        pts = np.array([xs,ys,zs]).T + displace
    else:
        x,y,z= center
        a, b = -bound, bound
        xs = scipy.stats.truncnorm.rvs(*get_trunc_ab(x, std, a, b), loc=x, scale=std, size=num_pts)
        ys = scipy.stats.truncnorm.rvs(*get_trunc_ab(y, std, a, b), loc=y, scale=std, size=num_pts)
        zs = scipy.stats.truncnorm.rvs(*get_trunc_ab(z, std, a, b), loc=z, scale=std, size=num_pts)
        pts = np.array([xs,ys,zs]).T
        
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(full_pts)
    dist, ind = x_nn.kneighbors(pts)#[0].squeeze()        
    dist = dist.squeeze()
    ind = ind.squeeze()
    #points_in = points_uniform[np.where(points_distance< 0.1)]
    occ = dist < 0.01
    #pt_class = ind[np.where(dist < 0.01)]
    pt_class = ind[occ != 0]
    return pts, occ, pt_class 

########################################################################
# Visualization
########################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def side_by_side_point_clouds(point_clouds, angle=(90,0)):
    fig = plt.figure()
    W = int(len(point_clouds) ** 0.5)
    H = math.ceil(len(point_clouds) / W)
    for i, pcloud in enumerate(point_clouds):
        action = None
        flow = None
        pts = pcloud['pts']
        title = pcloud['title']
        col = pcloud.get('col', None)
        flow = pcloud.get('flow', None)
        action = pcloud.get('action', None)
        ax = fig.add_subplot(W, H, i+1,projection='3d')
        ax.set_title(title)
        if flow is not None:
            flow_norm = np.linalg.norm(flow, axis=1)
            viz_idx = flow_norm > 0.0
            flow = flow[viz_idx]
            ax.quiver(
                pts[:,0][viz_idx], 
                pts[:,2][viz_idx], 
                pts[:,1][viz_idx], 
                flow[:,0], flow[:,2], flow[:,1],
                color = 'red', linewidth=3, alpha=0.2
            )
        if col is None:
            col = 'blue'
        ax.scatter(pts[:,0], 
                   pts[:,2], 
                   pts[:,1], color=col,s=0.5)
        set_axes_equal(ax)
        ax.view_init(*angle)
        if action is not None:
            ax.scatter(action[0], action[1], 0., 
                       edgecolors='tomato', color='turquoise', marker='*',s=80)
    return fig

def write_pointcoud_as_obj(xyzrgb, path):
    if xyzrgb.shape[1] == 6:
        with open(path, 'w') as fp:
            for x,y,z,r,g,b in xyzrgb:
                fp.write(f"v {x:.3f} {y:.3f} {z:.3f} {r:.3f} {g:.3f} {b:.3f}\n")
    else:
        with open(path, 'w') as fp:
            for x,y,z in xyzrgb:
                fp.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")

#################################
# Distance Metric
#################################
def subsample_points(points, resolution=0.0125, return_index=True):
    idx = np.unique(points// resolution * resolution, axis=0, return_index=True)[1]
    if return_index:
        return idx
    return points[idx]

def miou(x, y, th=0.01):
    x = subsample_points(x, resolution=th, return_index=False) // th 
    y = subsample_points(y, resolution=th, return_index=False) // th 
    xset = set([tuple(i) for i in x])
    yset = set([tuple(i) for i in y])
    return len(xset & yset) / len(xset | yset)

from sklearn.neighbors import NearestNeighbors
def chamfer_distance(x, y, metric='l2', direction='bi'):
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
    min_y_to_x = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    min_x_to_y = y_nn.kneighbors(x)[0]
    return np.mean(min_y_to_x) + np.mean(min_x_to_y)

def f1_score(x, y, metric='l2', th=0.01):
    # x is pred
    # y is gt
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
    d2 = x_nn.kneighbors(y)[0]
    y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
    d1 = y_nn.kneighbors(x)[0]
    recall = float(sum(d < th for d in d2)) / float(len(d2))
    precision = float(sum(d < th for d in d1)) / float(len(d1))

    if recall+precision > 0:
        fscore = 2 * recall * precision / (recall + precision)
    else:
        fscore = 0

    return fscore, precision, recall

from scipy.spatial import cKDTree

def find_nn_cpu(feat0, feat1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds

def find_emd_cpu(feat0, feat1, return_distance=False):
    import time
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    d = cdist(feat0, feat1)
    feat0_inds, feat1_inds = linear_sum_assignment(d)
    return feat0_inds, feat1_inds

def find_nn_cpu_symmetry_consistent(feat0, feat1, pts0, pts1, n_neighbor=10, local_radis=0.05, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=n_neighbor, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds

#################################
# ranking utilities
def overlap(list1, list2, depth):
    """Overlap which accounts for possible ties.
    This isn't mentioned in the paper but should be used in the ``rbo*()``
    functions below, otherwise overlap at a given depth might be > depth which
    inflates the result.
    There are no guidelines in the paper as to what's a good way to calculate
    this, but a good guess is agreement scaled by the minimum between the
    requested depth and the lengths of the considered lists (overlap shouldn't
    be larger than the number of ranks in the shorter list, otherwise results
    are conspicuously wrong when the lists are of unequal lengths -- rbo_ext is
    not between rbo_min and rbo_min + rbo_res.
    >>> overlap("abcd", "abcd", 3)
    3.0
    >>> overlap("abcd", "abcd", 5)
    4.0
    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 2)
    2.0
    >>> overlap(["a", {"b", "c"}, "d"], ["a", {"b", "c"}, "d"], 3)
    3.0
    """
    return agreement(list1, list2, depth) * min(depth, len(list1), len(list2))

def rbo_ext(list1, list2, p=0.9):
    """RBO point estimate based on extrapolating observed overlap.
    See equation (32) in paper.
    NOTE: The doctests weren't verified against manual computations but seem
    plausible.
    >>> _round(rbo_ext("abcdefg", "abcdefg", .9))
    1.0
    >>> _round(rbo_ext("abcdefg", "bacdefg", .9))
    0.9
    """
    S, L = sorted((list1, list2), key=len)
    s, l = len(S), len(L)
    x_l = overlap(list1, list2, l)
    x_s = overlap(list1, list2, s)
    # the paper says overlap(..., d) / d, but it should be replaced by
    # agreement(..., d) defined as per equation (28) so that ties are handled
    # properly (otherwise values > 1 will be returned)
    # sum1 = sum(p**d * overlap(list1, list2, d)[0] / d for d in range(1, l + 1))
    sum1 = sum(p ** d * agreement(list1, list2, d) for d in range(1, l + 1))
    sum2 = sum(p ** d * x_s * (d - s) / s / d for d in range(s + 1, l + 1))
    term1 = (1 - p) / p * (sum1 + sum2)
    term2 = p ** l * ((x_l - x_s) / l + x_s / s)
    return term1 + term2

def set_at_depth(lst, depth):
    ans = set()
    for v in lst[:depth]:
        if isinstance(v, set):
            ans.update(v)
        else:
            ans.add(v)
    return ans

def raw_overlap(list1, list2, depth):
    """Overlap as defined in the article.
    """
    set1, set2 = set_at_depth(list1, depth), set_at_depth(list2, depth)
    return len(set1.intersection(set2)), len(set1), len(set2)

def agreement(list1, list2, depth):
    """Proportion of shared values between two sorted lists at given depth.
    >>> _round(agreement("abcde", "abdcf", 1))
    1.0
    >>> _round(agreement("abcde", "abdcf", 3))
    0.667
    >>> _round(agreement("abcde", "abdcf", 4))
    1.0
    >>> _round(agreement("abcde", "abdcf", 5))
    0.8
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 1))
    0.667
    >>> _round(agreement([{1, 2}, 3], [1, {2, 3}], 2))
    1.0
    """
    len_intersection, len_set1, len_set2 = raw_overlap(list1, list2, depth)
    return 2 * len_intersection / (len_set1 + len_set2)
