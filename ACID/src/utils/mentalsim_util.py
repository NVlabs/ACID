import os
import glob
import json
import scipy
import itertools
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

########################################################################
# Viewpoint transform
########################################################################
view_to_order = {
    'cam0': ('X', 'Y', 'Z'),
    'cam1': ('-Z', 'Y', 'X'),
    'cam2': ('Z', 'Y', '-X'),
    'cam3': ('-X', 'Y', '-Z'),
}
def get_axis_pt(val, x, y, z):
    multiplier = -1 if '-' in val else 1
    if "X" in val:
        return x * multiplier
    elif "Y" in val:
        return y * multiplier
    elif "Z" in val:
        return z * multiplier

def world_coord_view_augmentation(view, pts):
    order = view_to_order[view]
    pts = pts.reshape([-1,3])
    x,y,z = np.moveaxis(pts, 1, 0)
    return np.array([get_axis_pt(o,x,y,z) for o in order]).T

########################################################################
# partial observation projection / transform / rendering utilities 
########################################################################
def transform_points_cam_to_world(cam_pts, camera_pose):
    world_pts = np.transpose(
        np.dot(camera_pose[0:3, 0:3], np.transpose(cam_pts)) + np.tile(camera_pose[0:3, 3:], (1, cam_pts.shape[0])))
    return world_pts

def transform_points_world_to_cam(world_points, cam_extr):
    return np.transpose(
        np.dot(
            np.linalg.inv(
                cam_extr[0:3, 0:3]), 
                np.transpose(world_points) 
                - np.tile(cam_extr[0:3, 3:], (1, world_points.shape[0]))))

def render_points_slowest(world_points, cam_extr, cam_intr):
    cam_points = transform_points_world_to_cam(world_points, cam_extr)
    cam_pts_x = cam_points[:,0]
    cam_pts_y = cam_points[:,1]
    cam_pts_z = cam_points[:,2]
    cam_pts_x = -cam_pts_x / cam_pts_z * cam_intr[0,0] + cam_intr[1,2]
    cam_pts_y = cam_pts_y / cam_pts_z * cam_intr[1,1] + cam_intr[0,2]
    cam_pts_x = np.rint(cam_pts_x).astype(int)
    cam_pts_y = np.rint(cam_pts_y).astype(int)
    points = np.stack([cam_pts_y, cam_pts_x, cam_pts_z, np.arange(len(cam_pts_x))]).T
    sorted_pts = sorted(points, key=lambda x: (x[0], x[1]))
    grouped_pts = [[*j] for i, j in itertools.groupby(
        sorted_pts, 
                    key=lambda x: (x[0] // 3, x[1] // 3))]
    min_depth = np.array([sorted(p, key=lambda x: -x[2])[0] for p in grouped_pts])
    min_idx = min_depth[:,-1]
    min_depth = min_depth[:,:-1]
    return world_points[min_idx.astype(int)]

def render_points_slow(world_points, cam_extr, cam_intr):
    cam_points = transform_points_world_to_cam(world_points, cam_extr)
    cam_pts_x = cam_points[:,0]
    cam_pts_y = cam_points[:,1]
    cam_pts_z = cam_points[:,2]
    cam_pts_x = -cam_pts_x / cam_pts_z * cam_intr[0,0] + cam_intr[1,2]
    cam_pts_y = cam_pts_y / cam_pts_z * cam_intr[1,1] + cam_intr[0,2]
    points = np.stack([cam_pts_y, cam_pts_x, cam_pts_z, np.arange(len(cam_pts_x))]).T
    points[:,:2] = np.rint(points[:,:2] / 2)
    points = points[points[:,1].argsort()]
    points = points[points[:,0].argsort(kind='mergesort')]
    grouped_pts = np.split(points[:,2:], np.unique(points[:, :2], axis=0, return_index=True)[1][1:])
    min_depth = np.array([p[p[:,0].argsort()][-1] for p in grouped_pts])
    min_idx = min_depth[:,-1].astype(int)
    return world_points[min_idx]
    
def render_points(world_points, cam_extr, cam_intr):
    cam_points = transform_points_world_to_cam(world_points, cam_extr)
    cam_pts_x = cam_points[:,0]
    cam_pts_y = cam_points[:,1]
    cam_pts_z = cam_points[:,2]
    cam_pts_x = -cam_pts_x / cam_pts_z * cam_intr[0,0] + cam_intr[1,2]
    cam_pts_y = cam_pts_y / cam_pts_z * cam_intr[1,1] + cam_intr[0,2]
    idx = np.rint(cam_pts_y / 2) * 1000 + np.rint(cam_pts_x / 2)
    val = np.stack([cam_pts_z, np.arange(len(cam_pts_x))]).T
    order = idx.argsort()
    idx = idx[order]
    val = val[order]
    grouped_pts = np.split(val, np.unique(idx, return_index=True)[1][1:])
    min_depth = np.array([p[p[:,0].argsort()][-1] for p in grouped_pts])
    min_idx = min_depth[:,-1].astype(int)
    return world_points[min_idx]

def project_depth_world_space(depth_image, camera_intr, camera_pose, keep_dim=False, project_factor=1.):
    cam_pts = project_depth_cam_space(depth_image, camera_intr, keep_dim=False,project_factor=project_factor)
    world_pts = transform_points_cam_to_world(cam_pts, camera_pose)
    W, H = depth_image.shape
    if keep_dim:
        world_pts = world_pts.reshape([W, H, 3])
    return world_pts

def project_depth_cam_space(depth_img, camera_intrinsics, keep_dim=True, project_factor=1.):
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]
    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
    cam_pts_x = np.multiply(pix_x - im_w / 2., -depth_img / camera_intrinsics[0, 0])
    cam_pts_y = np.multiply(pix_y - im_h / 2., depth_img / camera_intrinsics[1, 1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h * im_w, 1)
    cam_pts_y.shape = (im_h * im_w, 1)
    cam_pts_z.shape = (im_h * im_w, 1)
    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1) * project_factor
    if keep_dim:
        cam_pts = cam_pts.reshape([im_h, im_w, 3])
    return cam_pts

def get_trunc_ab(mean, std, a, b):
    return (a - mean) / std, (b - mean) /std 

########################################################################
# partial observation getter for full experiment
########################################################################
CAM_EXTR = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.6427898318479135, -0.766043895201295, -565.0], 
                     [0.0, 0.766047091387779, 0.6427871499290135, 550.0], [0.0, 0.0, 0.0, 1.0]])
CAM_INTR = np.array([[687.1868314210544, 0.0, 360.0], [0.0, 687.1868314210544, 360.0], [0.0, 0.0, 1.0]])
SCENE_RANGE = np.array([[-600, -400, 0], [600, 400, 400]])

def get_scene_partial_pointcloud(model_category, model_name, split_id, int_id, frame_id, data_root):
    path = f"{data_root}/{split_id}/{model_category}/{model_name}/img/{{}}_{int_id:04d}_{frame_id:06d}.{{}}"
    depth_img = path.format('depth', 'png') 
    depth_img = np.array(Image.open(depth_img).convert(mode='I'))
    depth_vals = -np.array(depth_img).astype(float) / 1000.

    rgb_img = path.format('rgb', 'jpg') 
    rgb_img = np.array(Image.open(rgb_img).convert(mode="RGB")).astype(float) / 255

    seg_img = path.format('seg', 'jpg')
    seg_img = np.array(Image.open(seg_img).convert('L')).squeeze()
    non_env = np.where(seg_img != 0)
    env = np.where(seg_img == 0)

    partial_points = project_depth_world_space(depth_vals, CAM_INTR, CAM_EXTR, keep_dim=True, project_factor=100.)
    partial_points_rgb = np.concatenate([partial_points, rgb_img], axis=-1)
    obj_pts = partial_points_rgb[non_env]
    env_pts = partial_points_rgb[env]
    return obj_pts, env_pts
    
########################################################################
# Get geometric state (full experiment)
########################################################################
def get_object_full_points(model_category, model_name, split_id, int_id, frame_id, data_root):
    path = f"{data_root}/{split_id}/{model_category}/{model_name}/geom/{int_id:04d}_{frame_id:06d}.npz"
    geom_data = np.load(path)
    loc = geom_data['loc']
    print(geom_data['rot'])
    w,x,y,z= geom_data['rot']
    rot = Rotation.from_quat(np.array([x,y,z,w]))
    scale = geom_data['scale']
    sim_pts = (rot.apply(geom_data['sim'] * scale)) + loc
    vis_pts = (rot.apply(geom_data['vis'] * scale)) + loc
    return sim_pts, vis_pts


########################################################################
# partial observation getter for teddy toy example
########################################################################

def get_teddy_partial_pointcloud(int_group, int_id, frame_id, data_root, cam_id='cam0'):
    #depth_img = glob.glob(f"{data_root}/{int_group}/img/{cam_id}/{int_id:06d}_*{frame_id:03d}_depth.png")[0]
    depth_img = f"{data_root}/{int_group}/img/{cam_id}/{int_id:06d}_{frame_id:03d}_depth.png"
    depth_img = np.array(Image.open(depth_img).convert(mode='I'))
    depth_vals = -np.array(depth_img).astype(float) / 1000.

    #rgb_img = glob.glob(f"{data_root}/{int_group}/img/{cam_id}/{int_id:06d}_*{frame_id:03d}_rgb.png")[0]
    rgb_img = f"{data_root}/{int_group}/img/{cam_id}/{int_id:06d}_{frame_id:03d}_rgb.png"
    rgb_img = np.array(Image.open(rgb_img).convert(mode="RGB")).astype(float) / 255

    #seg_img = glob.glob(f"{data_root}/{int_group}/img/{cam_id}/{int_id:06d}_*{frame_id:03d}_seg.png")[0]
    seg_img = f"{data_root}/{int_group}/img/{cam_id}/{int_id:06d}_{frame_id:03d}_seg.png"
    seg_img = np.array(Image.open(seg_img))
    non_env = np.where(seg_img != 0)

    ospdir= os.path.dirname
    root_dir = ospdir(ospdir(ospdir(os.path.realpath(__file__))))
    camera_json = os.path.join(root_dir, "metadata", "camera.json")
    with open(camera_json, 'r') as fp:
        cam_info = json.load(fp)
    for k in cam_info.keys():
        cam_extr, cam_intr = cam_info[k]
        cam_info[k] = np.array(cam_extr), np.array(cam_intr)

    cam_extr, cam_intr = cam_info[cam_id]
    partial_points = project_depth_world_space(depth_vals, cam_intr, cam_extr, keep_dim=True)
    partial_points_rgb = np.concatenate([partial_points, rgb_img], axis=-1)
    xyzrgb = partial_points_rgb[non_env]
    xyz = xyzrgb[:,:3]
    xyz = world_coord_view_augmentation(cam_id, xyz)
    rgb = xyzrgb[:,3:]
    return xyz/ 10. * 1.1, rgb


########################################################################
# Get meta info (teddy toy example)
########################################################################
def get_teddy_loc(int_group, int_id, frame_id, data_root):
    obj_info = f"{data_root}/{int_group}/info/{int_id:06d}.json"
    with open(obj_info, 'r') as fp:
        int_info = json.load(fp)
    return np.array(dict(zip(int_info['frames'], int_info['teddy_loc']))[frame_id])

def get_teddy_rot(int_group, int_id, frame_id, data_root):
    obj_info = f"{data_root}/{int_group}/info/{int_id:06d}.json"
    with open(obj_info, 'r') as fp:
        int_info = json.load(fp)
    w,x,y,z =  np.array(dict(zip(int_info['frames'], int_info['teddy_rot']))[frame_id])
    return np.array([x,y,z,w])

def get_action_info(int_group, int_id, data_root):
    obj_info = f"{data_root}/{int_group}/info/{int_id:06d}.json"
    with open(obj_info, 'r') as fp:
        int_info = json.load(fp)
    grasp_loc = np.array(int_info['grasp'])
    target_loc = np.array(int_info['target'])
    return grasp_loc, target_loc

def get_release_frame(int_group, int_id, data_root):
    obj_info = f"{data_root}/{int_group}/info/{int_id:06d}.json"
    with open(obj_info, 'r') as fp:
        return json.load(fp)['release_frame']
    # name = glob.glob(
    #     f"{data_root}/{int_group}/geom/{int_id:06d}_release_*_sim.npy")[0].split("/")[-1]
    # return int(name.split("_")[-2])

def get_end_frame(int_group, int_id, data_root):
    obj_info = f"{data_root}/{int_group}/info/{int_id:06d}.json"
    with open(obj_info, 'r') as fp:
        return json.load(fp)['end_frame']
    # name = glob.glob(
    #     f"{data_root}/{int_group}/geom/{int_id:06d}_static_*_sim.npy")[0].split("/")[-1]
    # return int(name.split("_")[-2])

########################################################################
# Get geometric state (teddy toy example)
########################################################################
def get_teddy_full_points(int_group, int_id, frame_id, data_root):
    #sim_data = glob.glob(f"{data_root}/{int_group}/geom/{int_id:06d}_*{frame_id:03d}_sim.npy")[0]
    sim_data = f"{data_root}/{int_group}/geom/{int_id:06d}_{frame_id:03d}_sim.npy"
    points = np.load(sim_data)
    teddy_loc = get_teddy_loc(int_group, int_id, frame_id, data_root)
    teddy_rot = Rotation.from_quat(get_teddy_rot(int_group, int_id, frame_id, data_root))
    return ( teddy_rot.apply(points) + teddy_loc ) / 10. * 1.1
    #return ( points + teddy_loc ) / 10. * 1.1

def get_teddy_vis_points(int_group, int_id, frame_id, data_root):
    #sim_data = glob.glob(f"{data_root}/{int_group}/geom/{int_id:06d}_*{frame_id:03d}_vis.npy")[0]
    sim_data = f"{data_root}/{int_group}/geom/{int_id:06d}_{frame_id:03d}_vis.npy"
    points = np.load(sim_data)
    teddy_loc = get_teddy_loc(int_group, int_id, frame_id, data_root)
    teddy_rot = Rotation.from_quat(get_teddy_rot(int_group, int_id, frame_id, data_root))
    return ( teddy_rot.apply(points) + teddy_loc ) / 10. * 1.1
    #return ( points + teddy_loc ) / 10. * 1.1

########################################################################
# Get point-based supervision data for implicit functions (teddy toy example)
########################################################################
def sample_occupancies(int_group, int_id, frame_id, data_root, sample_scheme='uniform'):
    if sample_scheme not in ['uniform', 'gaussian']:
        raise ValueError('Unsupported sampling scheme for occupancy')
    num_pts = 100000
    if sample_scheme == 'uniform':
        pts = np.random.rand(num_pts, 3)
        pts = 1.1 * (pts - 0.5)
    else:
        x,y,z= get_teddy_loc(int_group, int_id, frame_id, data_root) / 10. * 1.1
        std = 0.18
        a, b = -0.55, 0.55
        xs = scipy.stats.truncnorm.rvs(*get_trunc_ab(x, std, a, b), loc=x, scale=std, size=num_pts)
        ys = scipy.stats.truncnorm.rvs(*get_trunc_ab(y, std, a, b), loc=y, scale=std, size=num_pts)
        zs = scipy.stats.truncnorm.rvs(*get_trunc_ab(z, std, a, b), loc=z, scale=std, size=num_pts)
        pts = np.array([xs,ys,zs]).T
        
    teddy_sim_points = get_teddy_full_points(int_group, int_id, frame_id, data_root)
    x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric='l2').fit(teddy_sim_points)
    dist, ind = x_nn.kneighbors(pts)#[0].squeeze()        
    dist = dist.squeeze()
    ind = ind.squeeze()
    occ = dist < 0.01
    pt_class = ind[occ != 0]
    return pts, occ, pt_class 

def sample_occupancies_with_flow(int_group, int_id, release_frame, end_frame, data_root, sample_scheme='uniform'):
    pts, occ, ind = sample_occupancies(int_group, int_id, 0, data_root, sample_scheme)
    xyz0 = get_teddy_full_points(int_group, int_id, 0, data_root)
    f1 = get_teddy_full_points(int_group, int_id, release_frame, data_root) - xyz0
    f2 = get_teddy_full_points(int_group, int_id, end_frame, data_root) - xyz0
    return pts, occ, ind, f1[ind],f2[ind] 

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