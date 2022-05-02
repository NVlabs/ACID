import os
import glob
import json
import scipy
import itertools
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def get_color_map(x):
  colours = plt.cm.Spectral(x)
  return colours[:, :3]

def embed_tsne(data):
  """
  N x D np.array data
  """
  tsne = TSNE(n_components=1, verbose=0, perplexity=40, n_iter=300, random_state=0)
  tsne_results = tsne.fit_transform(data)
  tsne_results = np.squeeze(tsne_results)
  tsne_min = np.min(tsne_results)
  tsne_max = np.max(tsne_results)
  return (tsne_results - tsne_min) / (tsne_max - tsne_min)


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
    
def render_points(world_points, cam_extr, cam_intr, return_index=False):
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
    if return_index:
        return min_idx
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

def get_trunc_ab_range(mean_min, mean_max, std, a, b):
    return (a - mean_min) / std, (b - mean_max) /std 

def transform_points(pointcloud, from_range, to_range):
    if len(pointcloud.shape) == 1:
        pointcloud = pointcloud.reshape([1,-1])
    if pointcloud.shape[1] == 6:
        xyz = pointcloud[:,:3]
        rgb = pointcloud[:,3:]
    else:
        xyz = pointcloud
        rgb = None
    from_center = np.mean(from_range, axis=0)
    from_size = np.ptp(from_range, axis=0)
    to_center = np.mean(to_range, axis=0)
    to_size = np.ptp(to_range, axis=0)
    xyz = (xyz - from_center) / from_size * to_size + to_center 
    if rgb is None:
        return xyz
    else:
        return np.concatenate([xyz, rgb], axis=-1)

def extent_to_cube(extent):
    min_x,min_y,min_z = extent[0]
    max_x,max_y,max_z = extent[1]
    verts = np.array([
        (max_x,max_y,max_z),
        (max_x,max_y,min_z),
        (max_x,min_y,max_z),
        (max_x,min_y,min_z),
        (min_x,max_y,max_z),
        (min_x,max_y,min_z),
        (min_x,min_y,max_z),
        (min_x,min_y,min_z),])
    faces = np.array([
        (1,5,7,3),
        (4,3,7,8),
        (8,7,5,6),
        (6,2,4,8),
        (2,1,3,4),
        (6,5,1,2),])
    return verts, faces

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

def set_background_blank(ax):
    # Hide grid lines
    ax.grid(False)
    ax.set_axis_off()
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))

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
                pts[:,1][viz_idx], 
                pts[:,2][viz_idx], 
                flow[:,0], flow[:,1], flow[:,2],
                color = 'red', linewidth=3, alpha=0.2
            )
        if col is None:
            col = 'blue'
        ax.scatter(pts[:,0], 
                   pts[:,1], 
                   pts[:,2], color=col,s=0.5)
        ax.view_init(*angle)
        if action is not None:
            ax.scatter(action[0], action[1], 0., 
                       edgecolors='tomato', color='turquoise', marker='*',s=80)
        set_axes_equal(ax)
        set_background_blank(ax)
    fig.tight_layout()
    return fig

def write_pointcoud_as_obj(path, xyzrgb, faces=None):
    with open(path, 'w') as fp:
        if xyzrgb.shape[1] == 6:
            for x,y,z,r,g,b in xyzrgb:
                fp.write(f"v {x:.3f} {y:.3f} {z:.3f} {r:.3f} {g:.3f} {b:.3f}\n")
        else:
            for x,y,z in xyzrgb:
                fp.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
        if faces is not None:
            for f in faces:
                f_str = " ".join([str(i) for i in f])
                fp.write(f"f {f_str}\n")

#################################
# Distance Metric
#################################
def subsample_points(points, resolution=0.0125, return_index=True):
    if points.shape[1] == 6:
        xyz = points[:,:3]
    else:
        xyz = points
    if points.shape[0] == 0:
        if return_index:
            return np.arange(0)
        return points
    idx = np.unique(xyz// resolution * resolution, axis=0, return_index=True)[1]
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
    if x.shape[0] == 0:
        return 0,0,0
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