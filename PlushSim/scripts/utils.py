import os
import math
import omni
import numpy as np
from PIL import Image
from pxr import UsdGeom, Usd, UsdPhysics, Gf
import matplotlib.pyplot as plt

################################################################
# State Saving Utils
# (Geometry)
################################################################

def transform_points_cam_to_world(cam_pts, camera_pose):
    world_pts = np.transpose(
        np.dot(camera_pose[0:3, 0:3], np.transpose(cam_pts)) + np.tile(camera_pose[0:3, 3:], (1, cam_pts.shape[0])))
    return world_pts

def project_depth_world_space(depth_image, camera_intr, camera_pose, project_factor=1.):
    cam_pts = project_depth_cam_space(depth_image, camera_intr, keep_dim=False, project_factor=project_factor)
    world_pts = transform_points_cam_to_world(cam_pts, camera_pose)
    W, H = depth_image.shape
    pts = world_pts.reshape([W, H, 3])
    return pts

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
    # print("cam_pts: ", cam_pts.max(axis=0), cam_pts.min(axis=0))
    if keep_dim:
        cam_pts = cam_pts.reshape([im_h, im_w, 3])
    return cam_pts

def get_camera_params(viewport):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(viewport.get_active_camera())
    prim_tf = np.array(UsdGeom.Camera(prim).GetLocalTransformation())
    focal_length = prim.GetAttribute("focalLength").Get()
    horiz_aperture = prim.GetAttribute("horizontalAperture").Get()
    fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
    image_w, image_h = viewport.get_texture_resolution()
    camera_focal_length = (float(image_w) / 2) / np.tan(fov/ 2)
    cam_intr = np.array(
            [[camera_focal_length, 0, float(image_h) / 2],
             [0, camera_focal_length, float(image_w) / 2],
             [0, 0, 1]])
    return prim_tf.T, cam_intr

def get_partial_point_cloud(viewport, in_world_space=True, project_factor=1.):
    from omni.syntheticdata import sensors
    data = sensors.get_depth_linear(viewport)
    h, w = data.shape[:2]
    depth_data = -np.frombuffer(data, np.float32).reshape(h, w, -1)
    camera_pose, camera_intr = get_camera_params(viewport)
    if in_world_space:
        return project_depth_world_space(depth_data.squeeze(), camera_intr, camera_pose, project_factor=project_factor)
    else:
        return project_depth_cam_space(depth_data.squeeze(), camera_intr, project_factor=project_factor)

def export_visual_mesh(prim, export_path, loc=None, rot=None, binarize=True):
    assert prim.IsA(UsdGeom.Mesh), "prim needs to be a UsdGeom.Mesh"
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if binarize:
        path = os.path.splitext(export_path)[0]+'.npy'
        np.save(path, np.array(points, np.float16))
    else:
        print(export_path)
        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3) + 1
        uv = mesh.GetPrimvar("st").Get()
        with open(export_path, "w") as fp:
            fp.write("mtllib teddy.mtl\nusemtl Material.004\n")
            for x,y,z in points:
                fp.write(f"v {x:.3f} {y:.3f} {z:.3f}\n")
            for u,v in uv:
                fp.write(f"vt {u:=.4f} {v:.4f}\n")
            for i, (x,y,z) in enumerate(faces):
                fp.write(f"f {x}/{i*3+1} {y}/{i*3+2} {z}/{i*3+3}\n")

def get_sim_points(prim, loc=None, rot=None):
    from pxr import PhysxSchema
    sbAPI = PhysxSchema.PhysxDeformableBodyAPI(prim)
    points = sbAPI.GetSimulationPointsAttr().Get()
    if rot is not None:
        points = np.array(points)
        w,x,y,z = eval(str(rot))
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(np.array([x,y,z,w]))
        points = rot.apply(points)
    if loc is not None:
        loc = np.array(tuple(loc))
        points = points + loc
    return points

def get_sim_faces(prim):
    from pxr import PhysxSchema
    sbAPI = PhysxSchema.PhysxDeformableAPI(prim)
    faces = sbAPI.GetSimulationIndicesAttr().Get()
    return faces 

def export_simulation_voxels(prim, export_path, binarize=True, export_faces=False):
    points = get_sim_points(prim)
    if export_faces:
        faces = get_sim_faces(prim)
    if binarize:
        path = os.path.splitext(export_path)[0]+'.npy'
        if export_faces:
            np.savez(path, points=np.array(points, np.float16), faces=np.array(faces, int))
        else:
            np.save(path, np.array(points, np.float16))
    else:
        with open(export_path, 'w') as fp:
            for p in points:
                fp.write(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
            if export_faces:
                faces = np.array(faces, int).reshape([-1,4]) + 1
                for f in faces:
                    fp.write(f"f {f[0]} {f[1]} {f[2]} {f[3]}\n")

def visualize_sensors(gt, save_path):
    from omni.syntheticdata import visualize
    # GROUNDTRUTH VISUALIZATION
    # Setup a figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes = axes.flat
    for ax in axes:
        ax.axis("off")
    # RGB
    axes[0].set_title("RGB")
    for ax in axes[:-1]:
        ax.imshow(gt["rgb"])
    # DEPTH
    axes[1].set_title("Depth")
    depth_data = np.clip(gt["depth"], 0, 255)
    axes[1].imshow(visualize.colorize_depth(depth_data.squeeze()))
    # SEMSEG
    axes[2].set_title("Semantic Segmentation")
    semantic_seg = gt["semanticSegmentation"]
    semantic_rgb = visualize.colorize_segmentation(semantic_seg)
    axes[2].imshow(semantic_rgb, alpha=0.7)
    # Save figure
    fig.savefig(save_path)
    plt.close(fig)


def save_frame(frame_name, frame_data, save_dir, 
               save_rgb=True, save_seg=True, save_depth=True, save_partial_pointcloud=False):
    if save_rgb:
        rgb = frame_data['rgb_img']
        Image.fromarray(rgb).save(f"{save_dir}/rgb_{frame_name}.jpg")
    if save_seg:
        seg= frame_data['seg_img']
        sem = np.tile(seg[:,:,np.newaxis], (1,1,3)).astype(np.uint8) * 255
        Image.fromarray(sem).save(f"{save_dir}/seg_{frame_name}.jpg")
    if save_depth:
        depth_img = Image.fromarray((frame_data['dep_img'].squeeze() * 1000).astype(np.uint16), mode='I;16').convert(mode='I')
        depth_img.save(f"{save_dir}/depth_{frame_name}.png")
def save_state(state_name, state_data, save_dir):
    loc, rot, sim, vis = state_data
    state_dict = {}
    state_dict['loc'] = np.array(tuple(loc))
    state_dict['rot'] = np.array(eval(str(rot)))
    state_dict['sim'] = np.array(sim)
    state_dict['vis'] = np.array(vis)
    np.savez(f"{save_dir}/state_{state_name}.npz", **state_dict)


################################################################
# Interaction Utils
################################################################

def sample_pick_point(partial_point_cloud, segmentation):
    im_h = segmentation.shape[0]
    im_w = segmentation.shape[1]
    # point cloud "image" height and width
    pc_h = partial_point_cloud.shape[0]
    pc_w = partial_point_cloud.shape[1]
    assert im_h == pc_h and im_w == pc_w, "partial_point_cloud dimension should match with that of segmentation mask"

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def sample_direction(npoints):
    phi = np.random.randn(npoints) * 2 * np.pi
    theta = np.clip(np.random.normal(loc=np.pi / 4.,scale=np.pi / 12., size=npoints), np.pi / 6., np.pi / 2.)
    x = np.cos(phi) * np.sin(theta)
    z = np.sin(phi) * np.sin(theta)
    y = np.cos(theta) 
    vec = np.vstack([x,y,z])
    return vec

def sample_direction_zup(npoints):
    phi = np.random.randn(npoints) * 2 * np.pi
    theta = np.clip(np.random.normal(loc=np.pi / 4.,scale=np.pi / 12., size=npoints), np.pi / 6., np.pi / 2.)
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(theta) 
    vec = np.vstack([x,y,z])
    return vec

def interpolate(start_loc, end_loc, speed):
    start_loc = np.array(start_loc)
    end_loc = np.array(end_loc)
    dist = np.linalg.norm(end_loc - start_loc)
    chunks = dist // speed
    return start_loc + np.outer(np.arange(chunks+1,dtype=float), (end_loc - start_loc) / chunks)

class magic_eef(object):
    def __init__(self, end_effector, stage, eef_default_loc=None, default_speed=1,
        fingerL=None, fingerR=None):
        self.end_effector = end_effector
        self.eef_default_loc = eef_default_loc
        self.default_speed = default_speed
        self.stage = stage
        xform = UsdGeom.Xformable(end_effector)
        self.ops = xform.GetOrderedXformOps()
        assert self.ops[0].GetOpType() == UsdGeom.XformOp.TypeTranslate,\
             "Code is based on UsdGeom.Xformable with first op as translation"
        assert self.ops[1].GetOpType() == UsdGeom.XformOp.TypeOrient,\
             "Code is based on UsdGeom.Xformable with second op as orientation"
        self.attachmentPath = None
        self.set_translation(eef_default_loc)
        self.fingerL=fingerL
        if fingerL is not None: 
            xform = UsdGeom.Xformable(fingerL)
            self.fingerL_ops = xform.GetOrderedXformOps()[0]
            self.fingerL_ops.Set((-5,0,20))
        self.fingerR=fingerR
        if fingerR is not None: 
            xform = UsdGeom.Xformable(fingerR)
            self.fingerR_ops = xform.GetOrderedXformOps()[0]
            self.fingerL_ops.Set((5,0,20))
        
    
    def get_translation(self):
        return self.ops[0].Get()

    def set_translation(self, loc):
        self.ops[0].Set(loc)

    def reset_translation(self):
        self.set_translation(self.eef_default_loc)

    def get_orientation(self):
        return self.ops[1].Get()

    def set_orientation(self, rot):
        self.ops[1].Set(rot)

    def grasp(self, target_object):
        # enable collision
        self.end_effector.GetAttribute("physics:collisionEnabled").Set(True)
        # create magic grasp
        self.attachmentPath = target_object.GetPath().AppendChild("rigidAttachment_0")
        omni.kit.commands.execute(
            "AddSoftBodyRigidAttachmentCommand",
            target_attachment_path=self.attachmentPath,
            softbody_path=target_object.GetPath(),
            rigidbody_path=self.end_effector.GetPath(),
        )
        attachmentPrim = self.stage.GetPrimAtPath(self.attachmentPath)
        assert attachmentPrim
        assert attachmentPrim.GetAttribute("physxEnableHaloParticleFiltering").Set(True)
        assert attachmentPrim.GetAttribute("physxEnableVolumeParticleAttachments").Set(True)
        assert attachmentPrim.GetAttribute("physxEnableSurfaceTetraAttachments").Set(True)
        omni.physx.get_physx_interface().release_physics_objects()
        self.fingerL_ops.Set((-5,0,20))
        self.fingerR_ops.Set((5,0,20))

    def ungrasp(self):
        assert self.attachmentPath is not None, "nothing is grasped! (there is no attachment registered)"
        # release magic grasp
        omni.kit.commands.execute(
            "DeletePrimsCommand",
            paths=[self.attachmentPath]
        )
        self.end_effector.GetAttribute("physics:collisionEnabled").Set(False)
        omni.physx.get_physx_interface().release_physics_objects()
        self.attachmentPath = None
        self.fingerL_ops.Set((-80,0,20))
        self.fingerR_ops.Set((80,0,20))
        #self.reset_translation()

    def plan_trajectory(self, start_loc, end_loc, speed=None):
        return interpolate(start_loc, end_loc, self.default_speed if speed is None else speed)

################################
# Random utils
################################

def get_camera_name(viewport):
    stage = omni.usd.get_context().get_stage()
    return stage.GetPrimAtPath(viewport.get_active_camera()).GetName()

def rpy2quat(roll,pitch,yaw):        
    roll*=0.5
    pitch*=0.5
    yaw*=0.5
    cr = math.cos(roll)
    cp = math.cos(pitch)
    cy = math.cos(yaw)

    sr = math.sin(roll)
    sp = math.sin(pitch)
    sy = math.sin(yaw)

    cpcy = cp * cy
    spsy = sp * sy
    spcy = sp * cy
    cpsy = cp * sy

    qx = (sr * cpcy - cr * spsy)
    qy = (cr * spcy + sr * cpsy)
    qz = (cr * cpsy - sr * spcy)
    qw = cr * cpcy + sr * spsy        
    return Gf.Quatf(qw,qx,qy,qz)

################################
# Scene randomization utils
################################
def is_collider(prim):
    try:
        return prim.GetAttribute("physics:collisionEnabled").Get()
    except:
        return False

def find_collider(prim):
    #from pxr import UsdPhysics
    primRange = iter(Usd.PrimRange(prim))
    extent, transform = None, None
    for p in primRange:
        #if p.HasAPI(UsdPhysics.CollisionAPI):
        if is_collider(p):
            extent = p.GetAttribute("extent").Get()
            if extent is None:
                # this means that the object is a cube
                extent = np.array([[-50,-50,-50],[50,50,50]])
            transform = omni.usd.get_world_transform_matrix(p, Usd.TimeCode.Default())
            primRange.PruneChildren()
            break
    return np.array(extent), np.array(transform)

def find_immediate_children(prim):
    primRange = Usd.PrimRange(prim)
    primPath = prim.GetPath()
    immediate_children = []
    for p in primRange:
        if p.GetPath().GetParentPath() == primPath:
            immediate_children.append(p)
    return immediate_children

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

def transform_verts(verts, transform):
    verts_app = np.concatenate([verts,np.ones((verts.shape[0], 1))], axis=-1)
    return (verts_app @ transform)[:,:-1]

def export_quad_obj(verts, faces, export_path):
    with open(export_path, 'w') as fp:
        for p in verts:
            fp.write(f"v {p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
        for f in faces:
            fp.write(f"f {f[0]} {f[1]} {f[2]} {f[3]}\n")

def standardize_bbox(bbox):
    return np.array([bbox.min(axis=0),bbox.max(axis=0)])

def get_bbox_translation_range(bbox, scene_range):
    # bbox size
    size_x,size_y = bbox[1] - bbox[0]
    center_range = scene_range + np.array([[size_x, size_y],[-size_x,-size_y]]) / 2
    center = np.mean(bbox, axis=0)
    return center_range - center

def sample_bbox_translation(bbox, scene_range):
    translation_range = get_bbox_translation_range(bbox, scene_range)
    sample = np.random.rand(2)
    return translation_range[0] + sample * (translation_range[1] - translation_range[0])

def get_canvas(scene_range):
    scene_size = scene_range[1] - scene_range[0]
    scene_size = ( scene_size * 1.1 ).astype(int)
    return np.zeros(scene_size)

def fill_canvas(canvas, scene_range, bbox,val=1):
    canvas_center = np.array(canvas.shape) / 2        
    cb = (bbox - np.mean(scene_range, axis=0) + canvas_center).astype(int)
    if cb[0,0] < 0 or cb[0,1] < 0:
        return
    h,w = canvas.shape
    if cb[1,0] >= h or cb[1,1] >= w:
        return
    canvas[cb[0,0]:cb[1,0], cb[0,1]:cb[1,1]] = val

def get_occupancy_value(canvas, scene_range, pts):
    canvas_center = np.array(canvas.shape) / 2        
    pts = (pts - np.mean(scene_range, axis=0) + canvas_center).astype(int)
    return canvas[pts[:,0], pts[:,1]]

def overlaps_with_current(canvas, scene_range, bbox,val=0):
    canvas_center = np.array(canvas.shape) / 2        
    cb = (bbox - np.mean(scene_range, axis=0) + canvas_center).astype(int)
    return (canvas[cb[0,0]:cb[1,0], cb[0,1]:cb[1,1]] != val).any()

def pad_to_square(bbox):
    size_x,size_y = (bbox[1] - bbox[0]) / 2.
    center = np.mean(bbox, axis=0)
    length = max(size_x,size_y)
    return np.stack([center-length,center+length])

def scale(bbox,factor=1.1):
    size_x,size_y = (bbox[1] - bbox[0]) / 2. *factor
    center = np.mean(bbox, axis=0)
    return np.stack([center-[size_x,size_y],center+[size_x,size_y]])
