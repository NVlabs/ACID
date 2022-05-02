import os
import cv2
import time
import random
import asyncio
import numpy as np
from python_app import OmniKitHelper
import omni
import carb

from utils import *

RESOLUTION=720
# specify a custom config
CUSTOM_CONFIG = {
    "width": RESOLUTION,
    "height": RESOLUTION,
    "anti_aliasing": 3,  # 3 for dlss, 2 for fxaa, 1 for taa, 0 to disable aa
    "renderer": "RayTracedLighting",
    "samples_per_pixel_per_frame": 128,
    "max_bounces": 10,
    "max_specular_transmission_bounces": 6,
    "max_volume_bounces": 4,
    "subdiv_refinement_level": 2,
    "headless": True,
    "sync_loads": True,
    "experience": f'{os.environ["EXP_PATH"]}/omni.bloky.kit',
}
"""
plush animal material: /Root/physics/stuff_animal
magic gripper: /Root/physics/magic_gripper
real object group:  /Root/physics/real_objects
magic object group:  /Root/physics/magic_objects
"""
class attic_scene(object):

    def __init__(self, 
                SCENE_PATH,
                PLUSH_ANIMAL_PATH,
                PLUSH_SCALE=4,
                FALL_MAX=300,
                REST_THRESHOLD=8,
                PHYSX_DT=1/150.,
                SAVE_EVERY=25,
                DROP_MIN=20,
                RESET_STATIC=True,
                RAND_LAYOUT=True,
                RAND_LIGHTS=True,
                ROBOT_SPEED=1.):
        for k,v in locals().items():
            if k != 'self':
                self.__dict__[k] = v
        self.plush_animal_mat = "/Root/physics/stuff_animal"
        self.magic_gripper = "/Root/physics/magic_gripper"
        self.fingerL = "/Root/physics/magic_gripper/fingerL"
        self.fingerR = "/Root/physics/magic_gripper/fingerR"
        self.real_object_group = "/Root/physics/real_objects"
        self.magic_object_group = "/Root/physics/magic_objects"
        self.front_path = "/Root/scene_front"
        self.back_path = "/Root/scene_back"
        self.scene_range = np.array([[-50*12,-50*8,0],[50*12,50*8,50*8]])
        self.drop_range = np.array([[-50*self.PLUSH_SCALE,-50*self.PLUSH_SCALE,],
                                    [50*self.PLUSH_SCALE,50*self.PLUSH_SCALE,]]) #/ 2.
        self.back_clutter_range = np.array([[-50*12,50*8,],[50*12,50*12,]])
        self.total_range = np.array([[-50*12,-50*12,0],[50*12,50*12,50*8]])
        self.kit = OmniKitHelper(CUSTOM_CONFIG)
        self.kit.set_physics_dt(physics_dt=self.PHYSX_DT)

        physx_interface = omni.physx.get_physx_interface()
        physx_interface.force_load_physics_from_usd()
        physx_interface.reset_simulation()

        async def load_stage(path):
            await omni.usd.get_context().open_stage_async(path)
        setup_task = asyncio.ensure_future(load_stage(SCENE_PATH))
        while not setup_task.done():
            self.kit.update()

        self.kit.setup_renderer()
        self.kit.update()
        self.stage = omni.usd.get_context().get_stage()
        self.front_group = self.stage.GetPrimAtPath(self.front_path)
        self.back_group = self.stage.GetPrimAtPath(self.back_path)

        from syntheticdata import SyntheticDataHelper
        self.sd_helper = SyntheticDataHelper()
        # force RayTracedLighting mode for better performance while simulating physics
        self.kit.set_setting("/rtx/rendermode", "RayTracedLighting")
        # wait until all materials are loaded
        print("waiting for things to load...")
        # if self.kit.is_loading():
        #     time.sleep(10)
        while self.kit.is_loading():
            time.sleep(0.1)

        # set up cameras
        self._setup_cameras()
        _viewport_api = omni.kit.viewport.get_viewport_interface()
        viewport = _viewport_api.get_instance_list()[0]
        self._viewport = _viewport_api.get_viewport_window(viewport)

        # touch the sensors to kick in anti-aliasing
        for _ in range(20):
            _ = self.sd_helper.get_groundtruth(
                [ "rgb","depth","instanceSegmentation","semanticSegmentation",], self._viewport)

        # set up objects
        self._import_plush_animal(PLUSH_ANIMAL_PATH)
        self._setup_robots()

        # # start off Omniverse
        self.kit.play()

        # store original sim and vis points for reset
        self.sim_og_pts, self.vis_og_pts = self._get_plush_points()

        # # stop Omniverse
        # self.kit.pause()

        # reset the scene
        self.frame = 0
        self.reset()

    def step(self):
        self.kit.update(self.PHYSX_DT)
        self.frame += 1
        return self.frame

    def sample_action(self, grasp_point=None):
        if grasp_point is None:
            gt = self.sd_helper.get_groundtruth(
                [ "rgb","depth","instanceSegmentation","semanticSegmentation",], self._viewport)
            pts = get_partial_point_cloud(self._viewport, project_factor=100.)
            semseg = gt['semanticSegmentation']
            kernel = np.ones((2,2), np.uint8)
            semseg = cv2.erode(semseg, kernel, iterations=1)
            plush_pts = np.where(semseg == 1)
            if len(plush_pts[0]) == 0:
                return None
            idx = random.randint(0,len(plush_pts[0])-1)
            grasp_pixel = (plush_pts[0][idx], plush_pts[1][idx])
            grasp_point = tuple(pts[grasp_pixel[0], grasp_pixel[1],:])
        else:
            grasp_pixel = None
        target_point = self._sample_displacement_vector(grasp_point)
        if target_point is None:
            return None
        return grasp_point, target_point, grasp_pixel
    
    def reset(self):
        self.kit.stop()
        from pxr import Gf
        self.frame = 0
        print("Reseting plush geometry...")
        self._reset_plush_geometry(self.sim_og_pts, self.vis_og_pts)
        print("Finished reseting plush geometry...")
        # randonly drop the plush into the scene
        print("Reseting plush translation...")
        self.plush_translateOp.Set(Gf.Vec3f((0.,0.,250.)))
        print("Reseting plush rotation...")
        def randrot():
            return random.random() * 360.
        rotx,roty,rotz = randrot(), randrot(), randrot()
        self.plush_rotationOp.Set(rpy2quat(rotx,roty,rotz))
        print("Finished reseting plush pose...")
        print("Reseting scene...")
        self._randomize_scene()
        print("Finished reseting scene...")
        self.kit.play()

        # wait until stable
        if self.RESET_STATIC:
            print("Waiting to reach stable...")
            for _ in range(self.DROP_MIN):
                self.step()
            for ff in range(self.FALL_MAX*6):
                self.step()
                if self.check_scene_static():
                    print(f"Initial configuration becomes static after {ff} steps")
                    break
            print("Reset Finished")
        self.frame = 0

    def reset_to(self, state):
        self.kit.stop()
        loc = state['loc']
        rot = state['rot']
        sim = state['sim']
        vis = state['vis']
        self._reset_plush_geometry(sim, vis)
        self.plush_translateOp.Set(loc)
        self.plush_rotationOp.Set(rot)
        self.kit.play()

    def check_scene_static(self):
        _,_,_,v = self._get_object_velocity_stats()
        return v < self.REST_THRESHOLD

    def get_scene_metadata(self):
        from pxr import PhysxSchema
        sbAPI = PhysxSchema.PhysxDeformableAPI(self.plush)
        faces = sbAPI.GetSimulationIndicesAttr().Get()
        return {'plush_path': self.PLUSH_ANIMAL_PATH,
                'sim_faces':np.array(faces, int).tolist(),
                'sim_pts':np.array(self.sim_og_pts, np.float16).tolist(),
                'vis_pts':np.array(self.vis_og_pts, np.float16).tolist(),
                'scene_range': self.scene_range.tolist(),
                'back_clutter_range': self.back_clutter_range.tolist(),
                'cam_info': self._get_camera_info()}

    # background state is different per reset
    def get_scene_background_state(self):
        collider = {}
        for p in find_immediate_children(self.front_group):
            name = str(p.GetPath()).split("/")[-1]
            e,f = find_collider(p)
            collider[f"{name}_box"] = e
            collider[f"{name}_tran"] = f
        for p in find_immediate_children(self.back_group):
            name = str(p.GetPath()).split("/")[-1]
            e,f = find_collider(p)
            collider[f"{name}_box"] = e
            collider[f"{name}_tran"] = f
        return collider

    def get_scene_state_plush(self,raw=False,convert_to=None):
        sim,vis = self._get_plush_points()
        loc,rot,scale = self._get_plush_loc(),self._get_plush_rot(),self._get_plush_scale()
        if not raw:
            loc,rot,scale = tuple(loc),eval(str(rot)),tuple(scale)
        state = {'sim':sim, 'vis':vis,
                 'loc':loc, 'rot':rot, 'scale':scale}
        if convert_to is not None:
            for k,v in state.items():
                state[k] = np.array(v, convert_to)
        return state

    def get_observations(self, 
                sensors=["rgb","depth",
                        #  "instanceSegmentation",
                         "semanticSegmentation",],
                partial_pointcloud=False):
        frame = self.sd_helper.get_groundtruth(sensors, self._viewport)
        gt = {}
        gt['rgb_img'] = frame['rgb'][:,:,:-1]
        gt['seg_img'] = frame['semanticSegmentation']
        gt['dep_img'] = frame['depth'].squeeze()
        if partial_pointcloud:
            gt['pxyz'] = get_partial_point_cloud(self._viewport, project_factor=100.)
        return gt
   
    ################################################################
    #
    # Below are "private" functions ;) 
    #
    ################################################################
   
    def _import_plush_animal(self, usda_path):
        from omni.physx.scripts import physicsUtils
        mesh_name = usda_path.split('/')[-1].split('.')[0]
        from pxr import PhysxSchema,UsdGeom,UsdShade,Semantics
        ###################
        # import object
        abspath = carb.tokens.get_tokens_interface().resolve(usda_path)
        physics_root = "/Root"
        assert self.stage.DefinePrim(physics_root+f"/{mesh_name}").GetReferences().AddReference(abspath)
        self.mesh_path = f"{physics_root}/{mesh_name}/{mesh_name}_obj/mesh"
        self.plush= self.stage.GetPrimAtPath(self.mesh_path)
        ###################
        # add deformable property
        schema_parameters = {
                "self_collision": True,
                "vertex_velocity_damping": 0.005,
                "sleep_damping": 10, 
                "sleep_threshold": 5,
                "settling_threshold": 11,
                "solver_position_iteration_count": 60,
                "collisionRestOffset": 0.1,
                "collisionContactOffset": 0.5,
                "voxel_resolution": 45,
            }
        skin_mesh = UsdGeom.Mesh.Get(self.stage, self.mesh_path)
        skin_mesh.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, 300.0))
        skin_mesh.AddOrientOp().Set(Gf.Quatf(0.707, 0.707, 0, 0))
        skin_points = skin_mesh.GetPointsAttr().Get() 
        skin_indices = physicsUtils.triangulateMesh(skin_mesh)

        # Create tet meshes for simulation and collision based on the skin mesh
        simulation_resolution = schema_parameters["voxel_resolution"]
        skin_mesh_scale = Gf.Vec3f(1.0, 1.0, 1.0)
        collision_points, collision_indices = physicsUtils.create_conforming_tetrahedral_mesh(skin_points, skin_indices)
        simulation_points, simulation_indices = physicsUtils.create_voxel_tetrahedral_mesh(collision_points, collision_indices, skin_mesh_scale, simulation_resolution)

        # Apply PhysxDeformableBodyAPI and PhysxCollisionAPI to skin mesh and set parameter and tet meshes
        deformable_body_api = PhysxSchema.PhysxDeformableBodyAPI.Apply(skin_mesh.GetPrim())
        deformable_body_api.CreateSolverPositionIterationCountAttr().Set(schema_parameters['solver_position_iteration_count'])
        deformable_body_api.CreateSelfCollisionAttr().Set(schema_parameters['self_collision'])
        deformable_body_api.CreateCollisionIndicesAttr().Set(collision_indices)
        deformable_body_api.CreateCollisionRestPointsAttr().Set(collision_points)
        deformable_body_api.CreateSimulationIndicesAttr().Set(simulation_indices)
        deformable_body_api.CreateSimulationRestPointsAttr().Set(simulation_points)
        deformable_body_api.CreateVertexVelocityDampingAttr().Set(schema_parameters['vertex_velocity_damping'])
        deformable_body_api.CreateSleepDampingAttr().Set(schema_parameters['sleep_damping'])
        deformable_body_api.CreateSleepThresholdAttr().Set(schema_parameters['sleep_threshold'])
        deformable_body_api.CreateSettlingThresholdAttr().Set(schema_parameters['settling_threshold'])
        PhysxSchema.PhysxCollisionAPI.Apply(skin_mesh.GetPrim())
        ###################
        # add deformable material 
        def add_physics_material_to_prim(stage, prim, materialPath):
            bindingAPI = UsdShade.MaterialBindingAPI.Apply(prim)
            materialPrim = UsdShade.Material(stage.GetPrimAtPath(materialPath))
            bindingAPI.Bind(materialPrim, UsdShade.Tokens.weakerThanDescendants, "physics")
        add_physics_material_to_prim(self.stage, self.plush, self.plush_animal_mat)
        ###################
        # add collision group
        physicsUtils.add_collision_to_collision_group(self.stage, self.mesh_path, self.real_object_group)
        ###################
        # add semantic info
        sem = Semantics.SemanticsAPI.Apply(self.stage.GetPrimAtPath(self.mesh_path), "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("plush")
        ###################
        # standarize transform
        physicsUtils.setup_transform_as_scale_orient_translate(self.plush)
        xform = UsdGeom.Xformable(self.plush)
        ops = xform.GetOrderedXformOps()
        self.plush_translateOp = ops[0]
        self.plush_rotationOp = ops[1]
        self.plush_scaleOp = ops[2]
        scale_factor = self.PLUSH_SCALE
        self.plush_scaleOp.Set((scale_factor,scale_factor,scale_factor))

    def _get_object_velocity_stats(self):
        from pxr import PhysxSchema
        sbAPI = PhysxSchema.PhysxDeformableAPI(self.plush)
        velocity = np.array(sbAPI.GetSimulationVelocitiesAttr().Get())
        vnorm = np.linalg.norm(velocity, axis=1)
        return np.percentile(vnorm, [0,50,90,99])

    def _setup_robots(self):
        actor = self.stage.GetPrimAtPath(self.magic_gripper)
        fingerL = self.stage.GetPrimAtPath(self.fingerL)
        fingerR = self.stage.GetPrimAtPath(self.fingerR)
        self.gripper = magic_eef(actor, 
                self.stage, 
                eef_default_loc=(0.,0.,600.),
                default_speed=self.ROBOT_SPEED,
                fingerL=fingerL,
                fingerR=fingerR)
    
    def _setup_cameras(self):
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        # Need to set this before setting viewport window size
        carb.settings.acquire_settings_interface().set_int("/app/renderer/resolution/width", -1)
        carb.settings.acquire_settings_interface().set_int("/app/renderer/resolution/height", -1)

        viewport_window = omni.kit.viewport.get_default_viewport_window()
        viewport_window.set_active_camera("/Root/cam_light/Camera")
        viewport_window.set_texture_resolution(RESOLUTION,RESOLUTION)
        viewport_window.set_window_size(RESOLUTION, RESOLUTION)

    def _get_plush_loc(self):
        return self.plush_translateOp.Get()

    def _get_plush_rot(self):
        return self.plush_rotationOp.Get()

    def _get_plush_scale(self):
        return self.plush_scaleOp.Get()

    def _get_plush_points(self):
        from pxr import PhysxSchema
        sbAPI = PhysxSchema.PhysxDeformableBodyAPI(self.plush)
        sim = sbAPI.GetSimulationPointsAttr().Get()
        mesh = UsdGeom.Mesh(self.plush)
        vis = mesh.GetPointsAttr().Get()
        return sim, vis

    def _get_camera_info(self):
        cam_info = {}
        camera_pose, camera_intr = get_camera_params(self._viewport)
        cam_name = get_camera_name(self._viewport)
        cam_info[cam_name] = [camera_pose.tolist(), camera_intr.tolist()]
        return cam_info
    
    def _randomize_collection(self, collection_prim, scene_range, drop_range=None, rand_rot=True, padding=True):
        extents,objs = [],[]
        for p in find_immediate_children(collection_prim):
            objs.append(str(p.GetPath()))
            extent, transform = find_collider(p)
            extents.append(transform_verts(extent, transform))
        objects = [standardize_bbox(bbox) for bbox in  np.array(extents)[:,:,:-1]]
        canvas = get_canvas(scene_range)
        if drop_range is not None:
            fill_canvas(canvas, scene_range, drop_range)
        translations = []
        for b,n in zip(objects,objs):
            for _ in range(3):
                t = sample_bbox_translation(b, scene_range)
                if padding:
                    tb = scale(pad_to_square(b + t))
                else:
                    tb = b + t 
                if not overlaps_with_current(canvas, scene_range, tb):
                    fill_canvas(canvas, scene_range, tb)
                    translations.append((n,t))
                    break
            if len(translations) == 0 or translations[-1][0] != n:
                translations.append((n,np.array([0,-2000])))
        def randrot():
            return random.random() * 360.
        from pxr import UsdGeom
        from omni.physx.scripts import physicsUtils
        for n,t in translations:
            xform = UsdGeom.Xformable(self.stage.GetPrimAtPath(n))
            physicsUtils.setup_transform_as_scale_orient_translate(xform)
            ops = xform.GetOrderedXformOps()
            translateOp = ops[0] 
            translateOp.Set(tuple(np.array(tuple(translateOp.Get())) + np.append(t, 0)))
            if rand_rot:
                orientOp = ops[1]
                orientOp.Set(rpy2quat(0,0,randrot()))
    
    def _randomize_lighting(self):
        domelight = self.stage.GetPrimAtPath("/Root/cam_light/Lights/DomeLight")
        light = self.stage.GetPrimAtPath("/Root/cam_light/Lights/DistantLight")
        light1 = self.stage.GetPrimAtPath("/Root/cam_light/Lights/DistantLight_01")
        temp = np.random.rand(1)[0] * 5000 + 2500
        domelight.GetAttribute('colorTemperature').Set(temp)
        light.GetAttribute('colorTemperature').Set(temp)
        light1.GetAttribute('colorTemperature').Set(temp)
        int_range = 10000
        int_min = 2500
        for l in [domelight, light, light1]:
            intensity = np.random.rand(1)[0] * int_range + int_min
            l.GetAttribute('intensity').Set(intensity)

    def _randomize_scene(self):
        if self.RAND_LAYOUT:
            # randomize front scene
            self._randomize_collection(self.front_group, self.scene_range[:,:-1], self.drop_range)
            # randomize back scene
            self._randomize_collection(self.back_group, self.back_clutter_range,rand_rot=False, padding=False)
        if self.RAND_LIGHTS:
            # randomize lights
            self._randomize_lighting()

    def _get_2d_layout_occupancy_map(self):
        extents = []
        for p in find_immediate_children(self.front_group):
            extent, transform = find_collider(p)
            extents.append(transform_verts(extent, transform))
        for p in find_immediate_children(self.back_group):
            extent, transform = find_collider(p)
            extents.append(transform_verts(extent, transform))
        objects = [standardize_bbox(bbox) for bbox in  np.array(extents)[:,:,:-1]]
        #canvas = get_canvas(self.scene_range[:,:-1])
        canvas = get_canvas(self.total_range[:,:-1])
        for b in objects:
            fill_canvas(canvas, self.total_range[:,:-1], b)
        return canvas

    def _sample_displacement_vector(self, grasp_point):
        sampled_for = 0
        mean_len = 160
        std_len = 80
        max_len = 240
        min_len = 80
        canvas = self._get_2d_layout_occupancy_map()
        while(True):
            sampled_for = sampled_for + 1
            move_len = np.clip(np.random.normal(loc=mean_len,scale=std_len), min_len, max_len)
            move_dir = sample_direction_zup(100).squeeze()
            #move_dir[1,:] = np.abs(move_dir[1,:])
            move_vec = move_dir * move_len
            target_pts = grasp_point + move_vec.T
            in_world = np.logical_and(
                target_pts > self.total_range[0], 
                target_pts < self.total_range[1]).all(axis=1)
            occupancies = []
            try:
                # assure that no obstacle is in path for length times 1.3
                for i in range(int(max_len*1.3)):
                    temp = grasp_point + (target_pts - grasp_point) / max_len * i
                    temp[:,0] = np.clip(target_pts[:,0], self.total_range[0,0], self.total_range[1,0])
                    temp[:,1] = np.clip(target_pts[:,1], self.total_range[0,1], self.total_range[1,1])
                    occupancies.append(get_occupancy_value(
                        canvas, self.total_range[:,:-1], temp[:,:-1]))
                path_no_collision = (np.array(occupancies) == 0).all(axis=0)
                viable = np.logical_and(in_world, path_no_collision)
                in_idx = np.nonzero(viable)[0]
            except:
                continue
            if len(in_idx) > 0:
                target_point = target_pts[np.random.choice(in_idx)]
                return target_point
            else:
                if sampled_for > 10:
                    break
        return None

    def _reset_plush_geometry(self, sim, vis):
        from pxr import PhysxSchema, Gf, Vt
        # reset simulation points
        sbAPI = PhysxSchema.PhysxDeformableBodyAPI(self.plush)
        sbAPI.GetSimulationPointsAttr().Set(sim)
        # reset simulation points velocity
        sbAPI = PhysxSchema.PhysxDeformableAPI(self.plush)
        velocity = np.array(sbAPI.GetSimulationVelocitiesAttr().Get())
        zero_velocity = np.zeros_like(velocity)
        velocity_vec = Vt.Vec3fArray([Gf.Vec3f(tuple(m)) for m in zero_velocity])
        sbAPI.GetSimulationVelocitiesAttr().Set(velocity_vec)
        # reset visual points
        mesh = UsdGeom.Mesh(self.plush)
        mesh.GetPointsAttr().Set(vis)