#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import argparse
import json

from utils import *
parser = argparse.ArgumentParser("Dataset generation")
################################################################
# save to args
parser.add_argument("--save_dir", type=str, default="/result/interaction_sequence")
parser.add_argument("--img_subdir", type=str, default='img')
parser.add_argument("--geom_subdir", type=str, default='geom')
parser.add_argument("--info_subdir", type=str, default='info')
parser.add_argument("--save_every", type=int, default=25)

################################################################
# interaction args
parser.add_argument("--num_interaction", type=int, default=18)
parser.add_argument("--reset_every", type=int, default=6)

################################################################
# scene args
parser.add_argument("--asset_root", type=str, default="/result/assets")
parser.add_argument("--scene_path", type=str, default="attic_lean/Attic_clean_v2.usda")
parser.add_argument("--plush_path", type=str, default="animals/teddy/teddy_scaled/teddy_scaled.usda")
parser.add_argument("--skip_layout_randomization", action="store_true", default=False)
parser.add_argument("--skip_lights_randomization", action="store_true", default=False)

args = parser.parse_args()

os.makedirs(args.save_dir, exist_ok=True)
os.makedirs(os.path.join(args.save_dir,  args.img_subdir), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, args.geom_subdir), exist_ok=True)
os.makedirs(os.path.join(args.save_dir, args.info_subdir), exist_ok=True)
img_dir  = os.path.join(args.save_dir,  args.img_subdir)
geom_dir = os.path.join(args.save_dir, args.geom_subdir)
info_dir = os.path.join(args.save_dir, args.info_subdir)

def main():
    from attic_scene import attic_scene
    scene_path = os.path.join(args.asset_root, args.scene_path)
    plush_path = os.path.join(args.asset_root, args.plush_path)
    scene = attic_scene(
        scene_path, 
        plush_path, 
        RESET_STATIC=True, 
        RAND_LAYOUT=not args.skip_layout_randomization,
        RAND_LIGHTS=not args.skip_lights_randomization,)

    start_time = time.time()
    # save scene overall info
    with open(os.path.join(info_dir, "scene_meta.json"), 'w') as fp:
        json.dump(scene.get_scene_metadata(), fp)

    # number of resets
    num_resets = (args.num_interaction + args.reset_every - 1) // args.reset_every 
    for reset in range(num_resets):
        # save scene reset collider info
        np.savez_compressed(os.path.join(info_dir, f"clutter_info_{reset:04d}.npz"), **scene.get_scene_background_state())

        num_steps = min(args.num_interaction, (reset + 1) * args.reset_every) -  reset * args.reset_every
        # sample interactions 
        actions = {
            'grasp_points':[],
            'target_points':[],
            'grasp_pixels':[],
            'start_frames':[],
            'release_frames':[],
            'static_frames':[], }

        # save start frame    
        save_frame(f"{reset:04d}_{scene.frame:06d}", scene.get_observations(), img_dir)
        np.savez_compressed(
            os.path.join(geom_dir, f"{reset:04d}_{scene.frame:06d}.npz"), 
            **scene.get_scene_state_plush(convert_to=np.float16))

        for interaction in range(num_steps):
            # stop simulating
            scene.kit.pause()
            action = scene.sample_action() 
            if action is None:
                scene.kit.play()
                continue
            grasp_point, target_point, grasp_pixel =  action
            actions['grasp_points'].append(np.array(grasp_point,np.float16))
            actions['target_points'].append(np.array(target_point,np.float16))
            actions['grasp_pixels'].append(np.array(grasp_pixel,np.uint16))
            actions['start_frames'].append(np.array(scene.frame,np.uint16))

            save_frame(f"{reset:04d}_{scene.frame:06d}", scene.get_observations(), img_dir)
            np.savez_compressed(
                os.path.join(geom_dir, f"{reset:04d}_{scene.frame:06d}.npz"), 
                **scene.get_scene_state_plush(convert_to=np.float16))


            scene.kit.play()
            
            init_traj = scene.gripper.plan_trajectory(scene.gripper.eef_default_loc, grasp_point)
            # move
            for pos in init_traj:
                scene.step()
                scene.gripper.set_translation(tuple(pos))
                if scene.frame % args.save_every == args.save_every - 1:
                    save_frame(f"{reset:04d}_{scene.frame:06d}", scene.get_observations(), img_dir)
                    np.savez_compressed(
                        os.path.join(geom_dir, f"{reset:04d}_{scene.frame:06d}.npz"), 
                        **scene.get_scene_state_plush(convert_to=np.float16))
            
            scene.kit.pause()
            #init_move_traj = scene.gripper.set_translation(grasp_point)
            scene.gripper.grasp(scene.plush)

            scene.kit.play()
            traj = scene.gripper.plan_trajectory(grasp_point, target_point)

            # move
            for pos in traj:
                scene.step()
                scene.gripper.set_translation(tuple(pos))
                if scene.frame % args.save_every == args.save_every - 1:
                    save_frame(f"{reset:04d}_{scene.frame:06d}", scene.get_observations(), img_dir)
                    np.savez_compressed(
                        os.path.join(geom_dir, f"{reset:04d}_{scene.frame:06d}.npz"), 
                        **scene.get_scene_state_plush(convert_to=np.float16))

            # wait until stable
            for ff in range(scene.FALL_MAX):
                scene.step()
                if scene.check_scene_static():
                    print(f"grasp reaching a resting state after {ff} steps")
                    break

            save_frame(f"{reset:04d}_{scene.frame:06d}", scene.get_observations(), img_dir)
            np.savez_compressed(
                os.path.join(geom_dir, f"{reset:04d}_{scene.frame:06d}.npz"), 
                **scene.get_scene_state_plush(convert_to=np.float16))
            actions['release_frames'].append(np.array(scene.frame,np.uint16))

            # release
            scene.kit.pause()
            scene.gripper.ungrasp()
            # TODO: delete gripper collider
            scene.kit.play()

            for ff in range(scene.FALL_MAX+scene.DROP_MIN):
                scene.step()
                if scene.frame % args.save_every == args.save_every - 1:
                    save_frame(f"{reset:04d}_{scene.frame:06d}", scene.get_observations(), img_dir)
                    np.savez_compressed(
                        os.path.join(geom_dir, f"{reset:04d}_{scene.frame:06d}.npz"), 
                        **scene.get_scene_state_plush(convert_to=np.float16))
                if ff < scene.DROP_MIN:
                    continue
                if scene.check_scene_static():
                    print(f"release reaching a resting state after {ff} steps")
                    break
            scene.gripper.reset_translation()

            save_frame(f"{reset:04d}_{scene.frame:06d}", scene.get_observations(), img_dir)
            np.savez_compressed(
                os.path.join(geom_dir, f"{reset:04d}_{scene.frame:06d}.npz"), 
                **scene.get_scene_state_plush(convert_to=np.float16))
            actions['static_frames'].append(np.array(scene.frame,np.uint16))

        np.savez_compressed(os.path.join(info_dir, f"interaction_info_{reset:04d}.npz"), **actions)
        end_time = time.time()
        from datetime import timedelta
        time_str = str(timedelta(seconds=end_time - start_time))
        print(f'Sampling {num_steps} interactions takes: {time_str}')

        scene.reset()

    # cleanup
    scene.kit.shutdown()


if __name__ == "__main__":
    main()
