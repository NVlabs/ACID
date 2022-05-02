#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import carb
import omni.kit.app
import omni.kit

import os
import sys
import time
import asyncio
import argparse

DEFAULT_CONFIG = {
    "width": 1024,
    "height": 800,
    "renderer": "PathTracing",  # Can also be RayTracedLighting
    "anti_aliasing": 3,  # 3 for dlss, 2 for fxaa, 1 for taa, 0 to disable aa
    "samples_per_pixel_per_frame": 64,
    "denoiser": True,
    "subdiv_refinement_level": 0,
    "headless": True,
    "max_bounces": 4,
    "max_specular_transmission_bounces": 6,
    "max_volume_bounces": 4,
    "sync_loads": False,
    "experience": f'{os.environ["EXP_PATH"]}/omni.bloky.python.kit',

}


class OmniKitHelper:
    """Helper class for launching OmniKit from a Python environment.

    Launches and configures OmniKit and exposes useful functions.

        Typical usage example:

        .. highlight:: python
        .. code-block:: python

            config = {'width': 800, 'height': 600, 'renderer': 'PathTracing'}
            kit = OmniKitHelper(config)   # Start omniverse kit
            # <Code to generate or load a scene>
            kit.update()    # Render a single frame"""

    def __init__(self, config=DEFAULT_CONFIG):
        """The config variable is a dictionary containing the following entries

        Args:
            width (int): Width of the viewport and generated images. Defaults to 1024
            height (int): Height of the viewport and generated images. Defaults to 800
            renderer (str): Rendering mode, can be  `RayTracedLighting` or `PathTracing`. Defaults to `PathTracing`
            samples_per_pixel_per_frame (int): The number of samples to render per frame, used for `PathTracing` only. Defaults to 64
            denoiser (bool):  Enable this to use AI denoising to improve image quality. Defaults to True
            subdiv_refinement_level (int): Number of subdivisons to perform on supported geometry. Defaults to 0
            headless (bool): Disable UI when running. Defaults to True
            max_bounces (int): Maximum number of bounces, used for `PathTracing` only. Defaults to 4
            max_specular_transmission_bounces(int): Maximum number of bounces for specular or transmission, used for `PathTracing` only. Defaults to 6
            max_volume_bounces(int): Maximum number of bounces for volumetric, used for `PathTracing` only. Defaults to 4
            sync_loads (bool): When enabled, will pause rendering until all assets are loaded. Defaults to False
            experience (str): The config json used to launch the application.
        """
        # only import custom loop runner if we create this object
        # from omni.kit.loop import _loop

        # initialize vars
        self._exiting = False
        self._is_dirty_instance_mappings = True
        self._previous_physics_dt = 1.0 / 60.0
        self.config = DEFAULT_CONFIG
        if config is not None:
            self.config.update(config)

        # Load app plugin
        self._framework = carb.get_framework()
        print(os.environ["CARB_APP_PATH"])
        self._framework.load_plugins(
            loaded_file_wildcards=["omni.kit.app.plugin"],
            search_paths=[os.path.abspath(f'{os.environ["CARB_APP_PATH"]}/kit/plugins')],
        )
        print(DEFAULT_CONFIG)

        # launch kit
        self.last_update_t = time.time()
        self.app = omni.kit.app.get_app()
        self.kit_settings = None
        self._start_app()
        self.carb_settings = carb.settings.acquire_settings_interface()
        self.setup_renderer(mode="default")  # set rtx-defaults settings
        self.setup_renderer(mode="non-default")  # set rtx settings

        self.timeline = omni.timeline.get_timeline_interface()

        # Wait for new stage to open
        new_stage_task = asyncio.ensure_future(omni.usd.get_context().new_stage_async())
        print("OmniKitHelper Starting up ...")
        while not new_stage_task.done():
            time.sleep(0.001)  # This sleep prevents a deadlock in certain cases
            self.update()
        self.update()
        # Dock windows  if they exist
        main_dockspace = omni.ui.Workspace.get_window("DockSpace")

        def dock_window(space, name, location):
            window = omni.ui.Workspace.get_window(name)
            if window and space:
                window.dock_in(space, location)
            return window

        view = dock_window(main_dockspace, "Viewport", omni.ui.DockPosition.TOP)
        self.update()
        console = dock_window(view, "Console", omni.ui.DockPosition.BOTTOM)
        prop = dock_window(view, "Property", omni.ui.DockPosition.RIGHT)
        dock_window(view, "Main ToolBar", omni.ui.DockPosition.LEFT)
        self.update()
        dock_window(prop, "Render Settings", omni.ui.DockPosition.SAME)
        self.update()
        print("OmniKitHelper Startup Complete")

    def _start_app(self):
        args = [
            os.path.abspath(__file__),
            f'{self.config["experience"]}',
            "--/persistent/app/viewport/displayOptions=0",  # hide extra stuff in viewport
            # Forces kit to not render until all USD files are loaded
            f'--/rtx/materialDb/syncLoads={self.config["sync_loads"]}',
            f'--/rtx/hydra/materialSyncLoads={self.config["sync_loads"]}'
            f'--/omni.kit.plugin/syncUsdLoads={self.config["sync_loads"]}',
            "--/app/content/emptyStageOnStart=False",  # This is required due to a infinite loop but results in errors on launch
            "--/app/hydraEngine/waitIdle=True",
            "--/app/asyncRendering=False",
            f'--/app/renderer/resolution/width={self.config["width"]}',
            f'--/app/renderer/resolution/height={self.config["height"]}',
        ]

        args.append(f"--portable")
        args.append(f"--no-window")
        args.append(f"--allow-root")
        print(args)
        self.app.startup("kit", f'{os.environ["CARB_APP_PATH"]}/kit', args)

    def __del__(self):
        if self._exiting is False and sys.meta_path is None:
            print(
                "\033[91m"
                + "ERROR: Python exiting while OmniKitHelper was still running, Please call shutdown() on the OmniKitHelper object to exit cleanly"
                + "\033[0m"
            )

    def shutdown(self):
        self._exiting = True
        print("Shutting Down OmniKitHelper...")
        # We are exisitng but something is still loading, wait for it to load to avoid a deadlock
        if self.is_loading():
            print("   Waiting for USD resource operations to complete (this may take a few seconds)")
        while self.is_loading():
            self.app.update()
        self.app.shutdown()
        self._framework.unload_all_plugins()
        print("Shutting Down Complete")

    def get_stage(self):
        """Returns the current USD stage."""
        return omni.usd.get_context().get_stage()

    def set_setting(self, setting, value):
        """Convenience function to set settings.

        Args:
            setting (str): string representing the setting being changed
            value: new value for the setting being changed, the type of this value must match its repsective setting
        """
        if isinstance(value, str):
            self.carb_settings.set_string(setting, value)
        elif isinstance(value, bool):
            self.carb_settings.set_bool(setting, value)
        elif isinstance(value, int):
            self.carb_settings.set_int(setting, value)
        elif isinstance(value, float):
            self.carb_settings.set_float(setting, value)
        else:
            raise ValueError(f"Value of type {type(value)} is not supported.")

    def set_physics_dt(self, physics_dt: float = 1.0 / 150.0, physics_substeps: int = 1):
        """Specify the physics step size to use when simulating, default is 1/60.
        Note that a physics scene has to be in the stage for this to do anything

        Args:
            physics_dt (float): Use this value for physics step
        """
        if self.get_stage() is None:
            return
        if physics_dt == self._previous_physics_dt:
            return
        if physics_substeps is None or physics_substeps <= 1:
            physics_substeps = 1
        self._previous_physics_dt = physics_dt
        from pxr import UsdPhysics, PhysxSchema

        steps_per_second = int(1.0 / physics_dt)
        min_steps = int(steps_per_second / physics_substeps)
        physxSceneAPI = None
        for prim in self.get_stage().Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(prim)
        if physxSceneAPI is not None:
            physxSceneAPI.GetTimeStepsPerSecondAttr().Set(steps_per_second)

        settings = carb.settings.get_settings()
        settings.set_int("persistent/simulation/minFrameRate", min_steps)

    def update(self, dt=0.0, physics_dt=None, physics_substeps=None):
        """Render one frame. Optionally specify dt in seconds, specify None to use wallclock.
        Specify physics_dt and  physics_substeps to decouple the physics step size from rendering

        For example: to render with a dt of 1/30 and simulate physics at 1/120 use:
            - dt = 1/30.0
            - physics_dt = 1/120.0
            - physics_substeps = 4

        Args:
            dt (float): The step size used for the overall update, set to None to use wallclock
            physics_dt (float, optional): If specified use this value for physics step
            physics_substeps (int, optional): Maximum number of physics substeps to perform
        """
        # dont update if exit was called
        if self._exiting:
            return
        # a physics dt was specified and is > 0
        if physics_dt is not None and physics_dt > 0.0:
            self.set_physics_dt(physics_dt, physics_substeps)
        # a dt was specified and is > 0
        if dt is not None and dt > 0.0:
            # if physics dt was not specified, use rendering dt
            if physics_dt is None:
                self.set_physics_dt(dt)
            # self.loop_runner.set_runner_dt(dt)
            self.app.update()
        else:
            # dt not specified, run in realtime
            time_now = time.time()
            dt = time_now - self.last_update_t
            if physics_dt is None:
                self.set_physics_dt(1.0 / 60.0, 4)
            self.last_update_t = time_now
            # self.loop_runner.set_runner_dt(dt)
            self.app.update()

    def play(self):
        """Starts the editor physics simulation"""
        self.update()
        self.timeline.play()
        self.update()

    def pause(self):
        """Pauses the editor physics simulation"""
        self.update()
        self.timeline.pause()
        self.update()

    def stop(self):
        """Stops the editor physics simulation"""
        self.update()
        self.timeline.stop()
        self.update()

    def get_status(self):
        """Get the status of the renderer to see if anything is loading"""
        return omni.usd.get_context().get_stage_loading_status()

    def is_loading(self):
        """convenience function to see if any files are being loaded

        Returns:
            bool: True if loading, False otherwise
        """
        message, loaded, loading = self.get_status()
        return loading > 0

    def is_exiting(self):
        """get current exit status for this object
        Returns:
            bool: True if exit() was called previously, False otherwise
        """
        return self._exiting

    def execute(self, *args, **kwargs):
        """Allow use of omni.kit.commands interface"""
        omni.kit.commands.execute(*args, **kwargs)

    def setup_renderer(self, mode="non-default"):
        rtx_mode = "/rtx-defaults" if mode == "default" else "/rtx"
        """Reset render settings to those in config. This should be used in case a new stage is opened and the desired config needs to be re-applied"""
        self.set_setting(rtx_mode + "/rendermode", self.config["renderer"])
        # Raytrace mode settings
        self.set_setting(rtx_mode + "/post/aa/op", self.config["anti_aliasing"])
        self.set_setting(rtx_mode + "/directLighting/sampledLighting/enabled", True)
        # self.set_setting(rtx_mode + "/ambientOcclusion/enabled", True)
        # Pathtrace mode settings
        self.set_setting(rtx_mode + "/pathtracing/spp", self.config["samples_per_pixel_per_frame"])
        self.set_setting(rtx_mode + "/pathtracing/totalSpp", self.config["samples_per_pixel_per_frame"])
        self.set_setting(rtx_mode + "/pathtracing/clampSpp", self.config["samples_per_pixel_per_frame"])
        self.set_setting(rtx_mode + "/pathtracing/maxBounces", self.config["max_bounces"])
        self.set_setting(
            rtx_mode + "/pathtracing/maxSpecularAndTransmissionBounces",
            self.config["max_specular_transmission_bounces"],
        )
        self.set_setting(rtx_mode + "/pathtracing/maxVolumeBounces", self.config["max_volume_bounces"])
        self.set_setting(rtx_mode + "/pathtracing/optixDenoiser/enabled", self.config["denoiser"])
        self.set_setting(rtx_mode + "/hydra/subdivision/refinementLevel", self.config["subdiv_refinement_level"])

        # Experimental, forces kit to not render until all USD files are loaded
        self.set_setting(rtx_mode + "/materialDb/syncLoads", self.config["sync_loads"])
        self.set_setting(rtx_mode + "/hydra/materialSyncLoads", self.config["sync_loads"])
        self.set_setting("/omni.kit.plugin/syncUsdLoads", self.config["sync_loads"])

    def create_prim(
        self, path, prim_type, translation=None, rotation=None, scale=None, ref=None, semantic_label=None, attributes={}
    ):
        """Create a prim, apply specified transforms, apply semantic label and
        set specified attributes.

        args:
            path (str): The path of the new prim.
            prim_type (str): Prim type name
            translation (tuple(float, float, float), optional): prim translation (applied last)
            rotation (tuple(float, float, float), optional): prim rotation in radians with rotation
                order ZYX.
            scale (tuple(float, float, float), optional): scaling factor in x, y, z.
            ref (str, optional): Path to the USD that this prim will reference.
            semantic_label (str, optional): Semantic label.
            attributes (dict, optional): Key-value pairs of prim attributes to set.
        """
        from pxr import UsdGeom, Semantics

        prim = self.get_stage().DefinePrim(path, prim_type)

        for k, v in attributes.items():
            prim.GetAttribute(k).Set(v)
        xform_api = UsdGeom.XformCommonAPI(prim)
        if ref:
            prim.GetReferences().AddReference(ref)
        if semantic_label:
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr()
            sem.CreateSemanticDataAttr()
            sem.GetSemanticTypeAttr().Set("class")
            sem.GetSemanticDataAttr().Set(semantic_label)
        if rotation:
            xform_api.SetRotate(rotation, UsdGeom.XformCommonAPI.RotationOrderXYZ)
        if scale:
            xform_api.SetScale(scale)
        if translation:
            xform_api.SetTranslate(translation)
        return prim

    def set_up_axis(self, axis):
        """Change the up axis of the current stage

        Args:
            axis: valid values are `UsdGeom.Tokens.y`, or `UsdGeom.Tokens.z`
        """
        from pxr import UsdGeom, Usd

        stage = self.get_stage()
        rootLayer = stage.GetRootLayer()
        rootLayer.SetPermissionToEdit(True)
        with Usd.EditContext(stage, rootLayer):
            UsdGeom.SetStageUpAxis(stage, axis)
