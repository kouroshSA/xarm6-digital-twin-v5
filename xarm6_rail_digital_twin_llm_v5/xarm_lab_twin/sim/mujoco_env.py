# sim/mujoco_env.py
import mujoco
import mujoco.viewer
import numpy as np
import threading
import time
from typing import Optional

try:
    from transforms3d.euler import mat2euler
    HAS_TRANSFORMS3D = True
except ImportError:
    HAS_TRANSFORMS3D = False

JOINT_NAMES = ["joint1","joint2","joint3","joint4","joint5","joint6"]
ACT_NAMES   = ["act_rail","act1","act2","act3","act4","act5","act6"]
RAIL_ACT    = 0

# Magnetic-gripper config — maps graspable body name to its weld constraint name.
# Cubes are short and grasped near their center; tubes are tall and grasped near
# their cap. The dict is one entry per graspable free body in the scene.
GRIPPABLE_BODIES = {
    # Cubes
    "red_cube":   "grip_red_cube",
    "green_cube": "grip_green_cube",
    "blue_cube":  "grip_blue_cube",
    # Falcon tubes
    "tube_L1":    "grip_tube_L1",
    "tube_L2":    "grip_tube_L2",
    "tube_L3":    "grip_tube_L3",
    "tube_R1":    "grip_tube_R1",
    "tube_R2":    "grip_tube_R2",
    "tube_R3":    "grip_tube_R3",
    # Bins (now free bodies, can be pushed around)
    "red_bin":    "grip_red_bin",
    "green_bin":  "grip_green_bin",
    "blue_bin":   "grip_blue_bin",
    # Tube racks (also free now -- pushable)
    "left_tube_rack":  "grip_left_rack",
    "right_tube_rack": "grip_right_rack",
}
# Backward-compat alias (older code still references GRIPPABLE_CUBES)
GRIPPABLE_CUBES = GRIPPABLE_BODIES

GRIPPER_REACH_M = 0.07  # 70mm from EE site to body center (Claude's move_to targets
                        # the EE site, and the EE site is ~30mm below the gripper
                        # body in world frame when the arm points down)

# Cosmetic finger animation. The two finger joints have range [0, 0.015] m
# where 0 = open, 0.015 = closed (inward). Both joints are driven to the
# same ctrl value via act_finger_l / act_finger_r. Fingers are contype=0 so
# they don't affect contacts -- they're purely a visual cue for the magnetic
# grasp. Welds still do the actual carrying.
FINGER_ACT_NAMES = ("act_finger_l", "act_finger_r")
GRIPPER_OPEN_CTRL_M   = 0.0
GRIPPER_CLOSED_CTRL_M = 0.015

# Tube-rack slot grid (rack-local offsets). Both racks share the same 4x2 layout:
#   columns 1..4 at rel x = -0.060, -0.020, +0.020, +0.060
#   rows   1..2 at rel y = -0.020, +0.020
RACK_SLOTS = [
    (1, 1, -0.060, -0.020), (2, 1, -0.020, -0.020),
    (3, 1, +0.020, -0.020), (4, 1, +0.060, -0.020),
    (1, 2, -0.060, +0.020), (2, 2, -0.020, +0.020),
    (3, 2, +0.020, +0.020), (4, 2, +0.060, +0.020),
]
# Tube body center sits at this world z when seated in a slot (rack base plate
# top at z=0.760, tube half-height 0.0575, so center at 0.760+0.0575 = 0.8175)
TUBE_SLOT_Z_M = 0.8175
# Tube bodies (used to determine which slots are currently occupied)
TUBE_BODY_NAMES = ("tube_L1", "tube_L2", "tube_L3",
                   "tube_R1", "tube_R2", "tube_R3")
SLOT_OCCUPIED_TOL_M = 0.020   # cube/tube center within +/- this from slot xy
                              # (used by place_tube_in_rack's "which slots are
                              # already taken" scan -- intentionally loose so a
                              # roughly-aligned tube still counts as occupying
                              # its slot; do NOT confuse with the stringency
                              # tolerances below, which gate physical_outcome()
                              # reporting)

# Stringency levels for physical_outcome(). The grader's verdict (success vs.
# failure) is derived from the strings physical_outcome() emits, so tightening
# these tolerances makes outcomes harder to claim. Default is "loose" so the
# loop's existing behaviour is unchanged when callers don't pass a stringency.
#
# rack_xy_tol_m / rack_z_tol_m: how close to a slot's xy/z a tube must be
#   before it counts as "in <rack>".
# bin_xy_tol_m: how close to a bin's xy a cube must be (in addition to z
#   being inside the bin walls) before it counts as "in <bin>".
# tilt_deg_max: max tilt of a tube's local +z from world +z; tubes lying
#   sideways won't count as seated. Loose disables this check entirely.
STRINGENCY_LEVELS = {
    "loose":  {"rack_xy_tol_m": 0.020, "rack_z_tol_m": 0.030,
               "bin_xy_tol_m": 0.040, "tilt_deg_max": 90.0},
    "normal": {"rack_xy_tol_m": 0.012, "rack_z_tol_m": 0.015,
               "bin_xy_tol_m": 0.030, "tilt_deg_max": 30.0},
    "strict": {"rack_xy_tol_m": 0.006, "rack_z_tol_m": 0.006,
               "bin_xy_tol_m": 0.020, "tilt_deg_max": 10.0},
}
DEFAULT_STRINGENCY = "loose"

# When push_object operates on a rack, these tubes are welded too so the
# rack and everything inside it move as a unit. Empty (released) racks just
# move alone.
RACK_TUBE_GROUPS = {
    "left_tube_rack":  ("tube_L1", "tube_L2", "tube_L3"),
    "right_tube_rack": ("tube_R1", "tube_R2", "tube_R3"),
}


class SimXArmAPI:
    """
    Drop-in simulation replacement for xarm.wrapper.XArmAPI.

    Methods accept **kwargs to absorb extra LLM-emitted parameters
    (e.g. speed_mm_s) that don't match the SDK signature exactly.
    """

    def __init__(self, scene_xml: str, render: bool = True):
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.data  = mujoco.MjData(self.model)
        self.lock  = threading.Lock()
        self._running = True
        self._viewer = None  # set by _launch_viewer so disconnect() can close it

        self.act_ids   = [self.model.actuator(n).id for n in ACT_NAMES]
        self.joint_ids = [self.model.joint(n).id for n in JOINT_NAMES]
        self.rail_jid  = self.model.joint("rail").id
        self.ee_site   = self.model.site("end_effector").id
        self.gripper_bid = self.model.body("gripper").id

        # Optional finger actuators (cosmetic gripper animation). Missing on
        # scenes that haven't been migrated to the articulated-finger model;
        # the rest of SimXArmAPI degrades gracefully when empty.
        self.finger_act_ids = []
        for name in FINGER_ACT_NAMES:
            try:
                self.finger_act_ids.append(self.model.actuator(name).id)
            except KeyError:
                pass
        self.finger_jids = []
        for jname in ("finger_left_joint", "finger_right_joint"):
            try:
                self.finger_jids.append(self.model.joint(jname).id)
            except KeyError:
                pass
        self.cube_bids = {
            body_name: self.model.body(body_name).id
            for body_name in GRIPPABLE_BODIES
        }
        self.weld_eqids = {
            body_name: self.model.equality(weld_name).id
            for body_name, weld_name in GRIPPABLE_BODIES.items()
        }

        from sim.fk_validator import FKValidator
        from sim.ik_solver import IKSolver
        self.validator = FKValidator(self.model, self.data, self.lock)
        self.ik_solver = IKSolver(self.model, self.data, self.lock)

        # Baseline positions captured at scene-reset time. physical_outcome()
        # compares current positions against these to emit displacement
        # ("moved (Δx, Δy)mm") and proximity ("closer to <other>") facts on
        # top of the categorical events (in bin / fell to floor / etc.).
        # Initialised here so the first physical_outcome() before any
        # reset_scene() has something to compare against.
        self._initial_positions: dict = {}
        self._snapshot_positions()

        threading.Thread(target=self._sim_loop, daemon=True).start()
        if render:
            self._launch_viewer()

    def _snapshot_positions(self) -> None:
        """Record current world-frame positions of every tracked body.
        Used by physical_outcome() as the baseline for displacement +
        proximity facts. Safe to call any time -- overwrites the prior
        snapshot."""
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            snap = {}
            for name, bid in self.cube_bids.items():
                snap[name] = self.data.xpos[bid].copy()
            for fixture in ("red_bin", "green_bin", "blue_bin",
                            "left_tube_rack", "right_tube_rack"):
                try:
                    fbid = self.model.body(fixture).id
                except Exception:
                    continue
                snap[fixture] = self.data.xpos[fbid].copy()
            self._initial_positions = snap

    def _sim_loop(self):
        while self._running:
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            time.sleep(0.002)

    def _launch_viewer(self):
        def _run():
            v = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer = v   # remember handle so disconnect() can close it
            # Frame the bench/arm: lookat the working area, pull the
            # camera back enough to see the whole arm and the cubes/bins.
            v.cam.lookat[:] = (0.0, 0.15, 1.0)   # mid-scene
            v.cam.distance  = 2.8                # ~2.8 m back
            v.cam.azimuth   = 135.0              # front-right
            v.cam.elevation = -20.0              # slight downward tilt
            v.sync()
            try:
                while v.is_running() and self._running:
                    with self.lock:
                        v.sync()
                    time.sleep(0.016)
            finally:
                try:
                    v.close()
                except Exception:
                    pass
                self._viewer = None
        threading.Thread(target=_run, daemon=True).start()
        time.sleep(0.4)

    def motion_enable(self, enable: bool = True) -> int:  return 0
    def set_mode(self, mode: int) -> int:                 return 0
    def set_state(self, state: int) -> int:               return 0

    def disconnect(self):
        """Stop the sim_loop, close the viewer window if open, and let the
        viewer thread exit. After this returns the script can exit cleanly
        without waiting for the user to close the viewer manually."""
        self._running = False
        v = self._viewer
        if v is not None:
            try:
                v.close()
            except Exception:
                pass
        # Small grace period for the viewer thread to wind down its loop
        time.sleep(0.1)

    # ---- canonical home pose ----
    # A relaxed "ready" pose, not the straight-up "rocket" of all-zeros:
    #   joint1 = +90 deg  -> base rotated to face the bench (+y workspace)
    #   joint2 = +45 deg  -> shoulder tilted forward 45 deg
    #   joint3 = -45 deg  -> elbow bent 45 deg back
    #   joint5 = +30 deg  -> wrist pitched 30 deg down
    # Other joints stay at 0. Tweak by hand if you prefer a different look;
    # the value is used by go_home() and reset_scene() only.
    HOME_RAIL_M = 0.35
    HOME_JOINTS_RAD = (
        np.pi / 2,    # joint1: base rotation, +90 deg
        np.pi / 4,    # joint2: shoulder pitch forward, +45 deg
        -np.pi / 4,   # joint3: elbow bend back, -45 deg
        0.0,          # joint4
        np.pi / 6,    # joint5: wrist pitch down, +30 deg
        0.0,          # joint6
    )

    def go_home(self, wait: bool = True, **kwargs) -> int:
        """Drive rail to 350mm and all six joints to zero."""
        with self.lock:
            self.data.ctrl[self.act_ids[RAIL_ACT]] = self.HOME_RAIL_M
            for i, ang in enumerate(self.HOME_JOINTS_RAD):
                self.data.ctrl[self.act_ids[1 + i]] = float(ang)
        if wait:
            self._wait_rail_settled(self.HOME_RAIL_M)
            self._wait_arm_settled(np.array(self.HOME_JOINTS_RAD))
        return 0

    def place_tube_in_rack(self, rack_name: str, **kwargs) -> int:
        """Place currently-held tube in the first open slot of the named rack.

        Steps: find which tube is held (weld active), scan the rack's 4x2 slot
        grid for an empty slot (no tube within 20mm in xy), fly the arm above
        that slot, then snap the tube into place (teleport its free-joint qpos
        to the slot, zero its velocity, deactivate the weld). The teleport
        sidesteps tube-release instability while preserving the visual motion
        of the arm approaching and "placing" the tube.

        Returns 0 on success, 1 if no tube is held / no empty slot, 2 if the
        arm couldn't reach the slot.
        """
        if rack_name not in ("left_tube_rack", "right_tube_rack"):
            print(f"[SimXArm] place_tube_in_rack: unknown rack '{rack_name}'")
            return 1

        # Which tube is currently welded?
        held = None
        with self.lock:
            for body_name in TUBE_BODY_NAMES:
                eqid = self.weld_eqids[body_name]
                if self.data.eq_active[eqid]:
                    held = body_name
                    break
            if held is None:
                print("[SimXArm] place_tube_in_rack: no tube is currently held")
                return 1

            rack_bid = self.model.body(rack_name).id
            rack_pos = self.data.xpos[rack_bid].copy()
            # Snapshot positions of all OTHER tubes (the held one is moving with
            # the gripper and shouldn't disqualify any slot)
            other_tube_pos = {
                n: self.data.xpos[self.model.body(n).id].copy()
                for n in TUBE_BODY_NAMES if n != held
            }

        # First empty slot wins
        chosen = None
        for col, row, dx, dy in RACK_SLOTS:
            sx, sy = rack_pos[0] + dx, rack_pos[1] + dy
            occupied = False
            for n, p in other_tube_pos.items():
                if (abs(p[0] - sx) < SLOT_OCCUPIED_TOL_M and
                        abs(p[1] - sy) < SLOT_OCCUPIED_TOL_M):
                    occupied = True
                    break
            if not occupied:
                chosen = (col, row, sx, sy)
                break

        if chosen is None:
            print(f"[SimXArm] place_tube_in_rack: no empty slot in {rack_name}")
            return 1

        col, row, sx, sy = chosen
        print(f"[SimXArm] place_tube_in_rack: placing {held} into "
              f"{rack_name} col{col} row{row} at "
              f"({sx*1000:.0f}, {sy*1000:.0f}) mm")

        # Fly the arm over the chosen slot at a safe height (clear of rack walls
        # which top out at z=0.810 in world)
        ret = self.set_position(
            x=sx * 1000.0, y=sy * 1000.0, z=920.0,
            roll=180.0, pitch=0.0, yaw=0.0, wait=True,
        )
        if ret != 0:
            print(f"[SimXArm] place_tube_in_rack: approach failed (ret={ret})")
            return 2

        # Snap the tube into the slot + deactivate weld
        with self.lock:
            bid = self.model.body(held).id
            jnt_adr = self.model.body_jntadr[bid]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]
            dof_adr = self.model.jnt_dofadr[jnt_adr]
            # Position the tube upright at slot center, body bottom on plate top
            self.data.qpos[qpos_adr + 0] = sx
            self.data.qpos[qpos_adr + 1] = sy
            self.data.qpos[qpos_adr + 2] = TUBE_SLOT_Z_M
            self.data.qpos[qpos_adr + 3] = 1.0  # quat w
            self.data.qpos[qpos_adr + 4] = 0.0  # quat x
            self.data.qpos[qpos_adr + 5] = 0.0  # quat y
            self.data.qpos[qpos_adr + 6] = 0.0  # quat z
            self.data.qvel[dof_adr:dof_adr + 6] = 0.0
            # Release the weld
            self.data.eq_active[self.weld_eqids[held]] = 0
            mujoco.mj_forward(self.model, self.data)

        print(f"[SimXArm] place_tube_in_rack: {held} placed in "
              f"{rack_name} col{col} row{row}")
        return 0

    def _weld_lock(self, body_name: str):
        """Force-activate the weld for body_name with the current relpose.
        Bypasses the magnetic-reach distance check that close_lite6_gripper
        uses -- intended for "flyover" push mode (bins/tubes/racks) where
        the arm hovers above the target and we want it to be carried along
        regardless of exact gripper-to-body distance."""
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            bid = self.cube_bids[body_name]
            eqid = self.weld_eqids[body_name]
            gp = self.data.xpos[self.gripper_bid].copy()
            gq = self.data.xquat[self.gripper_bid].copy()
            cp = self.data.xpos[bid].copy()
            cq = self.data.xquat[bid].copy()
            gq_inv = np.zeros(4); mujoco.mju_negQuat(gq_inv, gq)
            rel_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_pos, cp - gp, gq_inv)
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, gq_inv, cq)
            self.model.eq_data[eqid, 0:3] = 0.0
            self.model.eq_data[eqid, 3:6] = rel_pos
            self.model.eq_data[eqid, 6:10] = rel_quat
            self.model.eq_data[eqid, 10] = 1.0
            self.data.eq_active[eqid] = 1

    # ---- pushing ----
    BENCH_X_MM = (-750.0, 750.0)   # bench top extents in world mm
    BENCH_Y_MM = (-450.0, 450.0)
    PUSH_TRANSIT_Z_MM = 830.0      # low drag height. Higher than the grasp height
                                   # so cubes clear (most of) the bin walls when
                                   # dragged across the bench. Going lower would
                                   # look more like "true pushing" but knocks bins
                                   # over when the path crosses the bin row.
    PUSH_APPROACH_Z_MM = 870.0     # safer height to reach over the object first
    PUSH_GRASP_Z_MM = 795.0        # grasp height (same as cube pick-place)

    def push_object(self, target_name: str,
                    to_x_mm: float, to_y_mm: float,
                    speed_mm_s: float = 100.0, **kwargs) -> int:
        """Push (drag) a cube to a target xy position.

        Mechanic: magnetic-grasp + low-altitude drag + release. The "drag"
        height is just above the bench so the motion looks like sliding
        rather than picking-and-placing. After release, physics takes over:
        on-bench targets land on the bench, off-bench targets fall to the
        floor under gravity.

        target_name: a cube body name ('red_cube' / 'green_cube' / 'blue_cube').
        to_x_mm, to_y_mm: destination in world frame, in millimetres. Pass
            values beyond BENCH_X_MM / BENCH_Y_MM to push the cube off the bench.

        Returns 0 on success, 1 if target_name isn't a graspable cube, 2 if a
        move sub-step fails validation.
        """
        if target_name not in self.cube_bids:
            print(f"[SimXArm] push_object: unknown target '{target_name}'. "
                  f"Graspable bodies: {list(self.cube_bids.keys())}")
            return 1

        # Look up object's starting position
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            cube_pos = self.data.xpos[self.cube_bids[target_name]].copy()
        cx_mm, cy_mm = cube_pos[0] * 1000.0, cube_pos[1] * 1000.0

        on_bench = (self.BENCH_X_MM[0] <= to_x_mm <= self.BENCH_X_MM[1] and
                    self.BENCH_Y_MM[0] <= to_y_mm <= self.BENCH_Y_MM[1])
        print(f"[SimXArm] push_object: {target_name} "
              f"({cx_mm:+.0f},{cy_mm:+.0f}) -> ({to_x_mm:+.0f},{to_y_mm:+.0f}) "
              f"{'(on bench)' if on_bench else '(OFF BENCH -- will fall to floor)'}")

        # Two execution modes:
        #   "grasp": cubes only -- magnetic-grasp + drag at low height + release
        #     (the natural pushing visual)
        #   "flyover": bins and tubes -- arm visits source then destination,
        #     the object is teleported on the "release" step. We use flyover
        #     for bins/tubes because their geometry (tall walls / tall body
        #     with cap) makes a clean magnetic grasp at the heights needed
        #     unreliable under position-only IK.
        use_grasp = (target_name in ("red_cube", "green_cube", "blue_cube"))
        grasp_z = self.PUSH_GRASP_Z_MM
        transit_z = self.PUSH_TRANSIT_Z_MM

        # Position the rail roughly under the source cube. World arm-base
        # x = rail_carriage_world_x = -0.35 + rail_qpos. To align under the
        # cube, set rail = cube_world_x + 0.35.
        rail_target_mm = float(np.clip(cx_mm + 350.0, 0.0, 700.0))
        self.set_rail_position(position_mm=rail_target_mm,
                               speed_mm_s=speed_mm_s, wait=True)

        # Helper: try to move arm but don't abort the whole push if it fails.
        # The snap step at the end teleports the object to target regardless,
        # so motion sub-steps are best-effort -- they make the visual motion
        # look right, but missing one isn't fatal.
        def _try_move(x, y, z, label):
            r = self.set_position(x=x, y=y, z=z, roll=180, pitch=0, yaw=0,
                                  speed=speed_mm_s, wait=True)
            if r != 0:
                print(f"[SimXArm] push_object: {label} sub-step failed "
                      f"(ret={r}) at ({x:+.0f},{y:+.0f},{z:+.0f}). Continuing.")

        # 1) Hover above the object at safe approach height
        _try_move(cx_mm, cy_mm, self.PUSH_APPROACH_Z_MM, "approach")

        if use_grasp:
            # 2) Lower to grasp height (cubes)
            _try_move(cx_mm, cy_mm, grasp_z, "lower-to-grasp")
            # 3) Magnetic grasp
            self.close_lite6_gripper()
            # 4) Lift to drag height
            _try_move(cx_mm, cy_mm, transit_z, "lift-to-drag")
        # Track every body we've welded so they can all be released later.
        bodies_welded = []
        if not use_grasp:
            # Bins/tubes/racks: arm is hovering above the object at
            # PUSH_APPROACH_Z_MM. Force-activate the weld with the current
            # relpose so the object follows the arm during the drag.
            self._weld_lock(target_name)
            bodies_welded.append(target_name)
            # For racks, ALSO weld the tubes sitting in them so the whole
            # assembly moves together. Without this, dragging the rack
            # leaves the tubes behind (and the rack walls slide past them,
            # generating contact forces that launch tubes off the bench).
            if target_name in RACK_TUBE_GROUPS:
                for tube_name in RACK_TUBE_GROUPS[target_name]:
                    self._weld_lock(tube_name)
                    bodies_welded.append(tube_name)
            print(f"[SimXArm] push_object: weld locked on {bodies_welded} -- "
                  f"will be carried by the arm")

        # 5) For off-bench targets, the arm may need to extend past the rail
        # workspace. Reposition the rail if needed before the final move_to.
        rail_target_mm = float(np.clip(to_x_mm + 350.0, 0.0, 700.0))
        self.set_rail_position(position_mm=rail_target_mm,
                               speed_mm_s=speed_mm_s, wait=True)

        # 6) Drag (or fly) toward the destination. Cubes drag at low height
        # (true pushing); bins/tubes fly at approach height (no grasp held).
        # IK may not reach exactly for extreme off-bench targets -- we don't
        # treat that as fatal because step 7 teleports for guaranteed placement.
        drag_z = transit_z if use_grasp else self.PUSH_APPROACH_Z_MM
        drag_ret = self.set_position(x=to_x_mm, y=to_y_mm, z=drag_z,
                                     roll=180, pitch=0, yaw=0,
                                     speed=speed_mm_s, wait=True)
        if drag_ret != 0:
            print(f"[SimXArm] push_object: arm couldn't fully reach drag "
                  f"target ({to_x_mm:+.0f},{to_y_mm:+.0f}). Snapping cube "
                  f"to target on release anyway.")

        # 7) Snap the target object to the destination + release all welds.
        # If a rack was pushed, we also snap the rack's tubes to their slots
        # relative to the rack's new position, so they end up "in" the rack
        # instead of left behind at the original rack location.
        tube_slot_offsets = {}  # tube_name -> (dx, dy) in rack frame
        if target_name in RACK_TUBE_GROUPS:
            # Initial (XML) layout for each tube relative to its rack
            INITIAL_TUBE_OFFSETS = {
                "tube_L1": (-0.060, -0.020), "tube_L2": (-0.020, +0.020),
                "tube_L3": (+0.060, -0.020),
                "tube_R1": (-0.060, +0.020), "tube_R2": (+0.020, -0.020),
                "tube_R3": (+0.060, +0.020),
            }
            for tn in RACK_TUBE_GROUPS[target_name]:
                tube_slot_offsets[tn] = INITIAL_TUBE_OFFSETS[tn]
        # For ON-bench: snap to bench-top z (0.765 m), zero velocity -- object
        #   settles cleanly at the destination.
        # For OFF-bench: snap to z=0.80 (well above the floor) so it falls
        #   ~800 mm under gravity, with a *real* visible drop instead of
        #   appearing on the floor. Do NOT zero velocity so any drag
        #   momentum continues to carry it past the edge.
        with self.lock:
            bid = self.cube_bids[target_name]
            jnt_adr  = self.model.body_jntadr[bid]
            qpos_adr = self.model.jnt_qposadr[jnt_adr]
            dof_adr  = self.model.jnt_dofadr[jnt_adr]
            if on_bench:
                # On-bench snap z varies by object type so the body's
                # bottom-most geom rests on the bench top (z=0.750):
                if target_name.endswith("_bin"):
                    snap_z = 0.749            # bin floor at body z=0.001
                elif target_name.endswith("_tube_rack"):
                    snap_z = 0.755            # rack base bottom at body z=-0.005
                elif target_name.startswith("tube_"):
                    snap_z = 0.8175           # tube body center
                else:
                    snap_z = 0.765            # cube center
                self.data.qvel[dof_adr:dof_adr + 6] = 0.0
            else:
                # Off-bench: dropped from mid-air past the edge, falls under gravity
                snap_z = 0.80  # mid-air over the floor -- let gravity drop it
                # Keep some forward velocity so the cube continues moving
                # in the push direction as it falls (visual realism).
                # qvel: 6 entries = (vx, vy, vz, wx, wy, wz)
                self.data.qvel[dof_adr + 3:dof_adr + 6] = 0.0  # zero angular
            self.data.qpos[qpos_adr + 0] = to_x_mm / 1000.0
            self.data.qpos[qpos_adr + 1] = to_y_mm / 1000.0
            self.data.qpos[qpos_adr + 2] = snap_z
            self.data.qpos[qpos_adr + 3] = 1.0; self.data.qpos[qpos_adr + 4] = 0.0
            self.data.qpos[qpos_adr + 5] = 0.0; self.data.qpos[qpos_adr + 6] = 0.0
            # Release the primary target's weld
            self.data.eq_active[self.weld_eqids[target_name]] = 0
            # Snap tubes to their slot positions relative to the rack
            # (only relevant when target is a rack carrying tubes)
            for tn, (dx, dy) in tube_slot_offsets.items():
                tube_bid = self.cube_bids[tn]
                tube_jnt = self.model.body_jntadr[tube_bid]
                tube_qpos = self.model.jnt_qposadr[tube_jnt]
                tube_dof  = self.model.jnt_dofadr[tube_jnt]
                tube_x = to_x_mm / 1000.0 + dx
                tube_y = to_y_mm / 1000.0 + dy
                # Tube body center sits 62.5mm above the rack's body origin
                # (rack base plate top at +0.005, tube half-height 0.0575)
                tube_z = snap_z + 0.0625
                self.data.qpos[tube_qpos:tube_qpos + 3] = [tube_x, tube_y, tube_z]
                self.data.qpos[tube_qpos + 3:tube_qpos + 7] = [1, 0, 0, 0]
                self.data.qvel[tube_dof:tube_dof + 6] = 0.0
                self.data.eq_active[self.weld_eqids[tn]] = 0
            mujoco.mj_forward(self.model, self.data)
        print(f"[SimXArm] push_object: {target_name} placed at "
              f"({to_x_mm:+.0f},{to_y_mm:+.0f}) "
              f"{'on bench' if on_bench else 'mid-air past bench edge — falling to floor'}")

        # 8) Lift the arm to a safe intermediate height before returning.
        # This leaves the IK in a clean overhead starting state so that a
        # following push_object (or any subsequent move_to) doesn't start
        # from far-out workspace coordinates where the Jacobian solver can
        # converge to a colliding local minimum.
        self.set_position(x=to_x_mm, y=to_y_mm, z=self.PUSH_APPROACH_Z_MM,
                          roll=180, pitch=0, yaw=0,
                          speed=speed_mm_s, wait=True)
        return 0

    def reset_scene(self) -> int:
        """Reset cubes to their initial poses and drive arm to home.

        Used between episodes in multi-episode scripts (auto_play, random_play)
        so each episode starts from a clean baseline without spawning a fresh
        SimXArmAPI / viewer window.
        """
        with self.lock:
            # Release any grasped cube
            for eqid in self.weld_eqids.values():
                self.data.eq_active[eqid] = 0
            # Restore each cube's free-joint qpos/qvel to XML defaults
            for cube_name, bid in self.cube_bids.items():
                jnt_adr = self.model.body_jntadr[bid]
                qpos_adr = self.model.jnt_qposadr[jnt_adr]
                dof_adr  = self.model.jnt_dofadr[jnt_adr]
                self.data.qpos[qpos_adr:qpos_adr + 7] = \
                    self.model.qpos0[qpos_adr:qpos_adr + 7]
                self.data.qvel[dof_adr:dof_adr + 6] = 0.0
            # Snap the fingers back open (qpos + ctrl) so each episode starts
            # with a clean, visibly-open gripper.
            for jid in self.finger_jids:
                qpos_adr = self.model.jnt_qposadr[jid]
                dof_adr  = self.model.jnt_dofadr[jid]
                self.data.qpos[qpos_adr] = 0.0
                self.data.qvel[dof_adr]  = 0.0
            self._set_finger_ctrl(GRIPPER_OPEN_CTRL_M)
            mujoco.mj_forward(self.model, self.data)
        self.go_home()
        # Re-snapshot baseline positions so per-episode displacement /
        # proximity facts in physical_outcome() are measured from this
        # clean state, not from whatever the prior episode left behind.
        self._snapshot_positions()
        return 0

    # ---- gestures ----
    def wave_goodbye(self, n_waves: int = 3, **kwargs) -> int:
        """Move to mid-rail, then sweep the whole arm left/right N times.

        Visualization-friendly motion: shoulder is bent forward 30 deg so
        the gripper hangs out-and-up, then joint1 (base rotation) sweeps
        +/-25 deg, making the gripper trace a clearly visible side-to-side
        arc. Returns to home pose afterward.
        """
        n_waves = max(1, int(n_waves))

        # 1) Mid-rail
        self.set_rail_position(position_mm=350, wait=True)

        # 2) Wave-base pose: shoulder bent 30 deg forward so motion is visible
        wave_base = np.deg2rad([0.0, 30.0, 0.0, 0.0, 0.0, 0.0])
        self._execute_joint_angles(wave_base)
        self._wait_arm_settled(wave_base, tol_rad=0.05, timeout=3.0)

        # 3) Sweep joint1 +/-25 deg, n_waves cycles
        swing = np.deg2rad(25.0)
        for _ in range(n_waves):
            right = wave_base.copy(); right[0] = +swing
            self._execute_joint_angles(right)
            self._wait_arm_settled(right, tol_rad=0.10, timeout=1.5)
            left = wave_base.copy(); left[0] = -swing
            self._execute_joint_angles(left)
            self._wait_arm_settled(left, tol_rad=0.10, timeout=1.5)

        # 4) Center the wave-base then go home
        self._execute_joint_angles(wave_base)
        self._wait_arm_settled(wave_base, tol_rad=0.05, timeout=2.0)
        self.go_home()
        print(f"[SimXArm] wave_goodbye ({n_waves} waves) complete")
        return 0

    # ---- rail control ----
    def set_rail_position(self, position_mm: float,
                          speed_mm_s: float = 50.0,
                          wait: bool = True, **kwargs) -> int:
        pos_m = float(np.clip(position_mm / 1000.0, 0.0, 0.7))
        # Determine pacing duration from the rail-distance / speed. Skip
        # pacing for tiny moves or non-positive speed (preserves the
        # historical instant-ctrl behaviour for legacy callers).
        with self.lock:
            start_m = float(self.data.ctrl[self.act_ids[RAIL_ACT]])
        dist_mm = abs(pos_m - start_m) * 1000.0
        if speed_mm_s and speed_mm_s > 0 and dist_mm > 1.0:
            duration_s = dist_mm / float(speed_mm_s)
            self._execute_paced_rail(pos_m, duration_s)
        else:
            with self.lock:
                self.data.ctrl[self.act_ids[RAIL_ACT]] = pos_m
        if wait:
            self._wait_rail_settled(pos_m)
        return 0

    def get_rail_position(self) -> tuple:
        with self.lock:
            pos_m = self.data.qpos[self.rail_jid]
        return 0, pos_m * 1000.0

    def _wait_rail_settled(self, target_m: float,
                           tol: float = 0.002, timeout: float = 5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                current = self.data.qpos[self.rail_jid]
            if abs(current - target_m) < tol:
                return
            time.sleep(0.05)

    # ---- arm control ----
    def set_position(self, x: float, y: float, z: float,
                     roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0,
                     speed: float = 100.0, wait: bool = True, **kwargs) -> int:
        target_pos = np.array([x, y, z]) / 1000.0

        # First attempt: IK from current qpos
        with self.lock:
            joint_angles = self.ik_solver.solve(target_pos)
        result = (self.validator.validate(joint_angles, target_pos)
                  if joint_angles is not None else None)

        # Retry from an all-zeros seed if (a) IK didn't converge or (b) the
        # solution it found is in collision. The iterative Jacobian solver
        # can fail or land in a colliding local minimum when the starting
        # qpos is bent (e.g. after go_home() to the new "ready" pose).
        # Seeding from all-zeros gives the search a clean over-the-top route.
        need_retry = (
            joint_angles is None
            or (result is not None and not result.is_valid)
        )
        if need_retry:
            # Two fallback seeds, tried in order. The all-zeros "rocket"
            # pose is too far from typical pick-and-place targets for the
            # iterative Jacobian to converge in 100 iters; the second seed
            # is a slightly-bent forward "ready" pose that starts the
            # search much closer to any over-bench target.
            fallback_seeds = [
                np.zeros(len(self.joint_ids)),
                np.deg2rad([0.0, 20.0, 0.0, 0.0, 50.0, 0.0]),
            ]
            for seed in fallback_seeds:
                with self.lock:
                    retry_angles = self.ik_solver.solve(target_pos, seed_q=seed)
                if retry_angles is None:
                    continue
                retry_result = self.validator.validate(retry_angles, target_pos)
                if retry_result.is_valid:
                    joint_angles = retry_angles
                    result = retry_result
                    break

        if joint_angles is None:
            print(f"[SimXArm] IK failed for target "
                  f"({x:.1f}, {y:.1f}, {z:.1f}) mm")
            return 1

        if not result.is_valid:
            print(f"[SimXArm] Validation failed: {result.reason}")
            return 2

        # Pace the motion based on Cartesian distance / requested speed
        # so the realised arm motion roughly tracks the user-specified
        # mm/s. The dispatch layer in LLMBrain already clamps `speed` to
        # the active session/per-command cap, so here we just translate
        # mm/s into wall-clock duration and let _execute_paced_arm do
        # the interpolation.
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            ee_now_m = self.data.site_xpos[self.ee_site].copy()
        dist_mm = float(np.linalg.norm(target_pos - ee_now_m) * 1000.0)
        if speed and speed > 0 and dist_mm > 1.0:
            duration_s = dist_mm / float(speed)
            self._execute_paced_arm(joint_angles, duration_s)
        else:
            self._execute_joint_angles(joint_angles)
        if wait:
            self._wait_arm_settled(joint_angles)
        return 0

    def set_servo_angle(self, angle, speed: float = 30.0,
                        wait: bool = True, **kwargs) -> int:
        if len(angle) != 6:
            print(f"[SimXArm] Expected 6 joint angles, got {len(angle)}")
            return 1
        angles_rad = np.deg2rad(angle)
        # Pace by max joint angular delta / speed (deg/s). When multiple
        # joints move, the largest delta dictates the duration so every
        # joint completes its move within the requested window.
        with self.lock:
            start_rad = np.array([float(self.data.ctrl[self.act_ids[1 + i]])
                                  for i in range(6)])
        max_delta_deg = float(np.max(np.abs(np.rad2deg(angles_rad - start_rad))))
        if speed and speed > 0 and max_delta_deg > 0.1:
            duration_s = max_delta_deg / float(speed)
            self._execute_paced_arm(angles_rad, duration_s)
        else:
            self._execute_joint_angles(angles_rad)
        if wait:
            self._wait_arm_settled(angles_rad)
        return 0

    def get_position(self) -> tuple:
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            pos = self.data.site_xpos[self.ee_site].copy()
            mat = self.data.site_xmat[self.ee_site].reshape(3, 3).copy()
        result = list(pos * 1000.0)
        if HAS_TRANSFORMS3D:
            result += list(np.rad2deg(mat2euler(mat, axes='sxyz')))
        else:
            result += [0.0, 0.0, 0.0]
        return 0, result

    def get_servo_angle(self) -> tuple:
        with self.lock:
            angles = [np.rad2deg(self.data.qpos[jid]) for jid in self.joint_ids]
        return 0, angles

    def _set_finger_ctrl(self, ctrl_value: float) -> None:
        """Drive both finger actuators to the same ctrl. Caller must hold self.lock.
        No-op if the scene has no finger actuators (older / different XML)."""
        if not self.finger_act_ids:
            return
        v = float(np.clip(ctrl_value, GRIPPER_OPEN_CTRL_M, GRIPPER_CLOSED_CTRL_M))
        for aid in self.finger_act_ids:
            self.data.ctrl[aid] = v

    def open_lite6_gripper(self) -> int:
        """Release any held cube by deactivating all gripper weld constraints."""
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            released = []
            for cube_name, eqid in self.weld_eqids.items():
                if self.data.eq_active[eqid]:
                    self.data.eq_active[eqid] = 0
                    released.append(cube_name)
            self._set_finger_ctrl(GRIPPER_OPEN_CTRL_M)
        print(f"[SimXArm] Gripper open  (released: {released or 'nothing'})")
        return 0

    def close_lite6_gripper(self) -> int:
        """Grasp the nearest cube within GRIPPER_REACH_M of the EE site.

        Captures the *current* gripper<->cube relative pose into eq_data
        before activating the weld, so the constraint locks the grasp in
        place without snapping the cube to coincide with the gripper body.
        """
        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            ee_pos = self.data.site_xpos[self.ee_site].copy()
            # For each candidate body, distance = min distance from EE to any
            # of its geoms (body or cap for tubes; single geom for cubes).
            # Using body center alone overshoots for tall objects when the
            # graspable feature (e.g. tube cap) is far from the body origin.
            nearest, nearest_d = None, float("inf")
            for cube_name, bid in self.cube_bids.items():
                geom_ids = [g for g in range(self.model.ngeom)
                            if self.model.geom_bodyid[g] == bid]
                if not geom_ids:
                    continue
                d = min(float(np.linalg.norm(self.data.geom_xpos[g] - ee_pos))
                        for g in geom_ids)
                if d < nearest_d:
                    nearest, nearest_d = cube_name, d
            # Animate fingers closing regardless of whether anything is in
            # reach -- a "tried to grasp but missed" visual is informative.
            self._set_finger_ctrl(GRIPPER_CLOSED_CTRL_M)
            if nearest is None or nearest_d > GRIPPER_REACH_M:
                print(f"[SimXArm] Gripper close (no cube in reach; "
                      f"nearest={nearest} at {nearest_d*1000:.1f}mm)")
                return 0

            # Compute cube pose in gripper's local frame and stamp it into
            # eq_data so the weld locks the *current* relative pose.
            cube_bid = self.cube_bids[nearest]
            gp = self.data.xpos[self.gripper_bid].copy()
            gq = self.data.xquat[self.gripper_bid].copy()
            cp = self.data.xpos[cube_bid].copy()
            cq = self.data.xquat[cube_bid].copy()
            gq_inv = np.zeros(4); mujoco.mju_negQuat(gq_inv, gq)
            rel_pos = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_pos, cp - gp, gq_inv)
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, gq_inv, cq)
            eqid = self.weld_eqids[nearest]
            # eq_data layout for weld in MuJoCo 3.8.1:
            #   [0:3]   anchor (body1 frame)
            #   [3:6]   relpose translation (body2 in body1 frame)
            #   [6:10]  relpose quaternion (body2 in body1 frame, wxyz)
            #   [10]    torquescale
            self.model.eq_data[eqid, 0:3]  = 0.0          # anchor at body1 origin
            self.model.eq_data[eqid, 3:6]  = rel_pos
            self.model.eq_data[eqid, 6:10] = rel_quat
            self.model.eq_data[eqid, 10]   = 1.0
            self.data.eq_active[eqid] = 1
        print(f"[SimXArm] Gripper close (grasped {nearest} at "
              f"{nearest_d*1000:.1f}mm)")
        return 0

    def _execute_joint_angles(self, angles_rad: np.ndarray):
        with self.lock:
            for i, angle in enumerate(angles_rad):
                self.data.ctrl[self.act_ids[1 + i]] = float(angle)

    # ---- speed-paced motion helpers --------------------------------------
    # These exist because MuJoCo position actuators have no built-in
    # velocity limit -- a one-shot `data.ctrl[...] = target` will drive the
    # joint to the new setpoint as fast as the PD gains and dynamics allow.
    # To honour a user-specified mm/s or deg/s on the motion primitives, we
    # interpolate the ctrl value from its current position to the target
    # over wall-clock time, writing intermediate setpoints at PACING_HZ.
    # The actual joint positions track the setpoints with some PD lag, so
    # the realised motion is roughly the requested speed (good enough as a
    # safety cap for real-arm transfer; not a precise velocity controller).

    PACING_HZ = 50.0
    MAX_MOTION_DURATION_S = 30.0  # safety upper bound; longer requested
                                  # durations are clamped so a bogus
                                  # speed=0.001 doesn't hang the session

    def _execute_paced_arm(self, target_angles_rad: np.ndarray,
                           duration_s: float) -> None:
        """Drive the 6 arm joint actuators from their current ctrl values
        to `target_angles_rad` over wall-clock `duration_s`. Skips pacing
        (just writes the target) for non-positive or zero durations."""
        duration_s = float(min(max(duration_s, 0.0), self.MAX_MOTION_DURATION_S))
        if duration_s <= 0.0:
            self._execute_joint_angles(target_angles_rad)
            return
        n_steps = max(int(round(duration_s * self.PACING_HZ)), 2)
        dt = duration_s / n_steps
        with self.lock:
            start = np.array([float(self.data.ctrl[self.act_ids[1 + i]])
                              for i in range(len(target_angles_rad))])
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            intermediate = start + alpha * (target_angles_rad - start)
            with self.lock:
                for i, ang in enumerate(intermediate):
                    self.data.ctrl[self.act_ids[1 + i]] = float(ang)
            if k < n_steps:
                time.sleep(dt)

    def _execute_paced_rail(self, target_pos_m: float,
                            duration_s: float) -> None:
        """Same idea for the single linear-rail actuator."""
        duration_s = float(min(max(duration_s, 0.0), self.MAX_MOTION_DURATION_S))
        if duration_s <= 0.0:
            with self.lock:
                self.data.ctrl[self.act_ids[RAIL_ACT]] = target_pos_m
            return
        n_steps = max(int(round(duration_s * self.PACING_HZ)), 2)
        dt = duration_s / n_steps
        with self.lock:
            start_m = float(self.data.ctrl[self.act_ids[RAIL_ACT]])
        for k in range(1, n_steps + 1):
            alpha = k / n_steps
            intermediate = start_m + alpha * (target_pos_m - start_m)
            with self.lock:
                self.data.ctrl[self.act_ids[RAIL_ACT]] = float(intermediate)
            if k < n_steps:
                time.sleep(dt)

    def physical_outcome(self, stringency: str = DEFAULT_STRINGENCY) -> str:
        """Inspect cube/tube positions and return a short human-readable summary.

        Categories detected:
          - "fell to floor" : object z is near floor level (< 0.1 m)
          - "in <bin>"      : cube center inside a bin footprint, low z above bin floor
          - "in <rack>"     : tube body seated in a rack slot of a NON-HOME rack
                              (tubes that haven't left their home rack are silent
                              -- the loop only cares about state changes)
          - "off bench"     : object xy is outside bench bounds but still elevated
                              (mid-air or in-flight)
        Anything else (resting loose on the bench or in its home rack) isn't called out.

        `stringency` selects how tight the rack-seated / bin-placed thresholds
        are; see STRINGENCY_LEVELS at module top. "loose" preserves the
        original (pre-stringency) behaviour and is the default.
        """
        cfg = STRINGENCY_LEVELS.get(stringency, STRINGENCY_LEVELS[DEFAULT_STRINGENCY])
        rack_xy_tol  = cfg["rack_xy_tol_m"]
        rack_z_tol   = cfg["rack_z_tol_m"]
        bin_xy_tol   = cfg["bin_xy_tol_m"]
        tilt_deg_max = cfg["tilt_deg_max"]

        with self.lock:
            mujoco.mj_forward(self.model, self.data)
            object_positions = {
                name: self.data.xpos[bid].copy()
                for name, bid in self.cube_bids.items()
            }
            # Body orientations as 3x3 rotation matrices -- column 2 is the
            # body's local +z axis expressed in the world frame. Used to gate
            # rack-seated reporting by uprightness.
            object_xmats = {
                name: self.data.xmat[bid].reshape(3, 3).copy()
                for name, bid in self.cube_bids.items()
            }
            bin_positions = {
                name: self.data.xpos[self.model.body(name).id].copy()
                for name in ("red_bin", "green_bin", "blue_bin")
            }
            rack_positions = {
                name: self.data.xpos[self.model.body(name).id].copy()
                for name in RACK_TUBE_GROUPS
            }

        bench_x_min, bench_x_max = self.BENCH_X_MM[0] / 1000.0, self.BENCH_X_MM[1] / 1000.0
        bench_y_min, bench_y_max = self.BENCH_Y_MM[0] / 1000.0, self.BENCH_Y_MM[1] / 1000.0

        # Reverse the RACK_TUBE_GROUPS mapping for quick home-rack lookup.
        tube_home_rack = {
            tube: rack for rack, tubes in RACK_TUBE_GROUPS.items() for tube in tubes
        }

        def _tube_seated_in_rack(tube_pos, rack_pos) -> bool:
            """True if a tube body is sitting in any slot of the given rack."""
            if abs(tube_pos[2] - TUBE_SLOT_Z_M) > rack_z_tol:
                return False
            for (_col, _row, dx, dy) in RACK_SLOTS:
                sx = rack_pos[0] + dx
                sy = rack_pos[1] + dy
                if (abs(tube_pos[0] - sx) < rack_xy_tol
                        and abs(tube_pos[1] - sy) < rack_xy_tol):
                    return True
            return False

        def _is_upright(xmat) -> bool:
            """True if the body's local +z axis tilts <= tilt_deg_max from world +z."""
            if tilt_deg_max >= 90.0:
                return True  # loose mode: no uprightness check
            local_z_world = xmat[:, 2]
            cos_tilt = float(np.clip(local_z_world[2], -1.0, 1.0))
            return np.rad2deg(np.arccos(cos_tilt)) <= tilt_deg_max

        notes = []
        # `categorical` tracks objects that triggered one of the four primary
        # event shapes (fell / in bin / in rack / off bench). Those are
        # exclusive -- we don't also emit a "displaced" fact for them. Bins
        # and racks have their own categorical event (off bench); if they
        # haven't moved off the bench, they're candidates for displacement
        # / proximity facts based on `_initial_positions`.
        categorical: set = set()
        for obj_name, p in object_positions.items():
            # Fell to floor?
            if p[2] < 0.10:
                notes.append(f"{obj_name} fell to floor")
                categorical.add(obj_name)
                continue
            # In a bin?
            placed_in = None
            for bin_name, bpos in bin_positions.items():
                dx = abs(p[0] - bpos[0])
                dy = abs(p[1] - bpos[1])
                dz = p[2] - bpos[2]
                if dx < bin_xy_tol and dy < bin_xy_tol and 0.0 < dz < 0.060:
                    placed_in = bin_name
                    break
            if placed_in is not None:
                notes.append(f"{obj_name} in {placed_in}")
                categorical.add(obj_name)
                continue
            # Tube seated in a NON-HOME rack (= a placement event worth reporting).
            if obj_name in tube_home_rack:
                home = tube_home_rack[obj_name]
                seated_in_other = None
                for rack_name, rack_pos in rack_positions.items():
                    if rack_name == home:
                        continue
                    if (_tube_seated_in_rack(p, rack_pos)
                            and _is_upright(object_xmats[obj_name])):
                        seated_in_other = rack_name
                        break
                if seated_in_other is not None:
                    notes.append(f"{obj_name} in {seated_in_other}")
                    categorical.add(obj_name)
                    continue
                # Not in any non-home rack -- fall through to off-bench check.
            # Past the bench edge in xy (but still elevated -- mid-fall or hanging)
            if not (bench_x_min <= p[0] <= bench_x_max and
                    bench_y_min <= p[1] <= bench_y_max):
                notes.append(f"{obj_name} off bench")
                categorical.add(obj_name)

        # --- Displacement + proximity facts ---
        # On top of the categorical events, also emit relative-position
        # facts for movable bodies (cubes + tubes + bins + racks) so the
        # grader can check tasks like "push X closer to Y" or "move X
        # by 100mm". Compared to _initial_positions captured at last
        # reset_scene().
        #
        # Bins / racks are tracked too (they're free bodies in this scene).
        # Tubes that haven't left their home rack stay silent in the
        # categorical pass; the displacement pass reports them too.
        DISPLACE_TOL_M = 0.020      # <20 mm is treated as noise
        PROX_DELTA_TOL_M = 0.020    # inter-object distance change to call out

        all_positions: dict = dict(object_positions)
        for bin_name, bpos in bin_positions.items():
            all_positions[bin_name] = bpos
        for rack_name, rpos in rack_positions.items():
            all_positions[rack_name] = rpos

        moved: dict = {}  # name -> (Δx, Δy) in mm
        for name, p in all_positions.items():
            init = self._initial_positions.get(name)
            if init is None:
                continue
            dx_m = float(p[0] - init[0])
            dy_m = float(p[1] - init[1])
            if (dx_m * dx_m + dy_m * dy_m) ** 0.5 < DISPLACE_TOL_M:
                continue
            moved[name] = (dx_m * 1000.0, dy_m * 1000.0)
            if name not in categorical:
                notes.append(
                    f"{name} moved ({moved[name][0]:.0f}, "
                    f"{moved[name][1]:.0f})mm"
                )

        # Pairwise proximity facts: for each pair where at least one
        # member moved measurably, report whether they're closer or
        # farther than before. Only emitted when the inter-object
        # distance changed by more than PROX_DELTA_TOL_M. Pair members
        # are sorted alphabetically so the fact `<a> closer to <b>`
        # comes out in a stable, predictable order (the dynamic grader's
        # prompt tells Haiku to expect alphabetical-first first).
        names_for_pairs = sorted(all_positions.keys())
        for i, a in enumerate(names_for_pairs):
            for b in names_for_pairs[i + 1:]:
                if a not in moved and b not in moved:
                    continue
                init_a = self._initial_positions.get(a)
                init_b = self._initial_positions.get(b)
                if init_a is None or init_b is None:
                    continue
                cur_a, cur_b = all_positions[a], all_positions[b]
                d_init = float(((init_a[0] - init_b[0]) ** 2 +
                                (init_a[1] - init_b[1]) ** 2) ** 0.5)
                d_now  = float(((cur_a[0] - cur_b[0]) ** 2 +
                                (cur_a[1] - cur_b[1]) ** 2) ** 0.5)
                delta = d_now - d_init
                if abs(delta) < PROX_DELTA_TOL_M:
                    continue
                if delta < 0:
                    notes.append(f"{a} closer to {b}")
                else:
                    notes.append(f"{a} farther from {b}")

        return "; ".join(notes) if notes else "no objects displaced"

    def _wait_arm_settled(self, target_rad: np.ndarray,
                          tol_rad: float = 0.02, timeout: float = 5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                current = np.array(
                    [self.data.qpos[jid] for jid in self.joint_ids]
                )
            if np.max(np.abs(current - target_rad)) < tol_rad:
                return
            time.sleep(0.05)
