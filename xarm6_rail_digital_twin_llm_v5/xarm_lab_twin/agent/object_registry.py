# agent/object_registry.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GraspConfig:
    approach_direction: list
    grip_orientation_rpy: list
    grip_depth: float
    approach_standoff_mm: float


@dataclass
class LabObject:
    name: str
    aliases: list
    position_xyz_m: list
    grasp: GraspConfig
    safety_notes: str
    optimal_rail_mm: float = 350.0
    is_container: bool = False
    object_type: str = "cube"   # "cube" | "tube" | "bin" | "rack" | "plate" | "instrument"
    cap_color: str = ""         # for tubes: "orange" or "blue"
    last_updated: str = ""


class ObjectRegistry:

    def __init__(self, registry_path: str = "agent/objects.json"):
        self.path = Path(registry_path)
        self.objects: dict = {}
        if self.path.exists():
            self.load()

    def register(self, obj: LabObject):
        self.objects[obj.name] = obj
        self.save()

    def find(self, query: str) -> Optional[LabObject]:
        q = query.lower().strip()
        for obj in self.objects.values():
            if q == obj.name.lower():
                return obj
            if any(q in alias.lower() for alias in obj.aliases):
                return obj
        return None

    def refresh_from_sim(self, arm) -> None:
        """Overwrite each object's position + live yaw from the running sim.

        Called before each LLM call so operator mouse-perturbations made
        in the viewer between episodes appear in the rendered context.
        Objects whose `name` does not exist as a MuJoCo body are skipped
        silently (registry can carry virtual or legacy entries).
        """
        self._live_yaw_deg = {}
        for name, obj in self.objects.items():
            rc, pose = arm.get_body_pose(name)
            if rc != 0 or pose is None:
                continue
            x_mm, y_mm, z_mm, _r, _p, yaw_deg = pose
            obj.position_xyz_m = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
            self._live_yaw_deg[name] = yaw_deg

    def to_llm_context(self) -> str:
        cubes = [o for o in self.objects.values() if o.object_type == "cube"]
        tubes = [o for o in self.objects.values() if o.object_type == "tube"]
        bins  = [o for o in self.objects.values() if o.object_type == "bin"]
        racks = [o for o in self.objects.values() if o.object_type == "rack"]
        plates = [o for o in self.objects.values() if o.object_type == "plate"]
        instruments = [o for o in self.objects.values()
                       if o.object_type == "instrument"]
        lines = []

        live_yaw = getattr(self, "_live_yaw_deg", {})

        def fmt_basic(obj):
            x, y, z = obj.position_xyz_m
            yaw_line = ""
            yaw = live_yaw.get(obj.name)
            # Surface yaw whenever it has actually moved (>1 deg from 0).
            # Idle scenes stay quiet so the prompt doesn't bloat.
            if yaw is not None and abs(yaw) > 1.0:
                yaw_line = f"  Current yaw: {yaw:+.1f} deg (operator-adjusted)\n"
            return (
                f"- **{obj.name}**  aliases: {', '.join(obj.aliases)}\n"
                f"  Position: x={x*1000:.0f}mm  y={y*1000:.0f}mm  z={z*1000:.0f}mm\n"
                f"  Optimal rail: {obj.optimal_rail_mm:.0f}mm\n"
                f"{yaw_line}"
            )

        if cubes:
            lines.append("## Cubes (graspable, 30mm)\n")
            for obj in cubes:
                lines.append(fmt_basic(obj))
        if tubes:
            lines.append("\n## Falcon tubes (graspable, 30mm dia x 115mm tall)\n")
            for obj in tubes:
                extra = f"  Cap color: {obj.cap_color}\n" if obj.cap_color else ""
                lines.append(fmt_basic(obj) + extra)
        if bins:
            lines.append("\n## Bins (place destinations)\n")
            for obj in bins:
                lines.append(fmt_basic(obj))
        if racks:
            lines.append("\n## Tube racks (static fixtures)\n")
            for obj in racks:
                lines.append(fmt_basic(obj))
        if plates:
            lines.append("\n## 96-well plates (graspable, "
                         "127 x 85 x 14 mm)\n")
            for obj in plates:
                lines.append(fmt_basic(obj))
        if instruments:
            lines.append("\n## Instruments (static fixtures, NOT graspable)\n")
            for obj in instruments:
                lines.append(fmt_basic(obj) + f"  {obj.safety_notes}\n")
        return "\n".join(lines)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {k: asdict(v) for k, v in self.objects.items()}, indent=2))

    def load(self):
        data = json.loads(self.path.read_text())
        for k, v in data.items():
            v["grasp"] = GraspConfig(**v["grasp"])
            self.objects[k] = LabObject(**v)


def build_default_registry() -> ObjectRegistry:
    """
    Default registry matching the cubes-and-bins scene.
    Replace with lab-specific objects when ready.
    """
    reg = ObjectRegistry()

    cube_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.7,
        approach_standoff_mm=40.0,
    )
    bin_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.0,
        approach_standoff_mm=60.0,
    )

    # red_cube was removed; its position is now occupied by the
    # Vortex-Genie 2 (registered as an instrument below).
    reg.register(LabObject(
        name="green_cube",
        aliases=["green cube", "green block", "green"],
        position_xyz_m=[0.00, 0.15, 0.78],
        optimal_rail_mm=350.0,
        grasp=cube_grasp,
        safety_notes="Small graspable cube. Approach from above.",
        object_type="cube",
    ))
    reg.register(LabObject(
        name="blue_cube",
        aliases=["blue cube", "blue block", "blue"],
        position_xyz_m=[0.20, 0.15, 0.78],
        optimal_rail_mm=550.0,
        grasp=cube_grasp,
        safety_notes="Small graspable cube. Approach from above.",
        object_type="cube",
    ))

    # red_bin was removed; see note above about red_cube.
    reg.register(LabObject(
        name="green_bin",
        aliases=["green bin", "green container", "green box"],
        position_xyz_m=[0.00, 0.35, 0.75],
        optimal_rail_mm=350.0,
        grasp=bin_grasp,
        safety_notes="Open-top bin. Release cube above bin opening.",
        is_container=True,
        object_type="bin",
    ))
    reg.register(LabObject(
        name="blue_bin",
        aliases=["blue bin", "blue container", "blue box"],
        position_xyz_m=[0.20, 0.35, 0.75],
        optimal_rail_mm=550.0,
        grasp=bin_grasp,
        safety_notes="Open-top bin. Release cube above bin opening.",
        is_container=True,
        object_type="bin",
    ))

    # Falcon tube grasp: approach from above, grasp the cap.
    # Tube body center is at z=0.8175. Cap at z=0.881. Grasp the cap from
    # above by bringing the EE site to z=0.85 (gripper body 30mm above EE
    # in world frame, so gripper body is at z=0.88, right at the cap).
    tube_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.5,
        approach_standoff_mm=60.0,
    )

    # Left rack tubes — scattered across a 4-col x 2-row grid (8 slots, 3 filled)
    # Coordinates derived from rack at (-0.45, 0.15) + slot offset.
    # L1 col1 row1 (front-left), L2 col2 row2 (back), L3 col4 row1 (front-right)
    reg.register(LabObject(
        name="tube_L1",
        aliases=["tube L1", "left rack column 1 row 1",
                 "front-left orange tube on the left", "orange tube front-left"],
        position_xyz_m=[-0.510, 0.130, 0.8175],
        optimal_rail_mm=0.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with orange cap, in left rack col 1 row 1 (front-left). Grasp at cap height (~881mm).",
        object_type="tube", cap_color="orange",
    ))
    reg.register(LabObject(
        name="tube_L2",
        aliases=["tube L2", "left rack column 2 row 2",
                 "back blue tube on the left", "blue tube back row left rack"],
        position_xyz_m=[-0.470, 0.170, 0.8175],
        optimal_rail_mm=0.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with blue cap, in left rack col 2 row 2 (back row). Grasp at cap height (~881mm).",
        object_type="tube", cap_color="blue",
    ))
    reg.register(LabObject(
        name="tube_L3",
        aliases=["tube L3", "left rack column 4 row 1",
                 "front-right orange tube on the left",
                 "orange tube front-right of left rack"],
        position_xyz_m=[-0.390, 0.130, 0.8175],
        optimal_rail_mm=0.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with orange cap, in left rack col 4 row 1 (front-right). Grasp at cap height (~881mm).",
        object_type="tube", cap_color="orange",
    ))

    # Right-rack tubes. The rack body itself has been relocated to the
    # back-left of the bench (directly behind left_tube_rack), so these
    # tubes now sit on the LEFT side of the bench despite their R*
    # naming. The R name is kept so prior lessons.md / world_model
    # references stay valid.
    # R1 col1 row2, R2 col3 row1, R3 col4 row2.
    reg.register(LabObject(
        name="tube_R1",
        aliases=["tube R1", "right rack column 1 row 2",
                 "back blue tube in the back rack",
                 "blue tube in the back-left rack column 1"],
        position_xyz_m=[-0.510, 0.320, 0.8175],
        optimal_rail_mm=0.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with blue cap, in right_tube_rack col 1 row 2. Located on the BACK-LEFT of the bench (the rack has been moved). Grasp at cap height (~881mm).",
        object_type="tube", cap_color="blue",
    ))
    reg.register(LabObject(
        name="tube_R2",
        aliases=["tube R2", "right rack column 3 row 1",
                 "front orange tube in the back rack"],
        position_xyz_m=[-0.430, 0.280, 0.8175],
        optimal_rail_mm=0.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with orange cap, in right_tube_rack col 3 row 1 (front row). Located on the BACK-LEFT of the bench. Grasp at cap height (~881mm).",
        object_type="tube", cap_color="orange",
    ))
    reg.register(LabObject(
        name="tube_R3",
        aliases=["tube R3", "right rack column 4 row 2",
                 "back blue tube in the back rack column 4"],
        position_xyz_m=[-0.390, 0.320, 0.8175],
        optimal_rail_mm=0.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with blue cap, in right_tube_rack col 4 row 2. Located on the BACK-LEFT of the bench. Grasp at cap height (~881mm).",
        object_type="tube", cap_color="blue",
    ))

    # Racks (static — listed so the LLM knows they're there as fixtures, not graspable)
    rack_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.0, approach_standoff_mm=0.0,
    )
    reg.register(LabObject(
        name="left_tube_rack",
        aliases=["left rack", "left tube rack"],
        position_xyz_m=[-0.45, 0.15, 0.755],
        optimal_rail_mm=0.0,
        grasp=rack_grasp,
        safety_notes="Static fixture holding tube_L1, tube_L2, tube_L3. NOT graspable.",
        object_type="rack",
    ))
    reg.register(LabObject(
        name="right_tube_rack",
        aliases=["right rack", "right tube rack", "back tube rack",
                 "back rack", "the second tube rack"],
        position_xyz_m=[-0.45, 0.30, 0.755],
        optimal_rail_mm=0.0,
        grasp=rack_grasp,
        safety_notes="Static fixture holding tube_R1, tube_R2, tube_R3. NOT graspable. RELOCATED: this rack used to be at (+0.45, +0.15) on the right side of the bench but has been moved to (-0.45, +0.30) -- directly behind left_tube_rack on the back-left of the bench. The 'right' in the name is historical.",
        object_type="rack",
    ))

    # Opentrons OT-2 (static fixture, not actuated by us). Sits adjacent
    # to the bench on the +x side with its deck at bench height (z=0.75 m)
    # so the arm can reach in from above. The xArm6 picks/places a
    # 96-well plate from/to one of the deck slots.
    ot2_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.0, approach_standoff_mm=0.0,
    )
    reg.register(LabObject(
        name="opentrons_ot2",
        aliases=["ot2", "ot-2", "opentrons", "pipetting robot",
                 "liquid handler"],
        position_xyz_m=[1.0, 0.0, 0.0],
        optimal_rail_mm=700.0,
        grasp=ot2_grasp,
        safety_notes=(
            "Static instrument adjacent to bench on the +x side. Outer "
            "footprint 630 x 570 mm. Side walls, back wall, and top are "
            "COLLIDING obstacles; the ONLY access is through the front "
            "(-x) opening that faces the bench. Deck surface at z=755 mm. "
            "11 SBS slots (3 cols x 4 rows) + TRASH at slot 12. Pitch "
            "132 mm cols x 89 mm rows. Slot world-xy (z=755 for all): "
            "  slot 1=(867, +132), 2=(867, 0), 3=(867, -132) [front row], "
            "  slot 4=(956, +132), 5=(956, 0), 6=(956, -132) [mid-front], "
            "  slot 7=(1044, +132), 8=(1044, 0), 9=(1044, -132) [mid-back], "
            "  slot 10=(1133, +132), 11=(1133, 0), TRASH=(1133, -132). "
            "REACH WARNING from rail=700: slots 1-6 reach cleanly; "
            "slots 7-9 are at the edge of arm reach; slots 10-11 + "
            "TRASH are usually unreachable through the front opening. "
            "Slots are visually identified by coloured corner tags: "
            "1=red, 2=orange, 3=yellow, 4=yellow-green, 5=green, "
            "6=teal, 7=cyan, 8=blue, 9=purple, 10=pink, 11=white, "
            "TRASH=dark grey. "
            "Do NOT try to grasp the OT-2 itself."
        ),
        object_type="instrument",
    ))

    # 96-well plates (graspable). Plate A starts in OT-2 slot 1
    # (front-left of deck); Plate B starts on the bench front-right.
    plate_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.7,
        approach_standoff_mm=50.0,
    )
    reg.register(LabObject(
        name="well_plate_A",
        aliases=["plate A", "well plate A", "well_plate_a", "plate a",
                 "96-well plate A", "OT-2 plate", "the plate on the OT-2",
                 "the plate on the deck"],
        position_xyz_m=[0.867, +0.132, 0.7625],
        optimal_rail_mm=700.0,
        grasp=plate_grasp,
        safety_notes=(
            "96-well SBS plate. Start position: OT-2 slot 1 "
            "(front-left of deck). Plate body centre z=762; top z=770. "
            "Grasp from directly above. Approach via OT-2 front opening."
        ),
        object_type="plate",
    ))
    reg.register(LabObject(
        name="well_plate_B",
        aliases=["plate B", "well plate B", "well_plate_b", "plate b",
                 "96-well plate B", "bench plate", "the plate on the bench"],
        position_xyz_m=[+0.550, -0.200, 0.7625],
        optimal_rail_mm=400.0,
        grasp=plate_grasp,
        safety_notes=(
            "96-well SBS plate. Start position: bench front-right at "
            "(+550, -200), between the PCR thermocycler and the OT-2. "
            "Plate body centre z=762; top z=770. Standard top-down grasp."
        ),
        object_type="plate",
    ))

    # Opentrons tip rack (96-position SBS). Starts on the OT-2 deck in
    # slot 4 (mid-front-left). Graspable.
    tip_box_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.7,
        approach_standoff_mm=60.0,
    )
    reg.register(LabObject(
        name="tip_box",
        aliases=["tip rack", "tip box", "pipette tips", "tips",
                 "tip_rack", "96-tip rack"],
        position_xyz_m=[1.133, +0.132, 0.795],
        optimal_rail_mm=700.0,
        grasp=tip_box_grasp,
        safety_notes=(
            "Opentrons-style 96-position tip rack. 127 x 85 x 80 mm "
            "(taller than a plate). Start position: OT-2 slot 10 "
            "(back-left, world (1133, +132)). Body centre z=795; "
            "top z=835. Grasp from directly above. "
            "REACH WARNING: slot 10 is at the back of the OT-2 deck, "
            "at the edge of the arm's reach through the front "
            "opening. Pick may fail with IK errors -- if so, try "
            "alternative rail positions or wrist orientations."
        ),
        object_type="plate",
    ))

    # Opentrons Heater-Shaker module (152 W x 90 D x 82 H mm). Sits on
    # the bench, holds a 96-well plate on its top platform. Heavy so it
    # stays put; not normally picked up but technically graspable (so
    # push_object works on it if needed).
    shaker_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.7,
        approach_standoff_mm=60.0,
    )
    reg.register(LabObject(
        name="heater_shaker",
        aliases=["shaker", "heater shaker", "opentrons shaker",
                 "heater-shaker", "shaker module"],
        position_xyz_m=[-0.300, -0.250, 0.791],
        optimal_rail_mm=50.0,
        grasp=shaker_grasp,
        safety_notes=(
            "Opentrons Heater-Shaker module on the bench front-left. "
            "152 x 90 x 82 mm. Top platform at z=836 mm holds a "
            "96-well plate. To PLACE a plate on the shaker: approach "
            "(-300, -250, 870), descend to (-300, -250, 845), open "
            "gripper. Heavy (2 kg); do NOT push or pick up unless the "
            "task explicitly asks. Two status LEDs on the left and "
            "right faces of the chassis indicate plate placement: "
            "GREEN = plate seated correctly (within +/-15 mm of "
            "platform centre, upright); RED = plate present but "
            "mis-aligned (re-pick and re-place); OFF = no plate."
        ),
        object_type="instrument",
    ))

    # Vortex-Genie 2 (classic benchtop vortex mixer). Static fixture on
    # the bench at the position where red_cube/red_bin used to live.
    # 165 W x 122 D x 165 H mm. Its top platform orbits at 4 mm radius
    # automatically when something rests on it (driven by
    # SimXArmAPI._vortex_tick).
    vortex_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.0, approach_standoff_mm=0.0,
    )
    # Opentrons Thermocycler Module (PCR machine). Static fixture
    # with an actuated lid (open/close via pcr_open / pcr_close
    # commands). Internal cavity at the chassis centre holds a
    # 96-well plate on a heated block. Two side LED indicators
    # auto-report state (RED = empty + lid closed, GREEN = plate
    # inside, grey = empty + lid open).
    pcr_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.0, approach_standoff_mm=0.0,
    )
    reg.register(LabObject(
        name="pcr_module",
        aliases=["pcr", "thermocycler", "thermocycler module",
                 "pcr machine", "opentrons thermocycler",
                 "thermocycler gen2"],
        position_xyz_m=[+0.200, -0.300, 0.750],
        optimal_rail_mm=350.0,
        grasp=pcr_grasp,
        safety_notes=(
            "Opentrons Thermocycler Module at world (+200, -300). "
            "Outer ~350 W (along x) x 170 D (along y) x 130 H mm. "
            "Footprint: x in (25..375)mm, y in (-385..-215)mm. "
            "Chassis walls top out at world z=850 mm -- any transit "
            "across the footprint with a plate in the gripper MUST "
            "be at z>=920 mm or the plate will collide with the wall. "
            "The lid hinges on the -x side and OPENS UPWARD AWAY "
            "FROM THE OT-2 (when open the lid stands vertical at "
            "world x ~30mm, pointing UP). pcr_open MUST be called "
            "before any plate can be loaded or removed. Cavity "
            "centre at world (+200, -300, 765); plate sits on the "
            "heated block with its body centre at ~z=780. To LOAD a "
            "plate: pcr_open, traverse at z>=920 (NOT 870), approach "
            "(+200, -300, 920), descend to (+200, -300, 790), "
            "gripper_open, lift back to (+200, -300, 920), pcr_close. "
            "Two LEDs on the front and back faces of the chassis "
            "indicate state: GREEN = plate correctly seated AND lid "
            "closed; RED = plate inside but mis-positioned (off-centre "
            "or tilted) AND lid closed; OFF = lid open OR no plate "
            "inside. Read-only, not controllable directly. "
            "The PCR chassis is heavy; do NOT push or grasp it. Do "
            "NOT issue pcr_close while the gripper is still inside "
            "the cavity."
        ),
        object_type="instrument",
    ))

    reg.register(LabObject(
        name="vortex_genie",
        aliases=["vortex", "vortex-genie", "vortex genie", "genie",
                 "vortex mixer", "vortex-genie 2", "genie 2"],
        position_xyz_m=[-0.200, 0.250, 0.750],
        optimal_rail_mm=150.0,
        grasp=vortex_grasp,
        safety_notes=(
            "Vortex-Genie 2 (classic benchtop vortex mixer). Static "
            "instrument on the bench at (-200, +250). Chassis 165 W x "
            "122 D x 165 H mm; top platform centre at world "
            "(-200, +250, 905), platform top surface at z=910 mm. "
            "AUTO-ON: when ANY movable body (tube, cube, plate, etc.) "
            "is resting on the platform (xy within 50 mm of platform "
            "centre and z within 80 mm above platform top), the "
            "platform starts orbiting at 4 mm radius / ~25 Hz; the "
            "object on it follows via friction. To vortex a tube: "
            "grasp it from its rack, position the gripper above "
            "(-200, +250, 940), descend to (-200, +250, 920) so the "
            "tube body rests on the platform, release. The vortex "
            "starts shaking automatically. Lift the gripper away "
            "first to get a clear view of the motion."
        ),
        object_type="instrument",
    ))

    return reg
