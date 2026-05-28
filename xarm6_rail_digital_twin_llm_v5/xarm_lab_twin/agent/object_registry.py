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

    def to_llm_context(self) -> str:
        cubes = [o for o in self.objects.values() if o.object_type == "cube"]
        tubes = [o for o in self.objects.values() if o.object_type == "tube"]
        bins  = [o for o in self.objects.values() if o.object_type == "bin"]
        racks = [o for o in self.objects.values() if o.object_type == "rack"]
        plates = [o for o in self.objects.values() if o.object_type == "plate"]
        instruments = [o for o in self.objects.values()
                       if o.object_type == "instrument"]
        lines = []

        def fmt_basic(obj):
            x, y, z = obj.position_xyz_m
            return (
                f"- **{obj.name}**  aliases: {', '.join(obj.aliases)}\n"
                f"  Position: x={x*1000:.0f}mm  y={y*1000:.0f}mm  z={z*1000:.0f}mm\n"
                f"  Optimal rail: {obj.optimal_rail_mm:.0f}mm\n"
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

    reg.register(LabObject(
        name="red_cube",
        aliases=["red cube", "red block", "red"],
        position_xyz_m=[-0.20, 0.15, 0.78],
        optimal_rail_mm=150.0,
        grasp=cube_grasp,
        safety_notes="Small graspable cube. Approach from above.",
        object_type="cube",
    ))
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

    reg.register(LabObject(
        name="red_bin",
        aliases=["red bin", "red container", "red box"],
        position_xyz_m=[-0.20, 0.35, 0.75],
        optimal_rail_mm=150.0,
        grasp=bin_grasp,
        safety_notes="Open-top bin. Release cube above bin opening.",
        is_container=True,
        object_type="bin",
    ))
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

    # Right rack tubes
    # R1 col1 row2 (back-left), R2 col3 row1 (front-right), R3 col4 row2 (back-right)
    reg.register(LabObject(
        name="tube_R1",
        aliases=["tube R1", "right rack column 1 row 2",
                 "back-left blue tube on the right", "blue tube back-left right rack"],
        position_xyz_m=[0.390, 0.170, 0.8175],
        optimal_rail_mm=700.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with blue cap, in right rack col 1 row 2 (back-left). Grasp at cap height (~881mm).",
        object_type="tube", cap_color="blue",
    ))
    reg.register(LabObject(
        name="tube_R2",
        aliases=["tube R2", "right rack column 3 row 1",
                 "front-right orange tube on the right"],
        position_xyz_m=[0.470, 0.130, 0.8175],
        optimal_rail_mm=700.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with orange cap, in right rack col 3 row 1 (front). Grasp at cap height (~881mm).",
        object_type="tube", cap_color="orange",
    ))
    reg.register(LabObject(
        name="tube_R3",
        aliases=["tube R3", "right rack column 4 row 2",
                 "back-right blue tube on the right"],
        position_xyz_m=[0.510, 0.170, 0.8175],
        optimal_rail_mm=700.0,
        grasp=tube_grasp,
        safety_notes="Falcon tube with blue cap, in right rack col 4 row 2 (back-right). Grasp at cap height (~881mm).",
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
        aliases=["right rack", "right tube rack"],
        position_xyz_m=[0.45, 0.15, 0.755],
        optimal_rail_mm=700.0,
        grasp=rack_grasp,
        safety_notes="Static fixture holding tube_R1, tube_R2, tube_R3. NOT graspable.",
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
            "Static instrument adjacent to bench on the +x side. Deck "
            "surface at z=755mm. 6 SBS slots (3 cols x 2 rows) with "
            "pitch 130mm x 94mm; slot centres in world coords: "
            "front-left (870, -90), front-mid (1000, -90), front-right "
            "(1130, -90), back-left (870, +90), back-mid (1000, +90), "
            "back-right (1130, +90). Enclosure walls are non-colliding "
            "(visual only) so the arm can descend through the top to "
            "reach a slot. Do NOT try to grasp the OT-2 itself."
        ),
        object_type="instrument",
    ))

    # 96-well plate (graspable, sits in the OT-2 front-left slot at scene
    # start). 127 x 85 x 14 mm. The plate is the *movable* OT-2 payload
    # the arm picks up or places.
    plate_grasp = GraspConfig(
        approach_direction=[0.0, 0.0, -1.0],
        grip_orientation_rpy=[180.0, 0.0, 0.0],
        grip_depth=0.7,
        approach_standoff_mm=50.0,
    )
    reg.register(LabObject(
        name="well_plate",
        aliases=["plate", "96-well plate", "well plate", "microplate",
                 "96 well plate"],
        position_xyz_m=[0.870, -0.090, 0.7625],
        optimal_rail_mm=700.0,
        grasp=plate_grasp,
        safety_notes=(
            "96-well SBS plate. Start position: OT-2 front-left slot. "
            "Plate body centre at z=762mm; top surface at z=770mm. "
            "Grasp from directly above. Reachable with rail near 700mm."
        ),
        object_type="plate",
    ))

    return reg
