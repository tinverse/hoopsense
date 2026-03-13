from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _rotation_matrix_x(angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_matrix_y(angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rotation_matrix_z(angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_rotation(order: str, values: dict[str, float]) -> np.ndarray:
    matrices = {
        "rx": _rotation_matrix_x,
        "ry": _rotation_matrix_y,
        "rz": _rotation_matrix_z,
    }
    result = np.eye(3)
    for axis in order:
        result = result @ matrices[axis](values.get(axis, 0.0))
    return result


@dataclass
class AcclaimRoot:
    order: list[str]
    axis_order: list[str]
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class AcclaimBone:
    name: str
    direction: np.ndarray
    length: float
    axis_order: list[str]
    axis_angles: dict[str, float]
    dof: list[str]
    children: list[str] = field(default_factory=list)
    parent: str | None = None

    @property
    def offset(self) -> np.ndarray:
        return self.direction * self.length

    @property
    def basis(self) -> np.ndarray:
        return euler_rotation(self.axis_order, self.axis_angles)


@dataclass
class AcclaimSkeleton:
    root: AcclaimRoot
    bones: dict[str, AcclaimBone]


class AcclaimParser:
    @staticmethod
    def parse_asf(path: str | Path) -> AcclaimSkeleton:
        text = Path(path).read_text()
        root_block = re.search(r":root\s+(.*?)(?=\n:)", text, re.DOTALL)
        if root_block is None:
            raise ValueError("ASF missing :root section")

        r_data = root_block.group(1)
        root = AcclaimRoot(
            order=[token.lower() for token in
                   AcclaimParser._parse_word_list(r_data, "order")],
            axis_order=[f"r{axis.lower()}" for axis in
                        AcclaimParser._parse_word_list(r_data, "axis")[0]],
            position=np.array(AcclaimParser._parse_float_list(r_data,
                                                              "position")),
            orientation=np.array(AcclaimParser._parse_float_list(r_data,
                                                                 "orientation")),
        )

        bones: dict[str, AcclaimBone] = {}
        bone_match = re.search(r":bonedata\s+(.*?)(?=\n:hierarchy)",
                               text, re.DOTALL)
        if bone_match is None:
            raise ValueError("ASF missing :bonedata section")

        for block in re.findall(r"begin\s+(.*?)\s+end",
                                bone_match.group(1), re.DOTALL):
            name = AcclaimParser._parse_word_list(block, "name")[0]
            direction = np.array(AcclaimParser._parse_float_list(block,
                                                                 "direction"))
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            length = float(AcclaimParser._parse_float_list(block, "length")[0])
            axis_tokens = AcclaimParser._parse_line_tokens(block, "axis")
            axis_angles = [float(token) for token in axis_tokens[:3]]
            axis_order = [f"r{axis.lower()}" for axis in axis_tokens[3]]
            dof = AcclaimParser._parse_word_list(block, "dof") \
                if "dof" in block else []
            bones[name] = AcclaimBone(
                name=name,
                direction=direction,
                length=length,
                axis_order=axis_order,
                axis_angles=dict(zip(["rx", "ry", "rz"], axis_angles)),
                dof=dof,
            )

        h_match = re.search(r":hierarchy\s+begin\s+(.*?)\s+end",
                            text, re.DOTALL)
        if h_match is None:
            raise ValueError("ASF missing :hierarchy section")
        for line in h_match.group(1).strip().splitlines():
            parts = line.split()
            if not parts:
                continue
            parent = parts[0]
            for child in parts[1:]:
                if child in bones:
                    bones[child].parent = parent
                    if parent in bones:
                        bones[parent].children.append(child)

        return AcclaimSkeleton(root=root, bones=bones)

    @staticmethod
    def parse_amc(path: str | Path,
                  root_order: list[str],
                  bone_map: dict[str, AcclaimBone]):
        frames: list[dict[str, dict[str, float]]] = []
        current: dict[str, dict[str, float]] | None = None
        for raw_line in Path(path).read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(":"):
                continue
            parts = line.split()
            if parts[0].isdigit():
                if current is not None:
                    frames.append(current)
                current = {}
                continue
            if current is None:
                continue
            bone_name = parts[0]
            values = [float(value) for value in parts[1:]]
            if bone_name == "root":
                current["root"] = dict(zip(root_order, values))
            elif bone_name in bone_map:
                current[bone_name] = dict(zip(bone_map[bone_name].dof, values))
        if current is not None:
            frames.append(current)
        return frames

    @staticmethod
    def _parse_word_list(block: str, key: str) -> list[str]:
        return AcclaimParser._parse_line_tokens(block, key)

    @staticmethod
    def _parse_float_list(block: str, key: str) -> list[float]:
        return [float(token) for token in
                AcclaimParser._parse_line_tokens(block, key)]

    @staticmethod
    def _parse_line_tokens(block: str, key: str) -> list[str]:
        match = re.search(rf"{re.escape(key)}\s+([^\n]+)", block)
        if match is None:
            raise ValueError(f"Missing key '{key}' in block")
        return match.group(1).split()


class KinematicOracle:
    def __init__(self, skeleton: AcclaimSkeleton):
        self.skeleton = skeleton

    def solve_frame(self, frame: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
        world_positions: dict[str, np.ndarray] = {}

        root_values = frame.get("root", {})
        # HS(x, y, z) = CMU(x, z, y)
        root_translation = np.array(
            [
                root_values.get("tx", self.skeleton.root.position[0]),
                root_values.get("tz", self.skeleton.root.position[2]),
                root_values.get("ty", self.skeleton.root.position[1]),
            ],
            dtype=float,
        ) * 2.54

        root_angles = {
            "rx": root_values.get("rx", self.skeleton.root.orientation[0]),
            "ry": root_values.get("ry", self.skeleton.root.orientation[1]),
            "rz": root_values.get("rz", self.skeleton.root.orientation[2]),
        }
        root_rotation = euler_rotation(self.skeleton.root.axis_order,
                                       root_angles)
        world_positions["root"] = root_translation

        def visit(bone_name: str,
                  p_pos: np.ndarray,
                  p_rot: np.ndarray) -> None:
            bone = self.skeleton.bones[bone_name]
            l_angles = {ax: frame.get(bone_name, {}).get(ax, 0.0)
                        for ax in bone.dof}
            l_rot = (bone.basis @
                     euler_rotation(bone.dof, l_angles) @
                     bone.basis.T)
            w_rot = p_rot @ l_rot
            w_pos = p_pos + p_rot @ (bone.offset * 2.54)
            world_positions[bone_name] = w_pos
            for child in bone.children:
                visit(child, w_pos, w_rot)

        roots = [n for n, b in self.skeleton.bones.items() if b.parent == "root"]
        for child in roots:
            visit(child, root_translation, root_rotation)
        return world_positions

    def solve_sequence(self,
                       frames: list[dict[str, dict[str, float]]]):
        return [self.solve_frame(frame) for frame in frames]


class Coco17Adapter:
    def map_frame(self, world_positions: dict[str, np.ndarray]) -> np.ndarray:
        head = world_positions["head"]
        ls = world_positions["lhumerus"]
        rs = world_positions["rhumerus"]
        lh = world_positions["lfemur"]
        rh = world_positions["rfemur"]
        fr = rs - ls
        fr = fr / (np.linalg.norm(fr) + 1e-6)
        ff = np.cross(np.array([0.0, 0.0, 1.0]), fr)
        ff = ff / (np.linalg.norm(ff) + 1e-6)
        joints = np.zeros((17, 3), dtype=float)
        joints[0] = head + ff * 5.0
        joints[1] = head + ff * 4.0 + np.array([-2.0, 0.0, 2.0])
        joints[2] = head + ff * 4.0 + np.array([2.0, 0.0, 2.0])
        joints[3] = head + ff * 2.0 + np.array([-5.0, 0.0, 1.0])
        joints[4] = head + ff * 2.0 + np.array([5.0, 0.0, 1.0])
        joints[5:7] = [ls, rs]
        joints[7:9] = [world_positions["lradius"], world_positions["rradius"]]
        joints[9:11] = [world_positions["lwrist"], world_positions["rwrist"]]
        joints[11:13] = [lh, rh]
        joints[13:15] = [world_positions["ltibia"], world_positions["rtibia"]]
        joints[15:17] = [world_positions["lfoot"], world_positions["rfoot"]]
        return joints

    def map_sequence(self,
                     sequence: list[dict[str, np.ndarray]]) -> np.ndarray:
        return np.stack([self.map_frame(frame) for frame in sequence])


def generate_oracle_sample(asf_path: str | Path,
                           amc_path: str | Path,
                           label: str) -> dict[str, object]:
    from tools.synthetic.generate_data import (compute_features_v2,
                                               get_look_at_matrix,
                                               project_to_2d)
    skeleton = AcclaimParser.parse_asf(asf_path)
    frames = AcclaimParser.parse_amc(amc_path,
                                     skeleton.root.order,
                                     skeleton.bones)
    oracle = KinematicOracle(skeleton)
    adapter = Coco17Adapter()
    world_sequence = oracle.solve_sequence(frames)
    coco_3d = adapter.map_sequence(world_sequence)
    if len(coco_3d) > 30:
        indices = np.linspace(0, len(coco_3d) - 1, 30).astype(int)
        coco_3d = coco_3d[indices]
    ball_3d = coco_3d[:, 10] + np.array([5, 0, 5])
    c_mat = np.array([[1000.0, 0.0, 960.0],
                      [0.0, 1000.0, 540.0],
                      [0.0, 0.0, 1.0]])
    c_pos = np.array([300.0, -600.0, 250.0])
    rot = get_look_at_matrix(c_pos, np.array([0, 0, 100]))
    trans = -rot @ c_pos
    skel_2d = project_to_2d(coco_3d, c_mat, rot, trans)
    norm = skel_2d.copy()
    for t in range(len(skel_2d)):
        bbox = [skel_2d[t, :, 0].min(), skel_2d[t, :, 1].min(),
                skel_2d[t, :, 0].max(), skel_2d[t, :, 1].max()]
        w, h = bbox[2]-bbox[0]+1e-6, bbox[3]-bbox[1]+1e-6
        norm[t, :, 0] = (skel_2d[t, :, 0] - bbox[0]) / w
        norm[t, :, 1] = (skel_2d[t, :, 1] - bbox[1]) / h
    features = compute_features_v2(norm, coco_3d, ball_3d)
    return {"label": label, "schema_version": "2.0.0",
            "features_v2": features}
