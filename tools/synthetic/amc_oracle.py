from __future__ import annotations

import json
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

        root = AcclaimRoot(
            order=[token.lower() for token in AcclaimParser._parse_word_list(root_block.group(1), "order")],
            axis_order=[f"r{axis.lower()}" for axis in AcclaimParser._parse_word_list(root_block.group(1), "axis")[0]],
            position=np.array(AcclaimParser._parse_float_list(root_block.group(1), "position")),
            orientation=np.array(AcclaimParser._parse_float_list(root_block.group(1), "orientation")),
        )

        bones: dict[str, AcclaimBone] = {}
        bone_data = re.search(r":bonedata\s+(.*?)(?=\n:hierarchy)", text, re.DOTALL)
        if bone_data is None:
            raise ValueError("ASF missing :bonedata section")

        for block in re.findall(r"begin\s+(.*?)\s+end", bone_data.group(1), re.DOTALL):
            name = AcclaimParser._parse_word_list(block, "name")[0]
            direction = np.array(AcclaimParser._parse_float_list(block, "direction"))
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            length = float(AcclaimParser._parse_float_list(block, "length")[0])
            axis_tokens = AcclaimParser._parse_line_tokens(block, "axis")
            axis_angles = [float(token) for token in axis_tokens[:3]]
            axis_order = [f"r{axis.lower()}" for axis in axis_tokens[3]]
            dof = AcclaimParser._parse_word_list(block, "dof") if "dof" in block else []
            bones[name] = AcclaimBone(
                name=name,
                direction=direction,
                length=length,
                axis_order=axis_order,
                axis_angles=dict(zip(["rx", "ry", "rz"], axis_angles)),
                dof=dof,
            )

        hierarchy_match = re.search(r":hierarchy\s+begin\s+(.*?)\s+end", text, re.DOTALL)
        if hierarchy_match is None:
            raise ValueError("ASF missing :hierarchy section")
        for line in hierarchy_match.group(1).strip().splitlines():
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
    def parse_amc(path: str | Path, root_order: list[str], bone_map: dict[str, AcclaimBone]) -> list[dict[str, dict[str, float]]]:
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
        return [float(token) for token in AcclaimParser._parse_line_tokens(block, key)]

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
        root_translation = np.array(
            [
                root_values.get("tx", self.skeleton.root.position[0]),
                root_values.get("ty", self.skeleton.root.position[1]),
                root_values.get("tz", self.skeleton.root.position[2]),
            ],
            dtype=float,
        )
        root_angles = {
            "rx": root_values.get("rx", self.skeleton.root.orientation[0]),
            "ry": root_values.get("ry", self.skeleton.root.orientation[1]),
            "rz": root_values.get("rz", self.skeleton.root.orientation[2]),
        }
        root_rotation = euler_rotation(self.skeleton.root.axis_order, root_angles)
        world_positions["root"] = root_translation

        def visit(bone_name: str, parent_position: np.ndarray, parent_rotation: np.ndarray) -> None:
            bone = self.skeleton.bones[bone_name]
            local_angles = {axis: frame.get(bone_name, {}).get(axis, 0.0) for axis in bone.dof}
            local_rotation = bone.basis @ euler_rotation(bone.dof, local_angles) @ bone.basis.T
            world_rotation = parent_rotation @ local_rotation
            world_position = parent_position + parent_rotation @ bone.offset
            world_positions[bone_name] = world_position
            for child in bone.children:
                visit(child, world_position, world_rotation)

        roots = [name for name, bone in self.skeleton.bones.items() if bone.parent == "root"]
        for child in roots:
            visit(child, root_translation, root_rotation)
        return world_positions

    def solve_sequence(self, frames: list[dict[str, dict[str, float]]]) -> list[dict[str, np.ndarray]]:
        return [self.solve_frame(frame) for frame in frames]


class Coco17Adapter:
    JOINT_NAMES = {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    }

    def map_frame(self, world_positions: dict[str, np.ndarray]) -> np.ndarray:
        head = world_positions["head"]
        left_shoulder = world_positions["lhumerus"]
        right_shoulder = world_positions["rhumerus"]
        left_hip = world_positions["lfemur"]
        right_hip = world_positions["rfemur"]
        face_right = right_shoulder - left_shoulder
        face_right = face_right / (np.linalg.norm(face_right) + 1e-6)
        face_forward = np.cross(np.array([0.0, 0.0, 1.0]), face_right)
        face_forward = face_forward / (np.linalg.norm(face_forward) + 1e-6)
        eye_offset = face_forward * 4.0
        ear_offset = face_forward * 7.0

        joints = np.zeros((17, 3), dtype=float)
        joints[0] = head + np.array([0.0, 0.0, 4.0])
        joints[1] = head + eye_offset + np.array([-1.0, 0.0, 2.0])
        joints[2] = head + eye_offset + np.array([1.0, 0.0, 2.0])
        joints[3] = head + ear_offset + np.array([-4.0, 0.0, 0.0])
        joints[4] = head + ear_offset + np.array([4.0, 0.0, 0.0])
        joints[5] = left_shoulder
        joints[6] = right_shoulder
        joints[7] = world_positions["lradius"]
        joints[8] = world_positions["rradius"]
        joints[9] = world_positions["lwrist"]
        joints[10] = world_positions["rwrist"]
        joints[11] = left_hip
        joints[12] = right_hip
        joints[13] = world_positions["ltibia"]
        joints[14] = world_positions["rtibia"]
        joints[15] = world_positions["lfoot"]
        joints[16] = world_positions["rfoot"]
        return joints

    def map_sequence(self, sequence: list[dict[str, np.ndarray]]) -> np.ndarray:
        return np.stack([self.map_frame(frame) for frame in sequence])


def resample_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    if len(sequence) == target_length:
        return sequence
    if len(sequence) < 2:
        return np.repeat(sequence, target_length, axis=0)
    sample_positions = np.linspace(0.0, len(sequence) - 1, target_length)
    lower = np.floor(sample_positions).astype(int)
    upper = np.ceil(sample_positions).astype(int)
    alpha = sample_positions - lower
    return ((1.0 - alpha)[:, None, None] * sequence[lower]) + (alpha[:, None, None] * sequence[upper])


def synthetic_ball_track(coco_sequence: np.ndarray, label: str) -> np.ndarray:
    right_wrist = coco_sequence[:, 10]
    if label == "jump_shot":
        progress = np.linspace(0.0, 1.0, len(coco_sequence))
        arc = np.stack(
            [
                progress * 35.0,
                np.zeros_like(progress),
                12.0 + 90.0 * progress - 35.0 * np.square(progress - 0.6),
            ],
            axis=1,
        )
        return right_wrist + arc
    return right_wrist + np.array([6.0, 0.0, 0.0])


def generate_oracle_sample(asf_path: str | Path, amc_path: str | Path, label: str) -> dict[str, object]:
    try:
        from tools.synthetic.generate_data import compute_features_v2
        from tools.synthetic.generate_data import get_look_at_matrix
        from tools.synthetic.generate_data import project_to_2d
    except ModuleNotFoundError:
        from generate_data import compute_features_v2
        from generate_data import get_look_at_matrix
        from generate_data import project_to_2d

    skeleton = AcclaimParser.parse_asf(asf_path)
    frames = AcclaimParser.parse_amc(amc_path, skeleton.root.order, skeleton.bones)
    oracle = KinematicOracle(skeleton)
    adapter = Coco17Adapter()

    world_sequence = oracle.solve_sequence(frames)
    coco_3d = adapter.map_sequence(world_sequence)
    coco_3d = resample_sequence(coco_3d, target_length=30)
    ball_3d = resample_sequence(synthetic_ball_track(coco_3d, label), target_length=30)

    camera_matrix = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])
    camera_position = np.array([220.0, -520.0, 240.0])
    camera_target = np.array([0.0, 0.0, 110.0])
    rotation = get_look_at_matrix(camera_position, camera_target)
    translation = -rotation @ camera_position

    projected = project_to_2d(coco_3d, camera_matrix, rotation, translation, noise_std=0.0)
    normalized = projected.copy()
    for frame_idx in range(len(projected)):
        x_min = projected[frame_idx, :, 0].min()
        y_min = projected[frame_idx, :, 1].min()
        width = projected[frame_idx, :, 0].max() - x_min + 1e-6
        height = projected[frame_idx, :, 1].max() - y_min + 1e-6
        normalized[frame_idx, :, 0] = (projected[frame_idx, :, 0] - x_min) / width
        normalized[frame_idx, :, 1] = (projected[frame_idx, :, 1] - y_min) / height

    features = compute_features_v2(normalized, coco_3d, ball_3d)
    return {
        "label": label,
        "schema_version": "2.0.0",
        "features_v2": features,
    }


def write_oracle_dataset(output_path: str | Path, samples: list[dict[str, object]]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")
