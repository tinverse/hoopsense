import numpy as np
import yaml
import os
from enum import Enum


class EntityState(Enum):
    IDLE = 0
    SHOOTING = 1
    PASSING = 2
    DRIBBLING = 3
    OFFICIAL_SIGNALING = 4


class KinematicRule:
    def evaluate(self, kpts_history, context=None):
        raise NotImplementedError


class DeclarativeRule(KinematicRule):
    def __init__(self, spec):
        self.id = spec['id']
        self.spec = spec
        self.predicates = spec.get('predicates', [])
        self.preconditions = spec.get('preconditions', [])
        self.conditions = spec.get('conditions', [])
        self.joint_map = {
            "hips": [11, 12],
            "wrists": [9, 10],
            "head": [0],
            "shoulders": [5, 6],
            "ankles": [15, 16]
        }

    def evaluate(self, kpts, context=None):
        if len(kpts) < 15:
            return False
        for pre in self.preconditions:
            if pre.get('actor') == 'self' and pre.get('state') == 'has_possession':
                if not context or not context.get('has_possession'):
                    return False
        for cond in self.conditions:
            if cond.get('ball_state') == 'controlled':
                if not context or context.get('ball_state') != 'controlled':
                    return False
        for pred in self.predicates:
            joint_idx = self.joint_map.get(pred['joint'])
            if not joint_idx:
                continue
            joint_data = kpts[:, joint_idx, :]
            metric = pred['metric']
            op = pred['operator']
            threshold = pred.get('threshold', 0.0)
            if metric in ["velocity_z", "velocity_y"]:
                y_coords = np.mean(joint_data[:, :, 1], axis=1)
                velocity = y_coords[-1] - y_coords[0]
                if op == ">" and not (velocity < -threshold):
                    return False
                if op == "<" and not (velocity > threshold):
                    return False
            elif metric == "velocity_x":
                x_coords = np.mean(joint_data[:, :, 0], axis=1)
                velocity = x_coords[-1] - x_coords[0]
                if op == ">" and not (velocity > threshold):
                    return False
                if op == "<" and not (velocity < -threshold):
                    return False
            elif metric == "pos_y":
                ref_joint = pred.get('reference')
                if ref_joint:
                    ref_idx = self.joint_map.get(ref_joint)
                    ref_y = np.mean(kpts[:, ref_idx, 1], axis=1)
                    curr_y = np.mean(joint_data[:, :, 1], axis=1)
                    if op == "<" and not np.mean(curr_y < ref_y) > 0.6:
                        return False
        return True


class PossessionEngine:
    def __init__(self):
        self.current_handler = None
        self.last_handler = None
        self.catch_threshold = 50.0

    def update(self, player_tracks, ball_pos_3d, t_ms):
        """
        player_tracks: dict of tid -> {'pos_3d': np.array([x,y,z]), 'team': int}
        """
        events = []
        if ball_pos_3d is None:
            return events

        best_tid = None
        min_dist = float('inf')
        for tid, data in player_tracks.items():
            dist = np.linalg.norm(data['pos_3d'] - ball_pos_3d)
            if dist < min_dist:
                min_dist = dist
                best_tid = tid

        if min_dist < self.catch_threshold:
            if self.current_handler != best_tid:
                if self.current_handler is not None:
                    # Pass vs Steal check
                    t1 = player_tracks[best_tid]['team']
                    t2 = player_tracks[self.current_handler]['team']
                    if t1 == t2:
                        events.append({
                            "kind": "pass", "from": self.current_handler,
                            "to": best_tid, "t_ms": t_ms,
                            "x": float(ball_pos_3d[0]), "y": float(ball_pos_3d[1])
                        })
                    else:
                        events.append({
                            "kind": "steal", "player_id": best_tid, "from": self.current_handler, "t_ms": t_ms,
                            "x": float(ball_pos_3d[0]), "y": float(ball_pos_3d[1])
                        })
                self.last_handler = self.current_handler
                self.current_handler = best_tid
                events.append({
                    "kind": "catch", "player_id": best_tid, "t_ms": t_ms,
                    "x": float(ball_pos_3d[0]), "y": float(ball_pos_3d[1])
                })
        elif min_dist > 200.0:
            self.current_handler = None
        return events


class BehaviorStateMachine:
    def __init__(self, is_ref=False, spec_path="specs/basketball_ncaa.yaml"):
        self.state = EntityState.IDLE
        self.custom_label = "idle"
        self.is_ref = is_ref
        self.rules_engine = self._init_rules(spec_path)

    def _init_rules(self, spec_path):
        engine = {}
        if os.path.exists(spec_path):
            with open(spec_path, 'r') as f:
                spec = yaml.safe_load(f)
                for r_spec in spec.get('rules', []):
                    rule = DeclarativeRule(r_spec)
                    rule_type = r_spec.get('type', 'kinematic')
                    target = EntityState.IDLE
                    if rule_type == "kinematic":
                        target = EntityState.SHOOTING
                    if rule_type == "signal":
                        target = EntityState.OFFICIAL_SIGNALING
                    if target not in engine:
                        engine[target] = []
                    engine[target].append(rule)
        return engine

    def update(self, kpts_history, learned_label=None, context=None):
        if learned_label and learned_label != "idle":
            self.custom_label = learned_label
            return self.state
        history_arr = np.array(kpts_history)
        for target_state, rules in self.rules_engine.items():
            for rule in rules:
                if rule.evaluate(history_arr, context=context):
                    self.state = target_state
                    self.custom_label = rule.id
                    return self.state
        self.state = EntityState.IDLE
        self.custom_label = "idle"
        return self.state

    def get_label(self):
        return self.custom_label
