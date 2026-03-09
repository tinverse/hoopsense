import numpy as np
import yaml
import os
from enum import Enum
from collections import deque

class EntityState(Enum):
    IDLE = 0
    SHOOTING = 1
    PASSING = 2
    DRIBBLING = 3
    OFFICIAL_SIGNALING = 4

class KinematicRule:
    def evaluate(self, kpts_history):
        raise NotImplementedError

class DeclarativeRule(KinematicRule):
    """
    Evaluates kinematic rules defined in the HoopScript DSL.
    Currently supports metric-based predicates (velocity, position).
    """
    def __init__(self, spec):
        self.id = spec['id']
        self.spec = spec
        self.predicates = spec.get('predicates', [])
        self.preconditions = spec.get('preconditions', [])
        self.conditions = spec.get('conditions', [])
        # Joint mapping for common NCAA names
        self.joint_map = {"hips": [11, 12], "wrists": [9, 10], "head": [0], "shoulders": [5, 6], "ankles": [15, 16]}

    def evaluate(self, kpts):
        """kpts: (T, 17, 2) normalized to box."""
        if len(kpts) < 15: return False
        
        # 1. Evaluate Preconditions (Mocked for now)
        for pre in self.preconditions:
            if pre.get('actor') == 'self' and pre.get('state') == 'has_possession':
                # In a real scenario, we'd check if the ball is near the player
                pass

        # 2. Evaluate Conditions (Mocked for now)
        for cond in self.conditions:
            if cond.get('ball_state') == 'controlled':
                pass

        # 3. Evaluate Predicates
        for pred in self.predicates:
            joint_idx = self.joint_map.get(pred['joint'])
            if not joint_idx: continue
            
            # Extract relevant joint(s) data
            joint_data = kpts[:, joint_idx, :] # (T, num_joints, 2)
            
            metric = pred['metric']
            op = pred['operator']
            threshold = pred.get('threshold', 0.0)
            
            if metric in ["velocity_z", "velocity_y"]:
                # In normalized 2D, vertical movement is Y axis
                # Y-up in image is negative Y-coord
                y_coords = np.mean(joint_data[:, :, 1], axis=1)
                velocity = y_coords[-1] - y_coords[0]
                if op == ">" and not (velocity < -threshold): return False 
                if op == "<" and not (velocity > threshold): return False
                
            elif metric == "velocity_x":
                x_coords = np.mean(joint_data[:, :, 0], axis=1)
                velocity = x_coords[-1] - x_coords[0]
                if op == ">" and not (velocity > threshold): return False
                if op == "<" and not (velocity < -threshold): return False

            elif metric == "pos_y":
                ref_joint = pred.get('reference')
                if ref_joint:
                    ref_idx = self.joint_map.get(ref_joint)
                    ref_y = np.mean(kpts[:, ref_idx, 1], axis=1)
                    curr_y = np.mean(joint_data[:, :, 1], axis=1)
                    # Wrist < Head means Wrist is higher (smaller Y)
                    if op == "<" and not np.mean(curr_y < ref_y) > 0.6: return False
                    
        return True

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
                    if rule_type == "kinematic": target = EntityState.SHOOTING
                    if rule_type == "signal": target = EntityState.OFFICIAL_SIGNALING
                    
                    if target not in engine: engine[target] = []
                    engine[target].append(rule)
        return engine

    def update(self, kpts_history, learned_label=None):
        if learned_label and learned_label != "idle":
            self.custom_label = learned_label
            return self.state

        history_arr = np.array(kpts_history)
        for target_state, rules in self.rules_engine.items():
            for rule in rules:
                if rule.evaluate(history_arr):
                    self.state = target_state
                    self.custom_label = rule.id
                    return self.state
        
        self.state = EntityState.IDLE
        self.custom_label = "idle"
        return self.state

    def get_label(self):
        return self.custom_label
