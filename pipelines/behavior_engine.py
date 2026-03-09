import numpy as np
from enum import Enum
from collections import deque

class EntityState(Enum):
    IDLE = 0
    SHOOTING = 1
    PASSING = 2
    DRIBBLING = 3
    OFFICIAL_SIGNALING = 4

class KinematicRule:
    """Base class for a spatial-temporal rule."""
    def evaluate(self, kpts_history):
        raise NotImplementedError

class JumpShotRule(KinematicRule):
    def evaluate(self, kpts):
        if len(kpts) < 15: return False
        # Calculate velocity and pose metrics
        # (Using the same logic as before, but encapsulated as a Rule)
        v_kpts = kpts[np.all(kpts[:, [11, 12], 0] > 0, axis=1)]
        if len(v_kpts) < 5: return False
        
        wrists_above_head = np.mean(v_kpts[:, [9, 10], 1] < v_kpts[:, 0, 1].reshape(-1, 1)) > 0.6
        hips_y = np.mean(v_kpts[:, [11, 12], 1], axis=1)
        is_rising = (hips_y[-1] - hips_y[0]) < -0.02
        return wrists_above_head and is_rising

class Ref3ptSuccessRule(KinematicRule):
    def evaluate(self, kpts):
        if len(kpts) < 10: return False
        v_kpts = kpts[np.all(kpts[:, [5, 6], 0] > 0, axis=1)]
        if len(v_kpts) < 5: return False
        
        left_arm_up = np.mean(v_kpts[:, 9, 1] < v_kpts[:, 5, 1].reshape(-1, 1)) > 0.7
        right_arm_up = np.mean(v_kpts[:, 10, 1] < v_kpts[:, 6, 1].reshape(-1, 1)) > 0.7
        return left_arm_up and right_arm_up

import yaml
import os

class DeclarativeRule(KinematicRule):
    def __init__(self, spec):
        self.id = spec['id']
        self.predicates = spec.get('predicates', [])

    def evaluate(self, kpts):
        # Placeholder for dynamic predicate evaluation logic
        # For now, we still map to the optimized JumpShotRule if id matches
        if self.id == "jump_shot":
            return JumpShotRule().evaluate(kpts)
        return False

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
                    # Use rule type to categorize
                    rule_type = r_spec.get('type', 'kinematic')
                    
                    target = EntityState.IDLE
                    if rule_type == "kinematic": target = EntityState.SHOOTING # Generic active state
                    if rule_type == "signal": target = EntityState.OFFICIAL_SIGNALING
                    if rule_type == "violation": target = EntityState.IDLE # Violations trigger metadata, not always pose
                    
                    if target not in engine: engine[target] = []
                    engine[target].append(rule)
        return engine

    def update(self, kpts_history, learned_label=None):
        """Evaluates rules or learned signals to trigger state transitions."""
        if learned_label and learned_label != "idle":
            self.custom_label = learned_label
            # ... state update logic ...
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
