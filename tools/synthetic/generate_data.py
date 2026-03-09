import numpy as np
import json
import os

class MoveLibrary:
    @staticmethod
    def jump_shot(t):
        """Procedural Jump Shot with 3D Ball Trajectory."""
        skeleton = np.zeros((17, 3))
        skeleton[0] = [0, 0, 180] # head
        # Hips at (1000, 762) - right side of court
        skeleton[11:13] = [[985, 762, 100], [1015, 762, 100]]
        jump_z = 50 * np.sin(np.pi * t)
        skeleton[:, 2] += jump_z
        # Ball starts at chest, goes above head
        ball_pos = np.array([1000, 780, 140 + 160 * t + jump_z])
        return skeleton, ball_pos

    @staticmethod
    def crossover(t):
        """Procedural Crossover with 3D Ball Trajectory."""
        skeleton = np.zeros((17, 3))
        # Moving across the key
        pos_x = 1400 + 100 * t
        skeleton[:, 0] = pos_x
        skeleton[:, 2] = 100 # standing
        # Ball switches from left to right hand
        if t < 0.5:
            ball_pos = np.array([pos_x - 30, 762, 40])
        else:
            ball_pos = np.array([pos_x + 30, 762, 40])
        return skeleton, ball_pos

    @staticmethod
    def rebound(t):
        """Max vertical extension + Ball capture at apex."""
        skeleton = np.zeros((17, 3))
        pos_x, pos_y = 160.0, 762.0
        skeleton[:, :2] = [pos_x, pos_y]
        jump_z = 70 * np.sin(np.pi * t)
        skeleton[:, 2] = 100 + jump_z
        skeleton[9:11, 2] += 100 * np.sin(np.pi * t * 0.8)
        ball_pos = np.array([pos_x, pos_y, 305]) * (1-t) + np.mean(skeleton[9:11], axis=0) * t
        return skeleton, ball_pos

    @staticmethod
    def block(t):
        """Lateral jump + swat at high-Z ball."""
        skeleton = np.zeros((17, 3))
        pos_x = 160.0 + 50 * t
        skeleton[:, :2] = [pos_x, 762.0]
        jump_z = 60 * np.sin(np.pi * t)
        skeleton[:, 2] = 100 + jump_z
        skeleton[10] = [pos_x + 20, 770, 150 + 100 * t]
        ball_pos = np.array([pos_x + 10, 762, 200 + 100 * t])
        return skeleton, ball_pos

    @staticmethod
    def steal(t):
        """Lunging torso + Low hand reach for ball."""
        skeleton = np.zeros((17, 3))
        pos_y = 500.0 + 100 * t
        skeleton[:, :2] = [800.0, pos_y]
        skeleton[:, 2] = 100
        skeleton[10] = [820, pos_y + 40, 40]
        ball_pos = np.array([825, pos_y + 45, 40])
        return skeleton, ball_pos

def compute_features_v2(skel_2d_seq, skel_3d_seq, ball_3d_seq):
    """
    Computes the frozen D=72 feature tensor using GROUND TRUTH 3D context.
    """
    T = skel_2d_seq.shape[0]
    features = []
    
    for t in range(T):
        # 1. Local Pose (34) - Normalized to box
        pose = skel_2d_seq[t].flatten()
        
        # 2. Temporal (34) - Velocity
        if t > 0:
            velocity = (skel_2d_seq[t] - skel_2d_seq[t-1]).flatten() * 0.1
        else:
            velocity = np.zeros(34)
            
        # 3. Interaction (2) - Ball to Wrist DISTANCE in CM (Scaled)
        dist_l = np.linalg.norm(skel_3d_seq[t, 9] - ball_3d_seq[t]) * 0.01
        dist_r = np.linalg.norm(skel_3d_seq[t, 10] - ball_3d_seq[t]) * 0.01
        
        # 4. Global (2) - Court Position in CM (Scaled)
        court_pos = np.mean(skel_3d_seq[t, [11, 12], :2], axis=0) * 0.001
        
        row = np.concatenate([pose, velocity, [dist_l, dist_r], court_pos])
        features.append(row.tolist())
        
    return features

def project_to_2d(skeleton_3d_seq, K, R, t_vec):
    skeleton_2d = np.zeros((skeleton_3d_seq.shape[0], 17, 2))
    extrinsic = np.hstack((R, t_vec.reshape(3, 1)))
    for i in range(skeleton_3d_seq.shape[0]):
        for j in range(17):
            X_w = np.append(skeleton_3d_seq[i, j], 1.0)
            x_cam = extrinsic @ X_w
            x_pix_h = K @ x_cam
            skeleton_2d[i, j] = x_pix_h[:2] / x_pix_h[2]
    return skeleton_2d

def run_generator(output_file, num_samples=20):
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    R = np.eye(3)
    t_vec = np.array([0, -600, -250])
    
    moves = [
        ("jump_shot", MoveLibrary.jump_shot),
        ("crossover", MoveLibrary.crossover),
        ("rebound", MoveLibrary.rebound),
        ("block", MoveLibrary.block),
        ("steal", MoveLibrary.steal)
    ]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for name, func in moves:
            for _ in range(num_samples):
                data_3d = [func(t) for t in np.linspace(0, 1, 30)]
                skel_3d = np.array([d[0] for d in data_3d])
                ball_3d = np.array([d[1] for d in data_3d])
                
                skel_2d = project_to_2d(skel_3d, K, R, t_vec)
                
                # Normalize 2D to box
                skel_2d_norm = skel_2d.copy()
                for t in range(30):
                    bbox = [skel_2d[t,:,0].min(), skel_2d[t,:,1].min(), skel_2d[t,:,0].max(), skel_2d[t,:,1].max()]
                    w, h = bbox[2]-bbox[0]+1e-6, bbox[3]-bbox[1]+1e-6
                    skel_2d_norm[t,:,0] = (skel_2d[t,:,0] - bbox[0]) / w
                    skel_2d_norm[t,:,1] = (skel_2d[t,:,1] - bbox[1]) / h
                
                feat_v2 = compute_features_v2(skel_2d_norm, skel_3d, ball_3d)
                
                f.write(json.dumps({
                    "label": name,
                    "schema_version": "2.0.0",
                    "features_v2": feat_v2
                }) + "\n")
    print(f"[INFO] Generated ground-truth features to {output_file}")

if __name__ == "__main__":
    run_generator("data/training/synthetic_dataset_v2.jsonl")
