import numpy as np
import json
import os

class MoveLibrary:
    @staticmethod
    def jump_shot(t):
        """Rising Z, Wrists above head + Ball at wrists."""
        skeleton = np.zeros((17, 3))
        skeleton[0] = [0, 0, 180] 
        skeleton[11:13] = [[985, 762, 100], [1015, 762, 100]]
        jump_z = 50 * np.sin(np.pi * t)
        skeleton[:, 2] += jump_z
        skeleton[9:11, 2] += 60 * t
        ball_pos = np.array([1000, 780, 140 + 160 * t + jump_z])
        return skeleton, ball_pos

    @staticmethod
    def crossover(t):
        """Lateral X movement + Ball switching hands."""
        skeleton = np.zeros((17, 3))
        skeleton[0] = [0, 0, 170]
        pos_x = 1400 + 100 * t
        skeleton[:, 0] = pos_x
        skeleton[:, 2] = 100 
        if t < 0.5:
            ball_pos = np.array([pos_x - 30, 762, 40])
        else:
            ball_pos = np.array([pos_x + 30, 762, 40])
        return skeleton, ball_pos

    @staticmethod
    def layup(t):
        """High-speed drive + one-handed release near rim."""
        skeleton = np.zeros((17, 3))
        pos_y = 600 + 150 * t
        skeleton[:, :2] = [160, pos_y]
        skeleton[:, 2] = 100 + 30 * np.sin(np.pi * t)
        ball_pos = np.array([160, pos_y + 10, 150 + 50 * t])
        return skeleton, ball_pos

    @staticmethod
    def dunk(t):
        """Hand keypoints intersecting rim plane at high velocity."""
        skeleton = np.zeros((17, 3))
        pos_y = 700 + 62 * t
        skeleton[:, :2] = [160, pos_y]
        jump_z = 120 * np.sin(np.pi * t)
        skeleton[:, 2] = 100 + jump_z
        skeleton[10] = [160, pos_y + 5, 200 + 150 * t]
        ball_pos = skeleton[10].copy()
        return skeleton, ball_pos

    @staticmethod
    def pass_chest(t):
        """Forward arm extension from chest."""
        skeleton = np.zeros((17, 3))
        pos_x, pos_y = 1000, 1000
        skeleton[:, :2] = [pos_x, pos_y]
        skeleton[:, 2] = 100
        ext = 50 * t
        skeleton[9:11] = [[pos_x-10, pos_y+ext, 130], [pos_x+10, pos_y+ext, 130]]
        ball_pos = np.array([pos_x, pos_y + ext + 10, 130])
        return skeleton, ball_pos

    @staticmethod
    def rebound(t):
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
        skeleton = np.zeros((17, 3))
        pos_y = 500.0 + 100 * t
        skeleton[:, :2] = [800.0, pos_y]
        skeleton[:, 2] = 100
        skeleton[10] = [820, pos_y + 40, 40]
        ball_pos = np.array([825, pos_y + 45, 40])
        return skeleton, ball_pos

    @staticmethod
    def euro_step(t, go_left=True):
        skeleton = np.zeros((17, 3))
        skeleton[0] = [0, 0, 180]
        pos_y = 500 + 100 * t
        direction = -1 if go_left else 1
        if t < 0.5:
            pos_x = 762 + (direction * 40 * np.sin(np.pi * t))
        else:
            pos_x = 762 + (direction * -40 * np.sin(np.pi * (t - 0.5)))
        skeleton[:, :2] = [pos_x, pos_y]
        skeleton[:, 2] = 100
        ball_pos = np.array([pos_x, pos_y + 20, 120])
        return skeleton, ball_pos

    @staticmethod
    def background_player(t, move_func, offset_x=2865):
        """Simulates an active player on an adjacent court."""
        skeleton, ball_pos = move_func(t)
        skeleton[:, 0] += offset_x
        if ball_pos is not None: ball_pos[0] += offset_x
        return skeleton, ball_pos

def compute_features_v2(skel_2d_norm, skel_3d_seq, ball_3d_seq):
    T = skel_2d_norm.shape[0]
    features = []
    for t in range(T):
        pose = skel_2d_norm[t].flatten()
        velocity = (skel_2d_norm[t] - skel_2d_norm[max(0, t-1)]).flatten() * 0.1
        if ball_3d_seq[t] is not None:
            dist_l = np.linalg.norm(skel_3d_seq[t, 9] - ball_3d_seq[t]) * 0.01
            dist_r = np.linalg.norm(skel_3d_seq[t, 10] - ball_3d_seq[t]) * 0.01
        else:
            dist_l, dist_r = 10.0, 10.0 
        court_pos = np.mean(skel_3d_seq[t, [11, 12], :2], axis=0) * 0.001
        row = np.concatenate([pose, velocity, [dist_l, dist_r], court_pos])
        features.append(row.tolist())
    return features

def get_look_at_matrix(cam_pos, target_pos):
    forward = target_pos - cam_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(np.array([0, 0, 1]), forward)
    if np.linalg.norm(right) < 1e-6: right = np.array([1, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.vstack([right, up, forward])

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
    
    moves = [
        ("jump_shot", MoveLibrary.jump_shot),
        ("crossover", MoveLibrary.crossover),
        ("layup", MoveLibrary.layup),
        ("dunk", MoveLibrary.dunk),
        ("pass_chest", MoveLibrary.pass_chest),
        ("rebound", MoveLibrary.rebound),
        ("block", MoveLibrary.block),
        ("steal", MoveLibrary.steal),
        ("euro_step_left", lambda t: MoveLibrary.euro_step(t, go_left=True)),
        ("euro_step_right", lambda t: MoveLibrary.euro_step(t, go_left=False)),
        ("idle_bystander", lambda t: MoveLibrary.background_player(t, MoveLibrary.jump_shot)) # Uses jump_shot kinematic but offset
    ]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for name, func in moves:
            for _ in range(num_samples):
                dist = np.random.uniform(400, 1200)
                height = np.random.uniform(150, 500)
                angle = np.random.uniform(-np.pi/4, np.pi/4)
                cam_pos = np.array([dist * np.sin(angle), -dist * np.cos(angle), height])
                target_pos = np.array([0.0, 0.0, 100.0])
                R = get_look_at_matrix(cam_pos, target_pos)
                t_vec = -R @ cam_pos
                
                data_3d = [func(t) for t in np.linspace(0, 1, 30)]
                skel_3d = np.array([d[0] for d in data_3d])
                ball_3d = np.array([d[1] for d in data_3d])
                skel_2d = project_to_2d(skel_3d, K, R, t_vec)
                skel_2d_norm = skel_2d.copy()
                for t in range(30):
                    bbox = [skel_2d[t,:,0].min(), skel_2d[t,:,1].min(), skel_2d[t,:,0].max(), skel_2d[t,:,1].max()]
                    w, h = bbox[2]-bbox[0]+1e-6, bbox[3]-bbox[1]+1e-6
                    skel_2d_norm[t,:,0] = (skel_2d[t,:,0] - bbox[0]) / w
                    skel_2d_norm[t,:,1] = (skel_2d[t,:,1] - bbox[1]) / h
                feat_v2 = compute_features_v2(skel_2d_norm, skel_3d, ball_3d)
                f.write(json.dumps({"label": name, "schema_version": "2.0.0", "features_v2": feat_v2}) + "\n")
    print(f"[INFO] Generated features with Background Player Noise to {output_file}")

if __name__ == "__main__":
    run_generator("data/training/synthetic_dataset_v2.jsonl")
