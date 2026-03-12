import numpy as np
import json
import os
import sys

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

def compute_features_v2(skel_2d_norm, skel_3d_seq, ball_3d_seq):
    T = skel_2d_norm.shape[0]
    features = []
    for t in range(T):
        pose = skel_2d_norm[t].flatten()
        velocity = (skel_2d_norm[t] - skel_2d_norm[max(0, t-1)]).flatten() * 0.1
        dist_l = np.linalg.norm(skel_3d_seq[t, 9] - ball_3d_seq[t]) * 0.01
        dist_r = np.linalg.norm(skel_3d_seq[t, 10] - ball_3d_seq[t]) * 0.01
        court_pos = np.mean(skel_3d_seq[t, [11, 12], :2], axis=0) * 0.001
        vel_t = (skel_2d_norm[t] - skel_2d_norm[max(0, t-1)]).flatten() * 0.1
        court_pos_t = np.mean(skel_3d_seq[t, [11, 12], :2], axis=0) * 0.001
        row = np.concatenate([pose, vel_t, [dist_l, dist_r], court_pos_t])
        features.append(row.tolist())
    return features

def get_look_at_matrix(cam_pos, target_pos):
    forward = target_pos - cam_pos
    forward /= (np.linalg.norm(forward) + 1e-6)
    right = np.cross(np.array([0, 0, 1]), forward)
    if np.linalg.norm(right) < 1e-6: right = np.array([1, 0, 0])
    right /= (np.linalg.norm(right) + 1e-6)
    up = np.cross(forward, right)
    return np.vstack([right, up, forward])

def project_to_2d(skeleton_3d_seq, K, R, t_vec, noise_std=0.0):
    skeleton_2d = np.zeros((skeleton_3d_seq.shape[0], 17, 2))
    extrinsic = np.hstack((R, t_vec.reshape(3, 1)))
    for i in range(skeleton_3d_seq.shape[0]):
        for j in range(17):
            X_w = np.append(skeleton_3d_seq[i, j], 1.0)
            x_cam = extrinsic @ X_w
            x_pix_h = K @ x_cam
            skeleton_2d[i, j] = x_pix_h[:2] / (x_pix_h[2] + 1e-6)
        if noise_std > 0.0:
            skeleton_2d[i] += np.random.normal(scale=noise_std, size=skeleton_2d[i].shape)
    return skeleton_2d


def run_oracle_generator(output_file, asf_path, amc_path, label="jump_shot"):
    try:
        from tools.synthetic.amc_oracle import generate_oracle_sample
        from tools.synthetic.amc_oracle import write_oracle_dataset
    except ImportError:
        from amc_oracle import generate_oracle_sample
        from amc_oracle import write_oracle_dataset

    sample = generate_oracle_sample(asf_path, amc_path, label)
    # Append to existing or create new
    mode = 'a' if os.path.exists(output_file) else 'w'
    with open(output_file, mode) as f:
        f.write(json.dumps(sample) + "\n")
    print(f"[INFO] Generated Oracle-backed features to {output_file}")

def run_multi_oracle_generator(output_file):
    mocap_base = "data/training/cmu_oracle"
    fixtures = [
        ("06.asf", "06_15.amc", "jump_shot"),
        ("06.asf", "06_11.amc", "crossover"),
        ("124.asf", "124_01.amc", "idle_bystander"),
        ("124.asf", "124_02.amc", "idle_bystander"),
        ("102.asf", "102_01.amc", "walk"), # Mapping walk to idle_bystander or similar
    ]
    
    for asf, amc, label in fixtures:
        asf_path = os.path.join(mocap_base, asf)
        amc_path = os.path.join(mocap_base, amc)
        if os.path.exists(asf_path) and os.path.exists(amc_path):
            # Mapping specific labels to ActionBrain categories
            target_label = label
            if label in ["walk", "idle"]:
                target_label = "idle_bystander"
            run_oracle_generator(output_file, asf_path, amc_path, label=target_label)
        else:
            print(f"[WARN] Skipping missing fixture: {asf}/{amc}")

def run_generator(output_file, num_samples=20):
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    R = np.eye(3)
    t_vec = np.array([0, -600, -250])
    
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
        ("euro_step_right", lambda t: MoveLibrary.euro_step(t, go_left=False))
    ]
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        # 1. Procedural Baseline
        for name, func in moves:
            for _ in range(num_samples):
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
    
    # 2. MoCap Oracle Augmentation
    run_multi_oracle_generator(output_file)
    
    print(f"[INFO] Generated enriched features to {output_file}")

if __name__ == "__main__":
    output_file = sys.argv[1] if len(sys.argv) > 1 else "data/training/synthetic_dataset_v2.jsonl"
    if "--multi-oracle" in sys.argv:
        run_multi_oracle_generator(output_file)
    elif "--oracle" in sys.argv:
        try:
            oracle_idx = sys.argv.index("--oracle")
            asf_path = sys.argv[oracle_idx + 1]
            amc_path = sys.argv[oracle_idx + 2]
            label = sys.argv[oracle_idx + 3] if len(sys.argv) > oracle_idx + 3 else "jump_shot"
            run_oracle_generator(output_file, asf_path, amc_path, label=label)
        except IndexError:
            print("Usage: generate_data.py --oracle <asf> <amc> [label]")
    else:
        run_generator(output_file)
