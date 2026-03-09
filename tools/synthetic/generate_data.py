import numpy as np
import json
import os

def generate_synthetic_jump_shot(num_frames=30):
    """
    Procedurally generates a 3D skeletal sequence of a jump shot.
    Returns: (T, 17, 3) array in world centimeters.
    """
    t = np.linspace(0, 1, num_frames)
    skeleton = np.zeros((num_frames, 17, 3))
    
    # 1. Base stance (X, Y, Z)
    # Joints: 0:head, 5:L_sho, 6:R_sho, 11:L_hip, 12:R_sho, 15:L_ank, 16:R_ank...
    base_pose = np.zeros((17, 3))
    base_pose[0] = [0, 0, 180] # Head
    base_pose[5:7] = [[-20, 0, 150], [20, 0, 150]] # Shoulders
    base_pose[11:13] = [[-15, 0, 100], [15, 0, 100]] # Hips
    base_pose[15:17] = [[-15, 0, 0], [15, 0, 0]] # Ankles
    
    # 2. Add Jump Motion (Z-axis)
    jump_height = 50 * np.sin(np.pi * t) # Parabolic jump
    for i in range(num_frames):
        skeleton[i] = base_pose.copy()
        skeleton[i, :, 2] += jump_height[i]
        
        # 3. Add Arm Motion (Raising wrists for shot)
        arm_reach = 60 * t[i]
        skeleton[i, 9:11, 2] += arm_reach # Wrists go up
        skeleton[i, 9:11, 1] += 10 # Wrists move forward
        
    return skeleton

def project_to_2d(skeleton_3d, K, R, t_vec):
    """
    Projects a 3D skeleton into 2D pixel coordinates.
    x_p = K * [R|t] * X_w
    """
    num_frames = skeleton_3d.shape[0]
    skeleton_2d = np.zeros((num_frames, 17, 2))
    
    # Form extrinsic matrix [R|t]
    extrinsic = np.hstack((R, t_vec.reshape(3, 1)))
    
    for i in range(num_frames):
        for j in range(17):
            X_w = np.append(skeleton_3d[i, j], 1.0)
            x_cam = extrinsic @ X_w
            x_pix_h = K @ x_cam
            skeleton_2d[i, j] = x_pix_h[:2] / x_pix_h[2]
            
    return skeleton_2d

def run_generator(output_file, num_samples=10):
    # Virtual Camera (1080p)
    K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])
    
    # Virtual Pose (Camera 5m away, 2m high)
    R = np.eye(3) 
    t_vec = np.array([0, -500, -200]) # World is (X, Z) floor, Y up in this coord sys? 
    # Let's adjust to our Rust system: X, Y is floor, Z is height.
    # Cam at X=0, Y=-500 (sideline), Z=200 (height)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for s in range(num_samples):
            # Generate 3D
            skel_3d = generate_synthetic_jump_shot()
            
            # Project to 2D (Simplified projection for prototype)
            # In real version, we use the actual R and t
            skel_2d = project_to_2d(skel_3d, K, R, t_vec)
            
            # Add label and save
            sample = {
                "sample_id": s,
                "label": "jump_shot",
                "keypoints_sequence": skel_2d.tolist()
            }
            f.write(json.dumps(sample) + "\n")
            
    print(f"Generated {num_samples} synthetic samples to {output_file}")

if __name__ == "__main__":
    run_generator("data/training/synthetic_shots.jsonl")
