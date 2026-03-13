import os

# Define the full HoopSense OS project structure with complete source code
project_structure = {
    "hoopsense": {
        "README.md": """# HoopSense OS
Practical game understanding from basketball video.

## Project Structure
- `core/`: Native perception and logic internals.
- `pipelines/`: Deterministic execution stages.
- `docs/`: Design-first documentation (Requirements, Architecture, Book).
- `cli/`: Local orchestration and Gemini Agent tools.
- `data/`: Local storage for training manifests and video links.
- `TASK_STATUS.md`: Current execution frontier.
""",
        "requirements.txt": """opencv-python-headless
numpy
pyyaml
torch
torchvision
ultralytics
easyocr
flask
""",
        "hoops_config.yaml": """# HoopSense Execution Config
project_name: "HoopSense MVP"
video_source: "local"
local_video_path: "data/sample.mp4"
output_dir: "data/outputs"

perception:
  model: "yolov8n-pose.pt"
  ocr_buffer_size: 30
  conf_threshold: 0.5
""",
        "TASK_STATUS.md": """# HoopSense Task Status
Current Frontier: Integration of Action Brain with refined court geometry.
""",
        "core": {
            "Cargo.toml": """[package]
name = "hoopsense-core"
version = "0.1.0"
edition = "2021"

[dependencies]
nalgebra = "0.32"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
anyhow = "1.0"
""",
            "src": {
                "lib.rs": """pub mod spatial;
pub mod ledger;
pub mod rules;
pub mod physics;

pub use spatial::SpatialResolver;
pub use ledger::GameStateLedger;
pub use rules::GeometricReferee;
pub use physics::Trajectory;
""",
                "spatial.rs": "// Spatial Resolver logic\n",
                "ledger.rs": "// Game State Ledger logic\n",
                "rules.rs": "// NCAA Rule logic\n",
                "physics.rs": "// Trajectory physics logic\n"
            }
        }
    }
}


def build_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            build_structure(path, content)
        else:
            with open(path, "w") as f:
                f.write(content)


def main():
    target = os.path.expanduser("~/hoopsense")
    os.makedirs(target, exist_ok=True)
    build_structure(target, project_structure["hoopsense"])
    print(f"\n✅ HoopSense OS Initialized in {target}")


if __name__ == "__main__":
    main()
