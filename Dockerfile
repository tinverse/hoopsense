# HoopSense Dockerfile: Optimized for GCP T4 / Orin Training
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 1. Install System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    ffmpeg \
    pkg-config \
    libssl-dev \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust Toolchain (for the Vision Core)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Setup Project Structure
WORKDIR /app
COPY . .

# 4. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 4b. Add DINOv3 for future bootstrap segmentation research in the cloud path.
# Keep this out of the Orin runtime story for now.
ARG DINOV3_REF=main
ENV HOOPSENSE_DINOV3_DIR=/opt/dinov3
ENV TORCH_HOME=/app/data/models/torch_hub
RUN git clone --depth 1 --branch "${DINOV3_REF}" https://github.com/facebookresearch/dinov3.git "${HOOPSENSE_DINOV3_DIR}" \
    && pip install --no-cache-dir --no-deps -e "${HOOPSENSE_DINOV3_DIR}" \
    && python3 -c "import dinov3; print('DINOv3 environment ready')"

# 5. Build Rust Core (Ensure performance binaries are ready)
RUN cd core && cargo build --release

# 6. Setup Cloud Environment
RUN chmod +x tools/infra/cloud_train_wrapper.sh
ENV PYTHONPATH="/app:${HOOPSENSE_DINOV3_DIR}:${PYTHONPATH}"

# Set Default Entrypoint to the Cloud Wrapper
# This allows Vertex AI to just pass training arguments
ENTRYPOINT ["/app/tools/infra/cloud_train_wrapper.sh"]
