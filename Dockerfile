# HoopSense Dockerfile: Unified ML & Vision Environment
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    libssl-dev \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust Toolchain (for the Vision Core)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Setup Project Structure
WORKDIR /app
COPY . .

# 4. Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Build Rust Core
RUN cd core && cargo build --release

# Set Entrypoint
ENTRYPOINT ["python3"]
