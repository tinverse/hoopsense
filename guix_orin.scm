(use-modules (guix profiles)
             (guix packages)
             (guix download)
             (guix build-system python))

;; Optimized Manifest for NVIDIA Orin (aarch64)
(specifications->manifest
 '("rust"
   "rust:cargo"
   "bash"
   "coreutils"
   "python"
   "python-pip"
   "python-numpy"
   "opencv"
   "python-huggingface-hub"
   "ffmpeg"
   "zlib"
   "libsm"
   "libxext"
   "libice"
   "mesa"
   "glib"
   "pkg-config"
   "openssl"
   "git"
   "gcc-toolchain"
   "make"))
