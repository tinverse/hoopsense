(use-modules (guix profiles))

(specifications->manifest
 '("rust"
   "rust:cargo"
   "python"
   "python-pip"
   "python-virtualenv"
   "python-numpy"
   "python-ipython"
   "opencv"
   "ffmpeg"
   "pkg-config"
   "openssl"
   "git"
   "coreutils"
   "findutils"
   "gcc-toolchain"
   "nss-certs"))
