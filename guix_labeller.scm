(use-modules (guix profiles))

;; Specialized Manifest for Geometry Calibration and Manual Labelling
;; Optimized for speed by excluding heavy ML libraries (PyTorch)
(specifications->manifest
 '("bash"
   "python"
   "python-pip"
   "python-numpy"
   "python-flask"
   "opencv"
   "ffmpeg"
   "pkg-config"
   "openssl"
   "git"
   "coreutils"
   "nss-certs"))
