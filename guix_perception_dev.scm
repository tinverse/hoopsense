(use-modules (guix profiles))

(specifications->manifest
 '("bash"
   "rust"
   "rust:cargo"
   "pkg-config"
   "openssl"
   "gcc-toolchain"
   "coreutils"
   "nss-certs"
   "python"
   "python-numpy"
   "python-pyyaml"
   "zlib"))
