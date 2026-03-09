(use-modules (guix packages)
             (gnu packages rust)
             (gnu packages python)
             (gnu packages python-xyz)
             (gnu packages python-science)
             (gnu packages video)
             (gnu packages pkg-config)
             (gnu packages tls)
             (gnu packages base)
             (gnu packages commencement))

(packages->manifest
 (list rust
       rust:cargo
       python
       python-pip
       python-virtualenv
       ffmpeg
       pkg-config
       openssl
       git
       coreutils
       findutils
       gcc-toolchain))
