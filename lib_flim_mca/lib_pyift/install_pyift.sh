#!/bin/bash
whl_path=pyift-0.1-cp311-cp311-linux_x86_64.whl
python3.11 -m pip install numpy==2.0.1
python3.11 -m pip install $whl_path

# Get path to site-packages
site_packages_path=$(python3.11 -c "import site; print(site.getusersitepackages())")
echo "Site-packages path: $site_packages_path"

abs_path=$(realpath "pyift/")

ln -sf "$abs_path"/_pyift.*.so "$site_packages_path/pyift/"
echo "PyIFT installation completed!"