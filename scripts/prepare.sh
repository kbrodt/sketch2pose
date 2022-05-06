#!/usr/bin/env sh


set -euo pipefail

venv_dir=venv
python -m venv --clear "${venv_dir}"

. "${venv_dir}"/bin/activate

pip install -U pip setuptools

extra="cpu"
[ -x "$(command -v nvcc)" ] && extra="cu113"
pip install \
    torch \
    torchvision \
    --extra-index-url https://download.pytorch.org/whl/"${extra}"

pip install -r requirements.txt

v=$(python -c 'import sys; v = sys.version_info; print(f"{v.major}.{v.minor}")')
for p in patches/*.diff; do
    patch -p0 < <(sed "s/python3.10/python${v}/" "${p}")
done
