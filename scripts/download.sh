#!/usr/bin/env sh


set -euo pipefail


asset_dir="./assets"

[ ! -e "${asset_dir}"/models_smplx_v1_1.zip ] \
    && echo Error: Download SMPL-X body model from https://smpl-x.is.tue.mpg.de \
    and save zip archive to "${asset_dir}" \
    && exit 1 \
    && :

asset_urls=(
    # Download constants (SPIN)
    http://visiondata.cis.upenn.edu/spin/data.tar.gz

    # Download essentials (SMPLify-XMC)
    https://download.is.tue.mpg.de/tuch/smplify-xmc-essentials.zip

    # Download sketch2pose models
    http://www-labs.iro.umontreal.ca/~bmpix/sketch2pose/models.zip

    # Download test images
    http://www-labs.iro.umontreal.ca/~bmpix/sketch2pose/images.zip
)
for asset_url in "${asset_urls[@]}"; do
    wget \
        -nc \
        -c \
        --directory-prefix "${asset_dir}" \
        "${asset_url}"
done

models_dir="./models"
mkdir -p "${models_dir}"

model_files=(
    # Unzip smplx models
    models_smplx_v1_1.zip

    # Unzip essentials (SMPLifu-XMC)
    smplify-xmc-essentials.zip

    # Unzip sketch2pose models
    models.zip
)

for model_file in "${model_files[@]}"; do
    unzip \
        -u \
        -d "${models_dir}" \
        "${asset_dir}"/"${model_file}"
done

# Unzip constants (SPIN)
tar \
    --skip-old-files \
    -xvf "${asset_dir}"/data.tar.gz \
    -C "${models_dir}" \
    data/smpl_mean_params.npz

data_dir="./data"
mkdir -p "${data_dir}"

# Unzip test images
unzip \
    -u \
    -d "${data_dir}" \
    "${asset_dir}"/images.zip
