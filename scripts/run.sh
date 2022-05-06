#!/usr/bin/env sh


set -euo pipefail


img_dir="./data/images"
out_dir="./output"

find "${img_dir}" -mindepth 1 -maxdepth 1 -type f -print0 \
    | xargs -0 -I "{}" python src/pose.py \
        --save-path "${out_dir}" \
        --img-path "{}" \
        --use-contacts \
        --use-natural \
        --use-cos \
        --use-angle-transf \

exit

# baseline (SMPLify-XMC)

find "${img_dir}" -mindepth 1 -maxdepth 1 -type f -print0 \
    | xargs -0 -I "{}" python src/pose.py \
        --save-path "${out_dir}_baseline" \
        --img-path "{}" \
        --c-mse 1 \
        --c-par 0 \
        --use-contacts \
        --use-cos \
        --use-angle-transf \

# ablation

find "${img_dir}" -mindepth 1 -maxdepth 1 -type f -print0 \
    | xargs -0 -I "{}" python src/pose.py \
        --save-path "${out_dir}_wocostransform" \
        --img-path "{}" \
        --use-contacts \
        --use-natural \
        --use-cos \


find "${img_dir}" -mindepth 1 -maxdepth 1 -type f -print0 \
    | xargs -0 -I "{}" python src/pose.py \
        --save-path "${out_dir}_wocontacts" \
        --img-path "{}" \
        --use-msc \
        --use-natural \
        --use-cos \
        --use-angle-transf \


find "${img_dir}" -mindepth 1 -maxdepth 1 -type f -print0 \
    | xargs -0 -I "{}" python src/pose.py \
        --save-path "${out_dir}_wonatural" \
        --img-path "{}" \
        --use-contacts \
        --use-cos \
        --use-angle-transf \
