# Sketch2Pose: Estimating a 3D Character Pose from a Bitmap Sketch

Artists frequently capture character poses via raster sketches, then use these
drawings as a reference while posing a 3D character in a specialized 3D
software --- a time-consuming process, requiring specialized 3D training and
mental effort. We tackle this challenge by proposing the first system for
automatically inferring a 3D character pose from a single bitmap sketch,
producing poses consistent with viewer expectations. Algorithmically
interpreting bitmap sketches is challenging, as they contain significantly
distorted proportions and foreshortening. We address this by predicting three
key elements of a drawing, necessary to disambiguate the drawn poses: 2D bone
tangents, self-contacts, and bone foreshortening. These elements are then
leveraged in an optimization inferring the 3D character pose consistent with
the artist's intent. Our optimization balances cues derived from artistic
literature and perception research to compensate for distorted character
proportions. We demonstrate a gallery of results on sketches of numerous
styles. We validate our method via numerical evaluations, user studies, and
comparisons to manually posed characters and previous work.

[Project Page](http://www-labs.iro.umontreal.ca/~bmpix/sketch2pose/)

# Prerequisites

- [GNU/Linux](https://www.gnu.org/gnu/linux-and-gnu.en.html)
- [`python`](https://python.org)
- [`pytorch`](https://pytorch.org/)
- [NVIDIA GPU] (optional, but highly recommended)

## Download body model (SMPL-X)

Download SMPL-X body model from
[https://smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de)

See [`download.sh`](./scripts/download.sh) and run

```bash
sh ./scripts/download.sh
```

## Virtual environement

Change [`pytorch`](https://pytorch.org/) version if needed in
[`prepare.sh`](./scripts/prepare.sh) and run

```bash
sh ./scripts/prepare.sh
```

# Demo

Activate virtual environement `. venv/bin/activate` and run

```bash
sh ./scripts/run.sh

# or

python src/pose.py \
        --save-path "${out_dir}" \
        --img-path "${img_path}" \
        --use-contacts \
        --use-natural \
        --use-cos \
        --use-angle-transf \

# or without contacts

python src/pose.py \
        --save-path "${out_dir}" \
        --img-path "${img_path}" \
        --use-natural \
        --use-cos \
        --use-angle-transf \
```

# Citation

```
@article{brodt2022sketch2pose,
    author = {Kirill Brodt and Mikhail Bessmeltsev},
    title = {Sketch2Pose: Estimating a 3D Character Pose from a Bitmap Sketch},
    journal = {ACM Transactions on Graphics},
    year = {2022},
    month = {7},
    volume = {41},
    number = {4},
    doi = {10.1145/3528223.3530106},
}
```

# Useful links

- [Deep High-Resolution Representation Learning for Human Pose Estimation](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/)
- [SMPLify-X](https://github.com/vchoutas/smplify-x) ([project](https://smpl-x.is.tue.mpg.de/))
- [SPIN](https://github.com/nkolot/SPIN) ([project](https://www.seas.upenn.edu/~nkolot/projects/spin/))
- [eft](https://github.com/facebookresearch/eft)
- [SMPLify-XMC](https://github.com/muelea/smplify-xmc), [selfcontact](https://github.com/muelea/selfcontact) ([project](https://tuch.is.tue.mpg.de/))
- [Mixamo](https://www.mixamo.com) models with animations and a
  [script](https://forums.unrealengine.com/community/community-content-tools-and-tutorials/1376068-script-mixamo-download-script)
  to download them
- Quaternion-based [Forward
  Kinematics](https://github.com/facebookresearch/QuaterNet)
