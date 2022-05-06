import math

import cv2
import numpy as np

IMG_SIZE = (288, 384)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

KPS = (
    "Head",
    "Neck",
    "Right Shoulder",
    "Right Arm",
    "Right Hand",
    "Left Shoulder",
    "Left Arm",
    "Left Hand",
    "Spine",
    "Hips",
    "Right Upper Leg",
    "Right Leg",
    "Right Foot",
    "Left Upper Leg",
    "Left Leg",
    "Left Foot",
    "Left Toe",
    "Right Toe",
)

SKELETON = (
    (0, 1),
    (1, 8),
    (8, 9),
    (9, 10),
    (9, 13),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (15, 16),
    (12, 17),
)


OPENPOSE_TO_GESTURE = (
    0,  # 0 Head\n",
    1,  #   Neck\n",
    2,  # 2 Right Shoulder\n",
    3,  #   Right Arm\n",
    4,  # 4 Right Hand\n",
    5,  #   Left Shoulder\n",
    6,  # 6 Left Arm\n",
    7,  #   Left Hand\n",
    9,  # 8 Hips\n",
    10,  #   Right Upper Leg\n",
    11,  # 10Right Leg\n",
    12,  #   Right Foot\n",
    13,  # 12Left Upper Leg\n",
    14,  #   Left Leg\n",
    15,  # 14Left Foot\n",
    -1,  # \n",
    -1,  # 16\n",
    -1,  # \n",
    -1,  # 18\n",
    16,  #   Left Toe\n",
    -1,  # 20\n",
    -1,  # \n",
    17,  # 22Right Toe\n",
    -1,  # \n",
    -1,  # 24\n",
)


def transform(img):
    img = img.astype("float32") / 255

    img = (img - MEAN) / STD

    return np.transpose(img, axes=(2, 0, 1))


def get_affine_transform(
    center,
    scale,
    rot,
    output_size,
    shift=np.array([0, 0], dtype=np.float32),
    inv=0,
    pixel_std=200,
):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * pixel_std
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def process_image(path, input_img_size, pixel_std=200):
    data_numpy = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    # BUG HERE. Must be uncommented
    # data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    h, w = data_numpy.shape[:2]
    c = np.array([w / 2, h / 2], dtype=np.float32)

    aspect_ratio = input_img_size[0] / input_img_size[1]
    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    s = np.array([w / pixel_std, h / pixel_std], dtype=np.float32) * 1.25
    r = 0
    trans = get_affine_transform(c, s, r, input_img_size, pixel_std=pixel_std)
    input = cv2.warpAffine(data_numpy, trans, input_img_size, flags=cv2.INTER_LINEAR)

    input = transform(input)

    return input, data_numpy, c, s


def get_final_preds(batch_heatmaps, center, scale, post_process=False):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if post_process:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                    diff = np.array(
                        [
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px],
                        ]
                    )
                    coords[n][p] += np.sign(diff) * 0.25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_max_preds(batch_heatmaps):
    """
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    """
    assert isinstance(
        batch_heatmaps, np.ndarray
    ), "batch_heatmaps should be numpy.ndarray"
    assert batch_heatmaps.ndim == 4, "batch_images should be 4-ndim"

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def infer_single_image(model, img_path, input_img_size=(288, 384), return_kps=True):
    img_path = str(img_path)
    pose_input, img, center, scale = process_image(
        img_path, input_img_size=input_img_size
    )
    model.setInput(pose_input[None])
    predicted_heatmap = model.forward()

    if not return_kps:
        return predicted_heatmap.squeeze(0)

    predicted_keypoints, confidence = get_final_preds(
        predicted_heatmap, center[None], scale[None], post_process=True
    )

    (predicted_keypoints, confidence, predicted_heatmap,) = (
        predicted_keypoints.squeeze(0),
        confidence.squeeze(0),
        predicted_heatmap.squeeze(0),
    )

    return img, predicted_keypoints, confidence, predicted_heatmap
