import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import PIL.Image as Image
import selfcontact
import selfcontact.losses
import shapely.geometry
import torch
import torch.nn as nn
import torch.optim as optim
import torchgeometry
import tqdm
import trimesh
from skimage import measure
from torch.utils.tensorboard.writer import SummaryWriter

import fist_pose
import hist_cub
import losses
import pose_estimation
import spin
import utils

PE_KSP_TO_SPIN = {
    "Head": "Head",
    "Neck": "Neck",
    "Right Shoulder": "Right ForeArm",
    "Right Arm": "Right Arm",
    "Right Hand": "Right Hand",
    "Left Shoulder": "Left ForeArm",
    "Left Arm": "Left Arm",
    "Left Hand": "Left Hand",
    "Spine": "Spine1",
    "Hips": "Hips",
    "Right Upper Leg": "Right Upper Leg",
    "Right Leg": "Right Leg",
    "Right Foot": "Right Foot",
    "Left Upper Leg": "Left Upper Leg",
    "Left Leg": "Left Leg",
    "Left Foot": "Left Foot",
    "Left Toe": "Left Toe",
    "Right Toe": "Right Toe",
}
MODELS_DIR = "models"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pose-estimation-model-path",
        type=str,
        default=f"./{MODELS_DIR}/hrn_w48_384x288.onnx",
        help="Pose Estimation model",
    )

    parser.add_argument(
        "--contact-model-path",
        type=str,
        default=f"./{MODELS_DIR}/contact_hrn_w32_256x192.onnx",
        help="Contact model",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Torch device",
    )

    parser.add_argument(
        "--spin-model-path",
        type=str,
        default=f"./{MODELS_DIR}/spin_model_smplx_eft_18.pt",
        help="SPIN model path",
    )

    parser.add_argument(
        "--smpl-type",
        type=str,
        default="smplx",
        choices=["smplx"],
        help="SMPL model type",
    )
    parser.add_argument(
        "--smpl-model-dir",
        type=str,
        default=f"./{MODELS_DIR}/models/smplx",
        help="SMPL model dir",
    )
    parser.add_argument(
        "--smpl-mean-params-path",
        type=str,
        default=f"./{MODELS_DIR}/data/smpl_mean_params.npz",
        help="SMPL mean params",
    )
    parser.add_argument(
        "--essentials-dir",
        type=str,
        default=f"./{MODELS_DIR}/smplify-xmc-essentials",
        help="SMPL Essentials folder for contacts",
    )

    parser.add_argument(
        "--parametrization-path",
        type=str,
        default=f"./{MODELS_DIR}/smplx_parametrization/parametrization.npy",
        help="Parametrization path",
    )
    parser.add_argument(
        "--bone-parametrization-path",
        type=str,
        default=f"./{MODELS_DIR}/smplx_parametrization/bone_to_param2.npy",
        help="Bone parametrization path",
    )
    parser.add_argument(
        "--foot-inds-path",
        type=str,
        default=f"./{MODELS_DIR}/smplx_parametrization/foot_inds.npy",
        help="Foot indinces",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        required=True,
        help="Path to save the results",
    )

    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="Path to img to test",
    )

    parser.add_argument(
        "--use-contacts",
        action="store_true",
        help="Use contact model",
    )
    parser.add_argument(
        "--use-msc",
        action="store_true",
        help="Use MSC loss",
    )
    parser.add_argument(
        "--use-natural",
        action="store_true",
        help="Use regularity",
    )
    parser.add_argument(
        "--use-cos",
        action="store_true",
        help="Use cos model",
    )
    parser.add_argument(
        "--use-angle-transf",
        action="store_true",
        help="Use cube foreshortening transformation",
    )

    parser.add_argument(
        "--c-mse",
        type=float,
        default=0,
        help="MSE weight",
    )
    parser.add_argument(
        "--c-par",
        type=float,
        default=10,
        help="Parallel weight",
    )

    parser.add_argument(
        "--c-f",
        type=float,
        default=1000,
        help="Cos coef",
    )
    parser.add_argument(
        "--c-parallel",
        type=float,
        default=100,
        help="Parallel weight",
    )
    parser.add_argument(
        "--c-reg",
        type=float,
        default=1000,
        help="Regularity weight",
    )
    parser.add_argument(
        "--c-cont2d",
        type=float,
        default=1,
        help="Contact 2D weight",
    )
    parser.add_argument(
        "--c-msc",
        type=float,
        default=17_500,
        help="MSC weight",
    )

    parser.add_argument(
        "--fist",
        nargs="+",
        type=str,
        choices=list(fist_pose.INT_TO_FIST),
    )

    args = parser.parse_args()

    return args


def freeze_layers(model):
    for module in model.modules():
        if type(module) is False:
            continue

        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for m in module.parameters():
                m.requires_grad = False

        if isinstance(module, nn.Dropout):
            module.eval()
            for m in module.parameters():
                m.requires_grad = False


def project_and_normalize_to_spin(vertices_3d, camera):
    vertices_2d = vertices_3d  # [:, :2]

    scale, translate = camera[0], camera[1:]
    translate = scale.new_zeros(3)
    translate[:2] = camera[1:]

    vertices_2d = vertices_2d + translate
    vertices_2d = scale * vertices_2d + 1
    vertices_2d = spin.constants.IMG_RES / 2 * vertices_2d

    return vertices_2d


def project_and_normalize_to_spin_legs(vertices_3d, A, camera):
    A, J = A
    A = A[0]
    J = J[0]
    L = vertices_3d.new_tensor(
        [
            [0.98619063, 0.16560926, 0.00127302],
            [-0.16560601, 0.98603675, 0.01749799],
            [0.00164258, -0.01746717, 0.99984609],
        ]
    )
    R = vertices_3d.new_tensor(
        [
            [0.9910211, -0.13368178, -0.0025208],
            [0.13367888, 0.99027076, 0.03864949],
            [-0.00267045, -0.03863944, 0.99924965],
        ]
    )
    scale = camera[0]
    R = A[2, :3, :3] @ R  # 2 - right
    L = A[1, :3, :3] @ L  # 1 - left
    r = J[5] - J[2]
    l = J[4] - J[1]

    rleg = scale * spin.constants.IMG_RES / 2 * R @ r
    lleg = scale * spin.constants.IMG_RES / 2 * L @ l

    rleg = rleg[:2]
    lleg = lleg[:2]

    return rleg, lleg


def rotation_matrix_to_angle_axis(rotmat):
    bs, n_joints, *_ = rotmat.size()
    rotmat = torch.cat(
        [
            rotmat.view(-1, 3, 3),
            rotmat.new_tensor([0, 0, 1], dtype=torch.float32)
            .view(bs, 3, 1)
            .expand(n_joints, -1, -1),
        ],
        dim=-1,
    )
    aa = torchgeometry.rotation_matrix_to_angle_axis(rotmat)
    aa = aa.reshape(bs, 3 * n_joints)

    return aa


def get_smpl_output(smpl, rotmat, betas, use_betas=True, zero_hands=False):
    if smpl.name() == "SMPL":
        smpl_output = smpl(
            betas=betas if use_betas else None,
            body_pose=rotmat[:, 1:],
            global_orient=rotmat[:, 0].unsqueeze(1),
            pose2rot=False,
        )
    elif smpl.name() == "SMPL-X":
        rotmat = rotation_matrix_to_angle_axis(rotmat)
        if zero_hands:
            for i in [20, 21]:
                rotmat[:, 3 * i : 3 * (i + 1)] = 0

            for i in [12, 15]:  # neck, head
                rotmat[:, 3 * i + 1] = 0  # y
        smpl_output = smpl(
            betas=betas if use_betas else None,
            body_pose=rotmat[:, 3:],
            global_orient=rotmat[:, :3],
            pose2rot=True,
        )
    else:
        raise NotImplementedError

    return smpl_output, rotmat


def get_predictions(model_hmr, smpl, input_img, use_betas=True, zero_hands=False):
    input_img = input_img.unsqueeze(0)
    rotmat, betas, camera = model_hmr(input_img)

    smpl_output, rotmat = get_smpl_output(
        smpl, rotmat, betas, use_betas=use_betas, zero_hands=zero_hands
    )

    rotmat = rotmat.squeeze(0)
    betas = betas.squeeze(0)
    camera = camera.squeeze(0)
    z = smpl_output.joints
    z = z.squeeze(0)

    return rotmat, betas, camera, smpl_output, z


def get_pred_and_data(
    model_hmr, smpl, selector, input_img, use_betas=True, zero_hands=False
):
    rotmat, betas, camera, smpl_output, zz = get_predictions(
        model_hmr, smpl, input_img, use_betas=use_betas, zero_hands=zero_hands
    )

    joints = smpl_output.joints.squeeze(0)
    joints_2d = project_and_normalize_to_spin(joints, camera)
    rleg, lleg = project_and_normalize_to_spin_legs(joints, smpl_output.A, camera)
    joints_2d_orig = joints_2d
    joints_2d = joints_2d[selector]

    vertices = smpl_output.vertices.squeeze(0)
    vertices_2d = project_and_normalize_to_spin(vertices, camera)

    zz = zz[selector]

    return (
        rotmat,
        betas,
        camera,
        joints_2d,
        zz,
        vertices_2d,
        smpl_output,
        (rleg, lleg),
        joints_2d_orig,
    )


def normalize_keypoints_to_spin(keypoints_2d, img_size):
    h, w = img_size
    if h > w:  # vertically
        ax1 = 1
        ax2 = 0
    else:  # horizontal
        ax1 = 0
        ax2 = 1

    shift = (img_size[ax1] - img_size[ax2]) / 2
    scale = spin.constants.IMG_RES / img_size[ax2]
    keypoints_2d_normalized = np.copy(keypoints_2d)
    keypoints_2d_normalized[:, ax2] -= shift
    keypoints_2d_normalized *= scale

    return keypoints_2d_normalized, shift, scale, ax2


def unnormalize_keypoints_from_spin(keypoints_2d, shift, scale, ax2):
    keypoints_2d_normalized = np.copy(keypoints_2d)
    keypoints_2d_normalized /= scale
    keypoints_2d_normalized[:, ax2] += shift

    return keypoints_2d_normalized


def get_vertices_in_heatmap(contact_heatmap):
    contact_heatmap_size = contact_heatmap.shape[:2]
    label = measure.label(contact_heatmap)

    y_data_conts = []
    for i in range(1, label.max() + 1):
        predicted_kps_contact = np.vstack(np.nonzero(label == i)[::-1]).T.astype(
            "float"
        )
        predicted_kps_contact_scaled, *_ = normalize_keypoints_to_spin(
            predicted_kps_contact, contact_heatmap_size
        )
        y_data_cont = torch.from_numpy(predicted_kps_contact_scaled).int().tolist()
        y_data_cont = shapely.geometry.MultiPoint(y_data_cont).convex_hull
        y_data_conts.append(y_data_cont)

    return y_data_conts


def get_contact_heatmap(model_contact, img_path, thresh=0.5):
    contact_heatmap = pose_estimation.infer_single_image(
        model_contact,
        img_path,
        input_img_size=(192, 256),
        return_kps=False,
    )
    contact_heatmap = contact_heatmap.squeeze(0)
    contact_heatmap_orig = contact_heatmap.copy()

    mi = contact_heatmap.min()
    ma = contact_heatmap.max()
    contact_heatmap = (contact_heatmap - mi) / (ma - mi)
    contact_heatmap_ = ((contact_heatmap > thresh) * 255).astype("uint8")

    contact_heatmap = np.repeat(contact_heatmap[..., None], repeats=3, axis=-1)
    contact_heatmap = (contact_heatmap * 255).astype("uint8")

    return contact_heatmap_, contact_heatmap, contact_heatmap_orig


def discretize(parametrization, n_bins=100):
    bins = np.linspace(0, 1, n_bins + 1)
    inds = np.digitize(parametrization, bins)
    disc_parametrization = bins[inds - 1]

    return disc_parametrization


def get_mapping_from_params_to_verts(verts, params):
    mapping = {}
    for v, t in zip(verts, params):
        mapping.setdefault(t, []).append(v)

    return mapping


def find_contacts(y_data_conts, keypoints_2d, bone_to_params, thresh=12, step=0.0072246375):
    n_bins = int(math.ceil(1 / step)) - 1  # mean face's circumradius
    contact = []
    contact_2d = []
    for_mask = []
    for y_data_cont in y_data_conts:
        contact_loc = []
        contact_2d_loc = []
        buffer = y_data_cont.buffer(thresh)
        mask_add = False
        for i, j in pose_estimation.SKELETON:
            verts, t3d = bone_to_params[(i, j)]
            if len(verts) == 0:
                continue

            t3d = discretize(t3d, n_bins=n_bins)
            t3d_to_verts = get_mapping_from_params_to_verts(verts, t3d)
            t3d_to_verts_sorted = sorted(t3d_to_verts.items(), key=lambda x: x[0])
            t3d_sorted_np = np.array([x for x, _ in t3d_to_verts_sorted])

            line = shapely.geometry.LineString([keypoints_2d[i], keypoints_2d[j]])
            lint = buffer.intersection(line)
            if len(lint.boundary.geoms) < 2:
                continue

            t2d_start = line.project(lint.boundary.geoms[0], normalized=True)
            t2d_end = line.project(lint.boundary.geoms[1], normalized=True)
            assert t2d_start <= t2d_end

            t2ds = discretize(
                np.linspace(t2d_start, t2d_end, n_bins + 1), n_bins=n_bins
            )
            to_add = False
            for t2d in t2ds:
                if t2d < t3d_sorted_np[0] or t2d > t3d_sorted_np[-1]:
                    continue

                t2d_ind = np.searchsorted(t3d_sorted_np, t2d)
                c = t3d_to_verts_sorted[t2d_ind][1]

                contact_loc.extend(c)
                to_add = True
                mask_add = True

                if t2d_ind + 1 < len(t3d_to_verts_sorted):
                    c = t3d_to_verts_sorted[t2d_ind + 1][1]
                    contact_loc.extend(c)

                if t2d_ind > 0:
                    c = t3d_to_verts_sorted[t2d_ind - 1][1]
                    contact_loc.extend(c)

            if to_add:
                contact_2d_loc.append((i, j, t2d_start + 0.5 * (t2d_end - t2d_start)))

        if mask_add:
            for_mask.append(buffer.exterior.coords.xy)

        contact_loc = sorted(set(contact_loc))
        contact_loc = np.array(contact_loc, dtype="int")
        contact.append(contact_loc)
        contact_2d.append(contact_2d_loc)

    for_mask = [np.stack((x, y), axis=0).T[:, None].astype("int") for x, y in for_mask]

    return contact, contact_2d, for_mask


def optimize(
    model_hmr,
    smpl,
    selector,
    input_img,
    keypoints_2d,
    optimizer,
    args,
    loss_mse=None,
    loss_parallel=None,
    c_mse=0.0,
    c_new_mse=1.0,
    c_beta=1e-3,
    sc_crit=None,
    msc_crit=None,
    contact=None,
    n_steps=60,
    save_path=None,
    writer=None,
    i_ini=0,
):
    to_save = False
    if save_path is not None:
        (
            img_original,
            predicted_keypoints_2d,
            save_path,
            shift,
            scale,
            ax2,
            prefix,
        ) = save_path
        to_save = True

    mean_zfoot_val = {}
    with tqdm.trange(n_steps) as pbar:
        for i in pbar:
            global_step = i + i_ini
            optimizer.zero_grad()

            (
                rotmat_pred,
                betas_pred,
                camera_pred,
                keypoints_3d_pred,
                z,
                vertices_2d_pred,
                smpl_output,
                (rleg, lleg),
                joints_2d_orig,
            ) = get_pred_and_data(
                model_hmr,
                smpl,
                selector,
                input_img,
            )
            keypoints_2d_pred = keypoints_3d_pred[:, :2]
            if to_save:
                utils.save_results_image(
                    camera=camera_pred.detach().cpu().numpy(),
                    focal_length_x=spin.constants.FOCAL_LENGTH,
                    focal_length_y=spin.constants.FOCAL_LENGTH,
                    vertices=smpl_output.vertices.detach()[0].cpu().numpy(),
                    input_img=img_original,
                    faces=smpl.faces,
                    keypoints=predicted_keypoints_2d,
                    keypoints_2=unnormalize_keypoints_from_spin(
                        keypoints_2d_pred.detach().cpu().numpy(), shift, scale, ax2
                    ),
                    # keypoints_2=unnormalize_keypoints_from_spin(joints_2d_orig.detach().cpu().numpy(), shift, scale, ax2),
                    # heatmap=predicted_contact_heatmap_raw,
                    filename=save_path / f"{prefix}_{i:0>4}.png",
                    contactlist=contact,
                    user_study=False,
                )

            loss = l2 = 0.0
            if c_mse > 0 and loss_mse is not None:
                l2 = loss_mse(keypoints_2d_pred, keypoints_2d)
                loss = loss + c_mse * l2

                if writer is not None:
                    writer.add_scalar("mse", l2, global_step=global_step)

            vertices_pred = smpl_output.vertices

            lpar = z_loss = loss_sh = 0.0
            if c_new_mse > 0 and loss_parallel is not None:
                Ltan, Lcos, Lpar, Lspine, Lgr, Lstraight3d, Lcon2d = loss_parallel(
                    keypoints_3d_pred,
                    keypoints_2d,
                    z,
                    (rleg, lleg),
                    writer=writer,
                    global_step=global_step,
                )
                lpar = (
                    Ltan
                    + c_new_mse * (args.c_f * Lcos + args.c_parallel * Lpar)
                    + Lspine
                    + args.c_reg * Lgr
                    + args.c_reg * Lstraight3d
                    + args.c_cont2d * Lcon2d
                )
                loss = loss + 300 * lpar

                if writer is not None:
                    writer.add_scalar("tan", Ltan, global_step=global_step)
                    writer.add_scalar("cos", Lcos, global_step=global_step)
                    writer.add_scalar("par", Lpar, global_step=global_step)
                    writer.add_scalar("spine", Lspine, global_step=global_step)
                    writer.add_scalar("ground/chain", Lgr, global_step=global_step)
                    writer.add_scalar(
                        "straight_in_3d", Lstraight3d, global_step=global_step
                    )
                    writer.add_scalar("contact/con2d", Lcon2d, global_step=global_step)

                for side in ["left", "right"]:
                    attr = f"{side}_foot_inds"
                    if hasattr(loss_parallel, attr):
                        foot_inds = getattr(loss_parallel, attr)
                        zind = 1
                        if attr not in mean_zfoot_val:
                            with torch.no_grad():
                                mean_zfoot_val[attr] = torch.median(
                                    vertices_pred[0, foot_inds, zind], dim=0
                                ).values

                        loss_foot = (
                            (vertices_pred[0, foot_inds, zind] - mean_zfoot_val[attr])
                            ** 2
                        ).sum()
                        loss = loss + args.c_reg * loss_foot

                        if writer is not None:
                            writer.add_scalar(
                                f"ground/{side} foot",
                                loss_foot,
                                global_step=global_step,
                            )

                if hasattr(loss_parallel, "silhuette_vertices_inds"):
                    inds = loss_parallel.silhuette_vertices_inds
                    loss_sh = (
                        (vertices_pred[0, inds, 1] - loss_parallel.ground) ** 2
                    ).sum()
                    loss = loss + args.c_reg * loss_sh

                    if writer is not None:
                        writer.add_scalar(
                            "ground/silhuette", loss_sh, global_step=global_step
                        )

            lbeta = (betas_pred**2).mean()
            lcam = ((torch.exp(-camera_pred[0] * 10)) ** 2).mean()
            loss = loss + c_beta * lbeta + lcam

            if writer is not None:
                writer.add_scalar("loss/beta", lbeta, global_step=global_step)
                writer.add_scalar("loss/cam", lcam, global_step=global_step)

            lgsc_a = gsc_contact_loss = faces_angle_loss = 0.0
            if sc_crit is not None:
                gsc_contact_loss, faces_angle_loss = sc_crit(
                    vertices_pred,
                )
                lgsc_a = 1000 * gsc_contact_loss + 0.1 * faces_angle_loss
                loss = loss + lgsc_a

                if writer is not None:
                    writer.add_scalar(
                        "contact/gsc", gsc_contact_loss, global_step=global_step
                    )
                    writer.add_scalar(
                        "contact/faces_angle", faces_angle_loss, global_step=global_step
                    )

            msc_loss = 0.0
            if contact is not None and len(contact) > 0 and msc_crit is not None:
                if not isinstance(contact, list):
                    contact = [contact]

                for cntct in contact:
                    msc_loss = msc_crit(
                        cntct,
                        vertices_pred,
                    )
                    loss = loss + args.c_msc * msc_loss

                    if writer is not None:
                        writer.add_scalar(
                            "contact/msc", msc_loss, global_step=global_step
                        )

            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()
            pbar.set_postfix(
                **{
                    "l": f"{epoch_loss:.3}",
                    "l2": f"{l2:.3}",
                    "par": f"{lpar:.3}",
                    "beta": f"{lbeta:.3}",
                    "cam": f"{lcam:.3}",
                    "z": f"{z_loss:.3}",
                    "gsc_contact": f"{float(gsc_contact_loss):.3}",
                    "faces_angle": f"{float(faces_angle_loss):.3}",
                    "msc": f"{float(msc_loss):.3}",
                }
            )

    with torch.no_grad():
        (
            rotmat_pred,
            betas_pred,
            camera_pred,
            keypoints_3d_pred,
            z,
            vertices_2d_pred,
            smpl_output,
            (rleg, lleg),
            joints_2d_orig,
        ) = get_pred_and_data(
            model_hmr,
            smpl,
            selector,
            input_img,
            zero_hands=True,
        )

    return (
        rotmat_pred,
        betas_pred,
        camera_pred,
        keypoints_3d_pred,
        vertices_2d_pred,
        smpl_output,
        z,
        joints_2d_orig,
    )


def optimize_ft(
    theta,
    camera,
    smpl,
    selector,
    input_img,
    keypoints_2d,
    args,
    loss_mse=None,
    loss_parallel=None,
    c_mse=0.0,
    c_new_mse=1.0,
    sc_crit=None,
    msc_crit=None,
    contact=None,
    n_steps=60,
    save_path=None,
    writer=None,
    i_ini=0,
    zero_hands=False,
    fist=None,
):
    to_save = False
    if save_path is not None:
        (
            img_original,
            predicted_keypoints_2d,
            save_path,
            shift,
            scale,
            ax2,
            prefix,
        ) = save_path
        to_save = True

    mean_zfoot_val = {}

    theta = theta.detach().clone()
    camera = camera.detach().clone()
    rotmat_pred = nn.Parameter(theta)
    camera_pred = nn.Parameter(camera)
    optimizer = torch.optim.Adam(
        [
            rotmat_pred,
            camera_pred,
        ],
        lr=1e-3,
    )
    global_step = i_ini

    with tqdm.trange(n_steps) as pbar:
        for i in pbar:
            global_step = i + i_ini
            optimizer.zero_grad()

            global_orient = rotmat_pred[:3]
            body_pose = rotmat_pred[3:]
            smpl_output = smpl(
                global_orient=global_orient.unsqueeze(0),
                body_pose=body_pose.unsqueeze(0),
                pose2rot=True,
            )

            z = smpl_output.joints
            z = z.squeeze(0)

            joints = smpl_output.joints.squeeze(0)
            joints_2d = project_and_normalize_to_spin(joints, camera_pred)
            rleg, lleg = project_and_normalize_to_spin_legs(
                joints, smpl_output.A, camera_pred
            )
            joints_2d = joints_2d[selector]
            z = z[selector]
            keypoints_3d_pred = joints_2d

            keypoints_2d_pred = keypoints_3d_pred[:, :2]
            if to_save:
                utils.save_results_image(
                    camera=camera_pred.detach().cpu().numpy(),
                    focal_length_x=spin.constants.FOCAL_LENGTH,
                    focal_length_y=spin.constants.FOCAL_LENGTH,
                    vertices=smpl_output.vertices.detach()[0].cpu().numpy(),
                    input_img=img_original,
                    faces=smpl.faces,
                    keypoints=predicted_keypoints_2d,
                    keypoints_2=unnormalize_keypoints_from_spin(
                        keypoints_2d_pred.detach().cpu().numpy(), shift, scale, ax2
                    ),
                    # keypoints_2=unnormalize_keypoints_from_spin(joints_2d_orig.detach().cpu().numpy(), shift, scale, ax2),
                    # heatmap=predicted_contact_heatmap_raw,
                    filename=save_path / f"{prefix}_{i:0>4}.png",
                    contactlist=contact,
                    user_study=False,
                )

            lprior = ((rotmat_pred - theta) ** 2).sum() + (
                (camera_pred - camera) ** 2
            ).sum()
            loss = lprior

            l2 = 0.0
            if c_mse > 0 and loss_mse is not None:
                l2 = loss_mse(keypoints_2d_pred, keypoints_2d)
                loss = loss + c_mse * l2

                if writer is not None:
                    writer.add_scalar("mse", l2, global_step=global_step)

            vertices_pred = smpl_output.vertices

            lpar = z_loss = loss_sh = 0.0
            if c_new_mse > 0 and loss_parallel is not None:
                Ltan, Lcos, Lpar, Lspine, Lgr, Lstraight3d, Lcon2d = loss_parallel(
                    keypoints_3d_pred,
                    keypoints_2d,
                    z,
                    (rleg, lleg),
                    writer=writer,
                    global_step=global_step,
                )
                lpar = (
                    Ltan
                    + c_new_mse * (args.c_f * Lcos + args.c_parallel * Lpar)
                    + Lspine
                    + args.c_reg * Lgr
                    + args.c_reg * Lstraight3d
                    + args.c_cont2d * Lcon2d
                )
                loss = loss + 300 * lpar

                if writer is not None:
                    writer.add_scalar("tan", Ltan, global_step=global_step)
                    writer.add_scalar("cos", Lcos, global_step=global_step)
                    writer.add_scalar("par", Lpar, global_step=global_step)
                    writer.add_scalar("spine", Lspine, global_step=global_step)
                    writer.add_scalar("ground/chain", Lgr, global_step=global_step)
                    writer.add_scalar(
                        "straight_in_3d", Lstraight3d, global_step=global_step
                    )
                    writer.add_scalar("contact/con2d", Lcon2d, global_step=global_step)

                for side in ["left", "right"]:
                    attr = f"{side}_foot_inds"
                    if hasattr(loss_parallel, attr):
                        foot_inds = getattr(loss_parallel, attr)
                        zind = 1
                        if attr not in mean_zfoot_val:
                            with torch.no_grad():
                                mean_zfoot_val[attr] = torch.median(
                                    vertices_pred[0, foot_inds, zind], dim=0
                                ).values

                        loss_foot = (
                            (vertices_pred[0, foot_inds, zind] - mean_zfoot_val[attr])
                            ** 2
                        ).sum()
                        loss = loss + args.c_reg * loss_foot

                        if writer is not None:
                            writer.add_scalar(
                                f"ground/{side} foot",
                                loss_foot,
                                global_step=global_step,
                            )

                if hasattr(loss_parallel, "silhuette_vertices_inds"):
                    inds = loss_parallel.silhuette_vertices_inds
                    loss_sh = (
                        (vertices_pred[0, inds, 1] - loss_parallel.ground) ** 2
                    ).sum()
                    loss = loss + args.c_reg * loss_sh

                    if writer is not None:
                        writer.add_scalar(
                            "ground/silhuette", loss_sh, global_step=global_step
                        )

            lgsc_a = gsc_contact_loss = faces_angle_loss = 0.0
            if sc_crit is not None:
                gsc_contact_loss, faces_angle_loss = sc_crit(vertices_pred)
                lgsc_a = 1000 * gsc_contact_loss + 0.1 * faces_angle_loss
                loss = loss + lgsc_a

                if writer is not None:
                    writer.add_scalar(
                        "contact/gsc", gsc_contact_loss, global_step=global_step
                    )
                    writer.add_scalar(
                        "contact/faces_angle", faces_angle_loss, global_step=global_step
                    )

            msc_loss = 0.0
            if contact is not None and len(contact) > 0 and msc_crit is not None:
                if not isinstance(contact, list):
                    contact = [contact]

                for cntct in contact:
                    msc_loss = msc_crit(
                        cntct,
                        vertices_pred,
                    )
                    loss = loss + args.c_msc * msc_loss

                    if writer is not None:
                        writer.add_scalar(
                            "contact/msc", msc_loss, global_step=global_step
                        )

            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()
            pbar.set_postfix(
                **{
                    "l": f"{epoch_loss:.3}",
                    "l2": f"{l2:.3}",
                    "par": f"{lpar:.3}",
                    "z": f"{z_loss:.3}",
                    "gsc_contact": f"{float(gsc_contact_loss):.3}",
                    "faces_angle": f"{float(faces_angle_loss):.3}",
                    "msc": f"{float(msc_loss):.3}",
                }
            )

    rotmat_pred = rotmat_pred.detach()

    if zero_hands:
        for i in [20, 21]:
            rotmat_pred[3 * i : 3 * (i + 1)] = 0

        for i in [12, 15]:  # neck, head
            rotmat_pred[3 * i + 1] = 0  # y

    global_orient = rotmat_pred[:3]
    body_pose = rotmat_pred[3:]
    left_hand_pose = None
    right_hand_pose = None
    if fist is not None:
        left_hand_pose = rotmat_pred.new_tensor(fist_pose.LEFT_RELAXED).unsqueeze(0)
        right_hand_pose = rotmat_pred.new_tensor(fist_pose.RIGHT_RELAXED).unsqueeze(0)
        for f in fist:
            pp = fist_pose.INT_TO_FIST[f]
            if pp is not None:
                pp = rotmat_pred.new_tensor(pp).unsqueeze(0)

            if f.startswith("lf"):
                left_hand_pose = pp
            elif f.startswith("rf"):
                right_hand_pose = pp
            elif f.startswith("l"):
                body_pose[19 * 3 : 19 * 3 + 3] = pp
                left_hand_pose = None
            elif f.startswith("r"):
                body_pose[20 * 3 : 20 * 3 + 3] = pp
                right_hand_pose = None
            else:
                raise RuntimeError(f"No such hand pose: {f}")

    with torch.no_grad():
        smpl_output = smpl(
            global_orient=global_orient.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            pose2rot=True,
        )

    return rotmat_pred, smpl_output


def create_bone(i, j, keypoints_2d):
    a = keypoints_2d[i]
    b = keypoints_2d[j]
    ab = b - a
    ab = torch.nn.functional.normalize(ab, dim=0)

    return ab


def is_parallel_to_plane(bone, thresh=21):
    return abs(bone[0]) > math.cos(math.radians(thresh))


def is_close_to_plane(bone, plane, thresh):
    dist = abs(bone[0] - plane)

    return dist < thresh


def get_selector():
    selector = []
    for kp in pose_estimation.KPS:
        tmp = spin.JOINT_NAMES.index(PE_KSP_TO_SPIN[kp])
        selector.append(tmp)

    return selector


def calc_cos(joints_2d, joints_3d):
    cos = []
    for i, j in pose_estimation.SKELETON:
        a = joints_2d[i] - joints_2d[j]
        a = nn.functional.normalize(a, dim=0)

        b = joints_3d[i] - joints_3d[j]
        b = nn.functional.normalize(b, dim=0)[:2]

        c = (a * b).sum()
        cos.append(c)

    cos = torch.stack(cos, dim=0)

    return cos


def get_natural(keypoints_2d, vertices, right_foot_inds, left_foot_inds, loss_parallel, smpl):
    height_2d = (
        keypoints_2d.max(dim=0).values[0] - keypoints_2d.min(dim=0).values[0]
    ).item()
    plane_2d = keypoints_2d.max(dim=0).values[0].item()

    ground_parallel = []
    parallel_in_3d = []
    parallel3d_bones = set()

    # parallel chains
    for i, j, k in [
        ("Right Upper Leg", "Right Leg", "Right Foot"),
        ("Right Leg", "Right Foot", "Right Toe"),  # to remove?
        ("Left Upper Leg", "Left Leg", "Left Foot"),
        ("Left Leg", "Left Foot", "Left Toe"),  # to remove?
        ("Right Shoulder", "Right Arm", "Right Hand"),
        ("Left Shoulder", "Left Arm", "Left Hand"),
        # ("Hips", "Spine", "Neck"),
        # ("Spine", "Neck", "Head"),
    ]:
        i = pose_estimation.KPS.index(i)
        j = pose_estimation.KPS.index(j)
        k = pose_estimation.KPS.index(k)
        upleg_leg = create_bone(i, j, keypoints_2d)
        leg_foot = create_bone(j, k, keypoints_2d)

        if is_parallel_to_plane(upleg_leg) and is_parallel_to_plane(leg_foot):
            if is_close_to_plane(
                upleg_leg, plane_2d, thresh=0.1 * height_2d
            ) or is_close_to_plane(leg_foot, plane_2d, thresh=0.1 * height_2d):
                ground_parallel.append(((i, j), 1))
                ground_parallel.append(((j, k), 1))

        if (upleg_leg * leg_foot).sum() > math.cos(math.radians(21)):
            parallel_in_3d.append(((i, j), (j, k)))
            parallel3d_bones.add((i, j))
            parallel3d_bones.add((j, k))

    # parallel feets
    for i, j in [
        ("Right Foot", "Right Toe"),
        ("Left Foot", "Left Toe"),
    ]:
        i = pose_estimation.KPS.index(i)
        j = pose_estimation.KPS.index(j)
        if (i, j) in parallel3d_bones:
            continue

        foot_toe = create_bone(i, j, keypoints_2d)
        if is_parallel_to_plane(foot_toe, thresh=25):
            if "Right" in pose_estimation.KPS[i]:
                loss_parallel.right_foot_inds = right_foot_inds
            else:
                loss_parallel.left_foot_inds = left_foot_inds

    loss_parallel.ground_parallel = ground_parallel
    loss_parallel.parallel_in_3d = parallel_in_3d

    vertices_np = vertices[0].cpu().numpy()
    if len(ground_parallel) > 0:
        # Silhuette veritices
        mesh = trimesh.Trimesh(vertices=vertices_np, faces=smpl.faces, process=False)
        silhuette_vertices_mask_1 = np.abs(mesh.vertex_normals[..., 2]) < 2e-1
        height_3d = vertices_np[:, 1].max() - vertices_np[:, 1].min()
        plane_3d = vertices_np[:, 1].max()
        silhuette_vertices_mask_2 = (
            np.abs(vertices_np[:, 1] - plane_3d) < 0.15 * height_3d
        )
        silhuette_vertices_mask = np.logical_and(
            silhuette_vertices_mask_1, silhuette_vertices_mask_2
        )
        (silhuette_vertices_inds,) = np.where(silhuette_vertices_mask)
        if len(silhuette_vertices_inds) > 0:
            loss_parallel.silhuette_vertices_inds = silhuette_vertices_inds
            loss_parallel.ground = plane_3d


def get_cos(keypoints_3d_pred, use_angle_transf, loss_parallel):
    keypoints_2d_pred = keypoints_3d_pred[:, :2]
    with torch.no_grad():
        cos_r = calc_cos(keypoints_2d_pred, keypoints_3d_pred)

    alpha = torch.acos(cos_r)
    if use_angle_transf:
        leg_inds = [
            5,
            6,  # right leg
            7,
            8,  # left leg
        ]
        foot_inds = [15, 16]
        nleg_inds = sorted(
            set(range(len(pose_estimation.SKELETON))) - set(leg_inds) - set(foot_inds)
        )
        alpha[nleg_inds] = alpha[nleg_inds] - alpha[nleg_inds].min()

        amli = alpha[leg_inds].min()
        leg_inds.extend(foot_inds)
        alpha[leg_inds] = alpha[leg_inds] - amli

        angles = alpha.detach().cpu().numpy()
        angles = hist_cub.cub(
            angles / (math.pi / 2),
            a=1.2121212121212122,
            b=-1.105527638190953,
            c=0.787878787878789,
        ) * (math.pi / 2)
        alpha = alpha.new_tensor(angles)

    loss_parallel.cos = torch.cos(alpha)

    return cos_r


def save_mesh_with_winding_numbers(sc_module, vertices, smpl, save_path):
    triangles = sc_module.triangles(vertices)
    exterior = sc_module.get_intersection_mask(vertices, triangles, test_segments=False)
    exterior = exterior.cpu().numpy().squeeze(0)
    utils.save_mesh_with_colors(
        vertices[0].cpu().numpy(),
        smpl.faces,
        save_path / "winding_numbers.ply",
        mask=exterior,
    )

    exterior = sc_module.get_intersection_mask(vertices, triangles)
    exterior = exterior.cpu().numpy().squeeze(0)
    utils.save_mesh_with_colors(
        vertices[0].cpu().numpy(),
        smpl.faces,
        save_path / "winding_numbers_filtered.ply",
        mask=exterior,
    )


def get_contacts(
    args,
    sc_module,
    y_data_conts,
    keypoints_2d,
    vertices,
    bone_to_params,
    loss_parallel,
    img_size_original,
    save_path,
):
    use_contacts = args.use_contacts
    use_msc = args.use_msc
    c_mse = args.c_mse

    if use_contacts:
        assert c_mse == 0
        contact, contact_2d, for_mask = find_contacts(
            y_data_conts, keypoints_2d, bone_to_params
        )
        if len(contact_2d) > 0:
            loss_parallel.contact_2d = contact_2d

            mask = np.zeros((spin.constants.IMG_RES, spin.constants.IMG_RES), dtype="uint8")
            mask += 255
            cv2.drawContours(mask, for_mask, -1, 0, 2)
            mask = cv2.resize(mask, img_size_original[::-1])
            cv2.imwrite(str(save_path / "mask.png"), mask)

        if len(contact) == 0:
            _, contact = sc_module.verts_in_contact(vertices, return_idx=True)
            contact = contact.cpu().numpy().ravel()
    elif use_msc:
        _, contact = sc_module.verts_in_contact(vertices, return_idx=True)
        contact = contact.cpu().numpy().ravel()
    else:
        contact = np.array([])

    return contact


def save_all(
    keypoints_3d_pred,
    rotmat_pred,
    camera_pred,
    betas_pred,
    smpl,
    contact,
    img_original,
    predicted_keypoints_2d,
    predicted_contact_heatmap_raw,
    loss_parallel,
    smpl_output,
    shift,
    scale,
    ax2,
    summary_writer,
    save_path,
    fname,
):
    keypoints_2d_pred = keypoints_3d_pred[:, :2]

    vertices = smpl_output.vertices.detach()
    betas_pred = betas_pred.detach().cpu().numpy()

    utils.save_pose_params(
        rotmat_pred,
        camera_pred,
        betas_pred,
        vertices,
        smpl,
        contact,
        save_path / f"{fname}.pkl",
    )

    if hasattr(loss_parallel, "silhuette_vertices_inds"):
        contact.append(loss_parallel.silhuette_vertices_inds)

    img_sw = utils.save_results_image(
        camera=camera_pred.detach().cpu().numpy(),
        focal_length_x=spin.constants.FOCAL_LENGTH,
        focal_length_y=spin.constants.FOCAL_LENGTH,
        vertices=vertices[0].cpu().numpy(),
        input_img=img_original,
        faces=smpl.faces,
        keypoints=predicted_keypoints_2d,
        keypoints_2=unnormalize_keypoints_from_spin(
            keypoints_2d_pred.cpu().numpy(), shift, scale, ax2
        )
        if shift is not None
        else None,
        # keypoints_2=unnormalize_keypoints_from_spin(joints_2d_orig.detach().cpu().numpy(), shift, scale, ax2) if shift is not None else None,
        heatmap=predicted_contact_heatmap_raw,
        filename=save_path / f"{fname}.png",
        contactlist=contact,
        contact2dlist=loss_parallel.contact_2d
        if hasattr(loss_parallel, "contact_2d")
        else None,
        cos=loss_parallel.cos.tolist() if loss_parallel.cos is not None else None,
    )

    utils.save_mesh_with_colors(
        smpl_output.vertices[0].cpu().numpy(),
        smpl.faces,
        save_path / f"{fname}.ply",
        inds=contact,
    )

    joints = smpl_output.joints.squeeze(0).cpu().numpy()
    fig = utils.plot_3D(joints, vertices.squeeze(0).cpu().numpy(), smpl.faces)
    fig.write_html(save_path / f"{fname}.html")

    summary_writer.add_image(
        fname, np.array(img_sw).astype("float32") / 255, dataformats="HWC"
    )
    summary_writer.add_mesh(
        fname,
        vertices=(
            vertices.cpu().float()[0]
            @ torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
        ).unsqueeze(0),
        faces=torch.from_numpy(smpl.faces[None].astype("int64")),
    )


def spin_step(
    model_hmr,
    smpl,
    selector,
    input_img,
    img_original,
    predicted_keypoints_2d,
    predicted_contact_heatmap_raw,
    loss_parallel,
    shift,
    scale,
    ax2,
    summary_writer,
    save_path,
):
    with torch.no_grad():
        (
            rotmat_pred,
            betas_pred,
            camera_pred,
            keypoints_3d_pred,
            _,
            _,
            smpl_output,
            _,
            _,
        ) = get_pred_and_data(
            model_hmr,
            smpl,
            selector,
            input_img,
            zero_hands=True,
        )

    save_all(
        keypoints_3d_pred,
        rotmat_pred,
        camera_pred,
        betas_pred,
        smpl,
        None,
        img_original,
        predicted_keypoints_2d,
        predicted_contact_heatmap_raw,
        loss_parallel,
        smpl_output,
        shift,
        scale,
        ax2,
        summary_writer,
        save_path,
        "spin",
    )


def eft_step(
    model_hmr,
    smpl,
    selector,
    input_img,
    keypoints_2d,
    optimizer,
    args,
    loss_mse,
    loss_parallel,
    c_beta,
    sc_module,
    y_data_conts,
    bone_to_params,
    img_original,
    predicted_keypoints_2d,
    predicted_contact_heatmap_raw,
    shift,
    scale,
    ax2,
    summary_writer,
    save_path,
):
    img_size_original = img_original.shape[:2]
    (
        rotmat_pred,
        betas_pred,
        camera_pred,
        keypoints_3d_pred,
        _,
        smpl_output,
        _,
        _,
    ) = optimize(
        model_hmr,
        smpl,
        selector,
        input_img,
        keypoints_2d,
        optimizer,
        args,
        loss_mse=loss_mse,
        loss_parallel=loss_parallel,
        c_mse=1,
        c_new_mse=0,
        c_beta=c_beta,
        sc_crit=None,
        msc_crit=None,
        contact=None,
        n_steps=60 + 90,
        writer=summary_writer,
    )

    # find contacts
    vertices = smpl_output.vertices.detach()
    contact = get_contacts(
        args,
        sc_module,
        y_data_conts,
        keypoints_2d,
        vertices,
        bone_to_params,
        loss_parallel,
        img_size_original,
        save_path,
    )

    save_all(
        keypoints_3d_pred,
        rotmat_pred,
        camera_pred,
        betas_pred,
        smpl,
        contact,
        img_original,
        predicted_keypoints_2d,
        predicted_contact_heatmap_raw,
        loss_parallel,
        smpl_output,
        shift,
        scale,
        ax2,
        summary_writer,
        save_path,
        "eft",
    )

    if sc_module is not None:
        save_mesh_with_winding_numbers(sc_module, vertices, smpl, save_path)

    return vertices, keypoints_3d_pred, contact


def dc_step(
    model_hmr,
    smpl,
    selector,
    input_img,
    keypoints_2d,
    optimizer,
    args,
    loss_mse,
    loss_parallel,
    c_mse,
    c_new_mse,
    c_beta,
    sc_crit,
    msc_crit,
    contact,
    use_contacts,
    use_msc,
    img_original,
    predicted_keypoints_2d,
    predicted_contact_heatmap_raw,
    shift,
    scale,
    ax2,
    summary_writer,
    save_path,
):
    (
        rotmat_pred,
        betas_pred,
        camera_pred,
        keypoints_3d_pred,
        _,
        smpl_output,
        _,
        _,
    ) = optimize(
        model_hmr,
        smpl,
        selector,
        input_img,
        keypoints_2d,
        optimizer,
        args,
        loss_mse=loss_mse,
        loss_parallel=loss_parallel,
        c_mse=c_mse,
        c_new_mse=c_new_mse,
        c_beta=c_beta,
        sc_crit=sc_crit,
        msc_crit=msc_crit if use_contacts or use_msc else None,
        contact=contact if use_contacts or use_msc else None,
        n_steps=60 if use_contacts or use_msc else 0,  # + 60,
        # save_path=(img_original, predicted_keypoints_2d, save_path, shift, scale, ax2, "dc"),
        writer=summary_writer,
        i_ini=60 + 90,
    )

    save_all(
        keypoints_3d_pred,
        rotmat_pred,
        camera_pred,
        betas_pred,
        smpl,
        contact,
        img_original,
        predicted_keypoints_2d,
        predicted_contact_heatmap_raw,
        loss_parallel,
        smpl_output,
        shift,
        scale,
        ax2,
        summary_writer,
        save_path,
        "dc",
    )

    return rotmat_pred


def us_step(
    model_hmr,
    smpl,
    selector,
    input_img,
    rotmat_pred,
    keypoints_2d,
    args,
    loss_mse,
    loss_parallel,
    c_mse,
    c_new_mse,
    sc_crit,
    msc_crit,
    contact,
    use_contacts,
    use_msc,
    img_original,
    keypoints_3d_pred,
    summary_writer,
    save_path,
):
    (_, _, camera_pred_us, _, _, _, smpl_output_us, _, _,) = get_pred_and_data(
        model_hmr,
        smpl,
        selector,
        input_img,
        use_betas=False,
        zero_hands=True,
    )

    rotmat_pred_us, smpl_output_us = optimize_ft(
        rotmat_pred,
        camera_pred_us,
        smpl,
        selector,
        input_img,
        keypoints_2d,
        args,
        loss_mse=loss_mse,
        loss_parallel=loss_parallel,
        c_mse=c_mse,
        c_new_mse=c_new_mse,
        sc_crit=sc_crit,
        msc_crit=msc_crit if use_contacts or use_msc else None,
        contact=contact if use_contacts or use_msc else None,
        n_steps=60 if use_contacts or use_msc else 0,  # + 60,
        # save_path=(img_original, predicted_keypoints_2d, save_path, shift, scale, ax2, "dc"),
        writer=summary_writer,
        i_ini=60 + 90 + 60,
        zero_hands=True,
        fist=args.fist,
    )

    save_all(
        keypoints_3d_pred,
        rotmat_pred_us,
        camera_pred_us,
        torch.zeros(1, 10, dtype=torch.float32),
        smpl,
        None,
        img_original,
        None,
        None,
        loss_parallel,
        smpl_output_us,
        None,
        None,
        None,
        summary_writer,
        save_path,
        "us",
    )


def main():
    args = parse_args()
    print(args)

    # models
    model_pose = cv2.dnn.readNetFromONNX(
        args.pose_estimation_model_path
    )  # "hrn_w48_384x288.onnx"
    model_contact = cv2.dnn.readNetFromONNX(
        args.contact_model_path
    )  # "contact_hrn_w32_256x192.onnx"

    device = (
        torch.device(args.device) if torch.cuda.is_available() else torch.device("cpu")
    )
    model_hmr = spin.hmr(args.smpl_mean_params_path)  # "smpl_mean_params.npz"
    model_hmr.to(device)
    checkpoint = torch.load(
        args.spin_model_path,  # "spin_model_smplx_eft_18.pt"
        map_location="cpu"
    )

    smpl = spin.SMPLX(
        args.smpl_model_dir,  # "models/smplx"
        batch_size=1,
        create_transl=False,
        use_pca=False,
        flat_hand_mean=args.fist is not None,
    )
    smpl.to(device)

    selector = get_selector()

    use_contacts = args.use_contacts
    use_msc = args.use_msc

    bone_to_params = np.load(args.bone_parametrization_path, allow_pickle=True).item()
    foot_inds = np.load(args.foot_inds_path, allow_pickle=True).item()
    left_foot_inds = foot_inds["left_foot_inds"]
    right_foot_inds = foot_inds["right_foot_inds"]

    if use_contacts:
        model_type = args.smpl_type
        sc_module = selfcontact.SelfContact(
            essentials_folder=args.essentials_dir,  # "smplify-xmc-essentials"
            geothres=0.3,
            euclthres=0.02,
            test_segments=True,
            compute_hd=True,
            model_type=model_type,
            device=device,
        )
        sc_module.to(device)

        sc_crit = selfcontact.losses.SelfContactLoss(
            contact_module=sc_module,
            inside_loss_weight=0.5,
            outside_loss_weight=0.0,
            contact_loss_weight=0.5,
            align_faces=True,
            use_hd=True,
            test_segments=True,
            device=device,
            model_type=model_type,
        )
        sc_crit.to(device)

        msc_crit = losses.MimickedSelfContactLoss(geodesics_mask=sc_module.geomask)
        msc_crit.to(device)
    else:
        sc_module = None
        sc_crit = None
        msc_crit = None

    loss_mse = losses.MSE([1, 10, 13])  # Neck + Right Upper Leg + Left Upper Leg

    ignore = (
        (1, 2),  # Neck + Right Shoulder
        (1, 5),  # Neck + Left Shoulder
        (9, 10),  # Hips + Right Upper Leg
        (9, 13),  # Hips + Left Upper Leg
    )
    loss_parallel = losses.Parallel(
        skeleton=pose_estimation.SKELETON,
        ignore=ignore,
    )

    c_mse = args.c_mse
    c_new_mse = args.c_par
    c_beta = 1e-3

    if c_mse > 0:
        assert c_new_mse == 0
    elif c_mse == 0:
        assert c_new_mse > 0

    root_path = Path(args.save_path)
    root_path.mkdir(exist_ok=True, parents=True)

    path_to_imgs = Path(args.img_path)
    if path_to_imgs.is_dir():
        path_to_imgs = path_to_imgs.iterdir()
    else:
        path_to_imgs = [path_to_imgs]

    for img_path in path_to_imgs:
        if not any(
            img_path.name.lower().endswith(ext) for ext in [".jpg", ".png", ".jpeg"]
        ):
            continue

        img_name = img_path.stem

        # use 2d keypoints detection
        (
            img_original,
            predicted_keypoints_2d,
            _,
            _,
        ) = pose_estimation.infer_single_image(
            model_pose,
            img_path,
            input_img_size=pose_estimation.IMG_SIZE,
            return_kps=True,
        )

        save_path = root_path / img_name
        save_path.mkdir(exist_ok=True, parents=True)
        # if (save_path / "us_orig.png").is_file():
        #     return

        summary_writer = SummaryWriter(log_dir=save_path / f"runDoknc2_{c_new_mse}")

        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_size_original = img_original.shape[:2]
        keypoints_2d, shift, scale, ax2 = normalize_keypoints_to_spin(
            predicted_keypoints_2d, img_size_original
        )
        keypoints_2d = torch.from_numpy(keypoints_2d)
        keypoints_2d = keypoints_2d.to(device)

        (
            predicted_contact_heatmap,
            predicted_contact_heatmap_raw,
            very_hm_raw,
        ) = get_contact_heatmap(model_contact, img_path)
        predicted_contact_heatmap_raw = Image.fromarray(
            predicted_contact_heatmap_raw
        ).resize(img_size_original[::-1])
        predicted_contact_heatmap_raw = cv2.resize(very_hm_raw, img_size_original[::-1])

        if c_new_mse == 0:
            predicted_contact_heatmap_raw = None

        y_data_conts = get_vertices_in_heatmap(predicted_contact_heatmap)

        model_hmr.load_state_dict(checkpoint["model"], strict=True)
        model_hmr.train()
        freeze_layers(model_hmr)

        _, input_img = spin.process_image(img_path, input_res=spin.constants.IMG_RES)
        input_img = input_img.to(device)

        spin_step(
            model_hmr,
            smpl,
            selector,
            input_img,
            img_original,
            predicted_keypoints_2d,
            predicted_contact_heatmap_raw,
            loss_parallel,
            shift,
            scale,
            ax2,
            summary_writer,
            save_path,
        )

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model_hmr.parameters()),
            lr=1e-6,
        )

        vertices, keypoints_3d_pred, contact = eft_step(
            model_hmr,
            smpl,
            selector,
            input_img,
            keypoints_2d,
            optimizer,
            args,
            loss_mse,
            loss_parallel,
            c_beta,
            sc_module,
            y_data_conts,
            bone_to_params,
            img_original,
            predicted_keypoints_2d,
            predicted_contact_heatmap_raw,
            shift,
            scale,
            ax2,
            summary_writer,
            save_path,
        )

        if args.use_natural:
            get_natural(
                keypoints_2d, vertices, right_foot_inds, left_foot_inds, loss_parallel, smpl,
            )

        if args.use_cos:
            cos_r = get_cos(keypoints_3d_pred, args.use_angle_transf, loss_parallel)
            np.save(save_path / "cos_hist", cos_r.cpu().numpy())

        rotmat_pred = dc_step(
            model_hmr,
            smpl,
            selector,
            input_img,
            keypoints_2d,
            optimizer,
            args,
            loss_mse,
            loss_parallel,
            c_mse,
            c_new_mse,
            c_beta,
            sc_crit,
            msc_crit,
            contact,
            use_contacts,
            use_msc,
            img_original,
            predicted_keypoints_2d,
            predicted_contact_heatmap_raw,
            shift,
            scale,
            ax2,
            summary_writer,
            save_path,
        )

        us_step(
            model_hmr,
            smpl,
            selector,
            input_img,
            rotmat_pred,
            keypoints_2d,
            args,
            loss_mse,
            loss_parallel,
            c_mse,
            c_new_mse,
            sc_crit,
            msc_crit,
            contact,
            use_contacts,
            use_msc,
            img_original,
            keypoints_3d_pred,
            summary_writer,
            save_path,
        )


if __name__ == "__main__":
    main()
