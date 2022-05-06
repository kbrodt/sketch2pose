import colorsys
import itertools
import json
import pickle

import cv2
import plotly.graph_objects as go
import trimesh
import torch
import numpy as np
import PIL.Image as pil_img
import PIL.ImageDraw as ImageDraw
from PIL import Image, ImageChops
from skimage import exposure

import spin
import renderer
import pose_estimation


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(o, path):
    with open(path, "w") as f:
        json.dump(o, f)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(o, path):
    with open(path, "wb") as f:
        pickle.dump(o, f)


def plot_3D(joints, vertices, faces):
    x, y, z = joints.T
    x1, y1, z1 = vertices.T
    i, j, k = faces.T

    data = [
        go.Mesh3d(
            x=x1,
            y=y1,
            z=z1,
            i=i,
            j=j,
            k=k,
        ),
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker_size=5,
        ),
    ]

    fig = go.Figure(
        data=data,
    )

    return fig


def draw_keypoints(
    input_img_kp,
    keypoints,
    skeleton,
    r,
    color,
    contact2dlist=None,
    contact2dlist_color="green",
    cos=None,
):
    if keypoints is not None:
        draw = ImageDraw.Draw(input_img_kp)

        for skidx, (i, j) in enumerate(skeleton):
            a = keypoints[i]
            b = keypoints[j]
            ln = np.linalg.norm(b - a)

            xy = [a[0], a[1], b[0], b[1]]
            if cos is not None:
                c = colorsys.hsv_to_rgb(cos[skidx] ** 8, 1, 1)
                c = tuple(int(c_ * 255) for c_ in c)
                draw.line(xy, fill=c, width=r)
            else:
                draw.line(xy, fill=color, width=r)

        draw_kpts = [(p[0] - r, p[1] - r, p[0] + r, p[1] + r) for p in keypoints]
        for _, elipse in enumerate(draw_kpts):
            draw.ellipse(elipse, fill="black", outline="black")

        if contact2dlist is not None:
            keypoints_torch = torch.from_numpy(keypoints)
            for c2d in contact2dlist:
                for (src_1, dst_1, t_1), (src_2, dst_2, t_2) in itertools.combinations(
                    c2d, 2
                ):
                    a = torch.lerp(
                        keypoints_torch[src_1], keypoints_torch[dst_1], t_1
                    ).tolist()
                    b = torch.lerp(
                        keypoints_torch[src_2], keypoints_torch[dst_2], t_2
                    ).tolist()

                    xy = [a[0], a[1], b[0], b[1]]
                    draw.line(xy, fill=contact2dlist_color, width=max(r // 3, 10))

    return input_img_kp


def save_results_image(
    camera,
    focal_length_x,
    focal_length_y,
    input_img,
    vertices,
    faces,
    filename,
    keypoints=None,
    keypoints_2=None,
    heatmap=None,
    cvt_camera=True,
    contactlist=None,
    contact2dlist=None,
    user_study=True,
    cos=None,
):
    if isinstance(contactlist, list) and len(contactlist) > 0:
        contactlist = np.concatenate(contactlist)

    H, W, _ = input_img.shape
    HW = max(H, W)
    camera_center = np.array([W // 2, H // 2])
    if not cvt_camera:
        camera_transl = camera.copy()
    else:
        camera_transl = np.stack(
            [
                camera[1],
                camera[2],
                1 / camera[0],
            ],
        )

    # draw keypoints
    input_img_kp = pil_img.fromarray(input_img)
    if keypoints is not None:
        draw_keypoints(
            input_img_kp,
            keypoints,
            pose_estimation.SKELETON,
            r=int(HW * 0.01),
            color=(255, 0, 0, 255),
            # contact2dlist=contact2dlist,
            # contact2dlist_color="orange",
        )

    if cos is not None:
        input_img_kp_cos = pil_img.fromarray(input_img)
        if keypoints is not None:
            draw_keypoints(
                input_img_kp_cos,
                keypoints,
                pose_estimation.SKELETON,
                r=int(HW * 0.01),
                color=(255, 0, 0, 255),
                # contact2dlist=contact2dlist,
                # contact2dlist_color="orange",
                cos=cos,
            )

    input_img_kp_2 = input_img_kp.copy()
    if keypoints_2 is not None:
        draw_keypoints(
            input_img_kp_2,
            keypoints_2,
            # spin.SMPLX.SKELETON if "eft" in str(filename) else pose_estimation.SKELETON,
            pose_estimation.SKELETON,
            r=int(HW * 0.01),
            color=(0, 0, 255, 255),
            contact2dlist=contact2dlist,
            contact2dlist_color="purple",
        )

    # heatmap = ImageOps.invert(heatmap)
    if heatmap is not None:
        # input_img_kp_2 = pil_img.blend(input_img_kp_2, heatmap, 0.5)

        hm = np.copy(input_img)
        gray_img = exposure.rescale_intensity(heatmap, out_range=(0, 255))
        gray_img = gray_img.astype(np.uint8)
        heatmap_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        hm = pil_img.fromarray(cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB))
        hm.save(filename.with_stem(f"{filename.stem}_heatmap"))
        input_img_kp.save(filename.with_stem(f"{filename.stem}_2dkps"))
        heatmap = pil_img.fromarray(heatmap)

        if cos is not None:
            input_img_kp_cos.save(filename.with_stem(f"{filename.stem}_2dkpscos"))

    # render fitted mesh from different views
    overlay_fit_img = renderer.overlay_mesh(
        vertices,
        faces,
        camera_transl,
        focal_length_x,
        focal_length_y,
        camera_center,
        H,
        W,
        input_img.astype("float32") / 255,
        None,
        rotaround=None,
    )

    # overlay_fit_img = pil_img.fromarray(overlay_fit_img)
    # draw_keypoints(overlay_fit_img, keypoints_2, r=int(HW * 0.01), color=(0, 0, 255, 255))

    # camera_transl[-1] *= 1
    view1_fit = renderer.overlay_mesh(
        vertices,
        faces,
        camera_transl.astype(np.float32),
        focal_length_x,
        focal_length_y,
        camera_center,
        H,
        W,
        None,
        None,
        rotaround=-45,
        contactlist=contactlist,
    )
    view2_fit = renderer.overlay_mesh(
        vertices,
        faces,
        camera_transl.astype(np.float32),
        focal_length_x,
        focal_length_y,
        camera_center,
        H,
        W,
        None,
        None,
        rotaround=None,
        contactlist=contactlist,
    )
    view3_fit = renderer.overlay_mesh(
        vertices,
        faces,
        camera_transl.astype(np.float32),
        focal_length_x,
        focal_length_y,
        camera_center,
        H,
        W,
        None,
        None,
        rotaround=90,
        contactlist=contactlist,
        scale=1,
    )

    IMG = np.vstack(
        (
            np.hstack(
                (
                    np.asarray(input_img_kp)
                    if keypoints is not None
                    else 255 * np.ones_like(np.asarray(input_img_kp)),
                    np.asarray(input_img_kp_2),
                    overlay_fit_img,
                    # np.asanyarray(overlay_fit_img),
                ),
            ),
            np.hstack(
                (
                    view1_fit,
                    view2_fit,
                    view3_fit,
                ),
            ),
        ),
    )
    IMG = pil_img.fromarray(IMG)
    IMG.thumbnail((2000, 2000))

    IMG.save(filename)

    if user_study:
        w = 768
        input_img_kp.thumbnail((w, w))
        input_img_kp.save(filename.with_stem(f"{filename.stem}_orig"))
        W, H = input_img_kp.size

        camera_transl[-1] *= 2
        view2_fit = renderer.overlay_mesh(
            vertices,
            faces,
            camera_transl.astype(np.float32),
            focal_length_x,
            focal_length_y,
            camera_center,
            H,
            W,
            None,
            None,
            rotaround=None,
            contactlist=contactlist,
            scale=2,
        )
        view2_fit = pil_img.fromarray(view2_fit)
        w *= 2
        view2_fit.thumbnail((w, w))
        view2_fit.save(filename.with_stem(f"{filename.stem}_same"))
        view3_fit = renderer.overlay_mesh(
            vertices,
            faces,
            camera_transl.astype(np.float32),
            focal_length_x,
            focal_length_y,
            camera_center,
            H,
            W,
            None,
            None,
            rotaround=90,
            contactlist=contactlist,
            scale=2,
        )
        view3_fit = pil_img.fromarray(view3_fit)
        view3_fit.thumbnail((w, w))
        view3_fit.save(filename.with_stem(f"{filename.stem}_alt"))

    return IMG


def save_3d_model_on_img(
    camera,
    vertices,
    faces,
    img,
    filename,
    save_path,
):
    img_res = max(img.shape[:2])
    r = renderer.Renderer(
        focal_length=spin.constants.FOCAL_LENGTH,
        img_res=img_res,
        faces=faces,
    )

    # Calculate camera parameters for rendering
    camera_translation = np.stack(
        [
            camera[1],
            camera[2],
            2 * spin.constants.FOCAL_LENGTH / (img_res * camera[0] + 1e-9),
        ],
    )
    # Render parametric shape
    img_shape = r(vertices, camera_translation, img)
    img_shape = (255 * img_shape).astype("uint8")
    img_shape = cv2.cvtColor(img_shape, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path / f"shape_{filename}.png"), img_shape)

    # Render side views
    aroundy = cv2.Rodrigues(np.array([0, np.radians(90.0), 0]))[0]
    center = vertices.mean(axis=0)
    rot_vertices = np.dot((vertices - center), aroundy) + center

    # Render non-parametric shape
    img_shape = r(rot_vertices, camera_translation, np.ones_like(img))
    img_shape = (255 * img_shape).astype("uint8")
    img_shape = cv2.cvtColor(img_shape, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path / f"shape_rot_{filename}.png"), img_shape)


def save_mesh_with_colors(vertices, faces, save_path, mask=None, inds=None):
    if inds is not None and isinstance(inds, list):
        inds = np.concatenate(inds)
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    color = np.array(mesh.visual.vertex_colors)
    color[:] = [233, 233, 233, 255]
    if mask is not None and any(mask):
        color[~mask] = [255, 0, 0, 255]
    elif inds is not None and len(inds) > 0:
        color[inds] = [255, 0, 0, 255]
    mesh.visual.vertex_colors = color
    mesh.export(save_path)


def save_pose_params(rotmat, camera, betas, vertices, smpl, contact, save_path):
    if contact is not None and isinstance(contact, list) and len(contact) > 0:
        contact = np.concatenate(contact)

    rotmat = rotmat.detach()
    camera = camera.detach()
    if smpl.name() == "SMPL-X":
        rotmat = rotmat[: -2 * 3]

    res = {
        "camera_s_t": camera.unsqueeze(0).cpu().numpy(),
        "global_orient": rotmat[:3].unsqueeze(0).cpu().numpy(),
        "betas": betas,
        "body_pose": rotmat[3:].unsqueeze(0).cpu().numpy(),
        "left_hand_pose": smpl.left_hand_pose.unsqueeze(0).detach().cpu().numpy(),
        "right_hand_pose": smpl.right_hand_pose.unsqueeze(0).detach().cpu().numpy(),
        "model": smpl.name().replace("-", ""),
        "gender": smpl.gender,
        "vertices": vertices[0].cpu().numpy(),
    }

    if contact is not None and len(contact) > 0:
        contact = np.array(contact)
        res["contact"] = contact
    else:
        res["v"] = vertices[0].cpu().numpy()

    save_pkl(res, save_path)

    np.savez(save_path, **res)
