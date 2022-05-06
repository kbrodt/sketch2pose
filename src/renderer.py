import numpy as np
import pyrender
import torch
import trimesh
from torchvision.utils import make_grid


class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """

    def __init__(self, focal_length=5000, img_res=224, faces=None):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=img_res, viewport_height=img_res, point_size=1.0
        )
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation, images):
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(
                np.transpose(
                    self.__call__(vertices[i], camera_translation[i], images_np[i]),
                    (2, 0, 1),
                )
            ).float()
            rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow=2)
        return rend_imgs

    def __call__(self, vertices, camera_translation, image):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2, alphaMode="OPAQUE", baseColorFactor=(0.8, 0.3, 0.3, 1.0)
        )

        camera_translation[0] *= -1.0

        mesh = trimesh.Trimesh(vertices, self.faces)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, "mesh")

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=self.camera_center[0],
            cy=self.camera_center[1],
        )
        scene.add(camera, pose=camera_pose)

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)

        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:, :, None]
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image
        return output_img


def overlay_mesh(
    verts,
    faces,
    camera_transl,
    focal_length_x,
    focal_length_y,
    camera_center,
    H,
    W,
    img,
    camera_rotation=None,
    rotaround=None,
    contactlist=None,
    color=False,
    scale=1,
):

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0, alphaMode="OPAQUE", baseColorFactor=(1.0, 1.0, 0.9, 1.0)
    )
    out_mesh = trimesh.Trimesh(verts, faces, process=False)
    out_mesh_col = np.array(out_mesh.visual.vertex_colors)

    if contactlist is not None and len(contactlist) > 0:
        color = [255, 0, 0, 255]
        out_mesh_col[contactlist] = color
        out_mesh.visual.vertex_colors = out_mesh_col

    if camera_rotation is None:
        camera_rotation = np.eye(3)
    else:
        camera_rotation = camera_rotation[0]

    # rotate mesh and stack output images
    if rotaround is None:
        out_mesh.vertices = np.matmul(verts, camera_rotation.T) + camera_transl
    else:
        base_mesh = trimesh.Trimesh(verts, faces, process=False)
        # rot_center = (base_mesh.vertices[5615] + base_mesh.vertices[5614] ) / 2
        rot = trimesh.transformations.rotation_matrix(
            np.radians(rotaround), [0, 1, 0], base_mesh.vertices[4297]
        )
        base_mesh.apply_transform(rot)
        out_mesh.vertices = (
            np.matmul(base_mesh.vertices, camera_rotation.T) + camera_transl
        )

    out_mesh.vertices += np.array([0, 0, 50])
    # add mesh to scene
    mesh = pyrender.Mesh.from_trimesh(
        out_mesh,
        material=material,
        smooth=False,
    )
    if img is not None:
        scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=(0.3, 0.3, 0.3, 1.0),
        )
    else:
        scene = pyrender.Scene(
            bg_color=[1.0, 1.0, 1.0, 1.0],
            ambient_light=(0.3, 0.3, 0.3, 1.0),
        )
    scene.add(mesh, "mesh")

    # create and add camera
    camera_pose = np.eye(4)
    camera_pose[1, :] = -camera_pose[1, :]
    camera_pose[2, :] = -camera_pose[2, :]
    pyrencamera = pyrender.camera.OrthographicCamera(
        camera_transl[2],
        camera_transl[2],
        znear=1e-6,
        zfar=1000000,
    )
    scene.add(pyrencamera, pose=camera_pose)

    # create and add light
    light = pyrender.PointLight(
        color=[1.0, 1.0, 1.0],
        intensity=1,
    )
    light_pose = np.eye(4)
    for lp in [[1, 1, -1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]:
        light_pose[:3, 3] = out_mesh.vertices.mean(0) + np.array(lp)
        scene.add(light, pose=light_pose)

    r = pyrender.OffscreenRenderer(
        viewport_width=int(scale * W),
        viewport_height=int(scale * H),
        point_size=1.0,
    )
    color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    if img is not None:
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = color[:, :, :-1] * valid_mask + (1 - valid_mask) * img
    else:
        output_img = color

    output_img = (output_img * 255).astype(np.uint8)[..., :3]

    return output_img
