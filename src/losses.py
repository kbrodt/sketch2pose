import itertools

import torch
import torch.nn as nn

import pose_estimation


class MSE(nn.Module):
    def __init__(self, ignore=None):
        super().__init__()

        self.mse = torch.nn.MSELoss(reduction="none")
        self.ignore = ignore if ignore is not None else []

    def forward(self, y_pred, y_data):
        loss = self.mse(y_pred, y_data)

        if len(self.ignore) > 0:
            loss[self.ignore] *= 0

        return loss.sum() / (len(loss) - len(self.ignore))


class Parallel(nn.Module):
    def __init__(self, skeleton, ignore=None, ground_parallel=None):
        super().__init__()

        self.skeleton = skeleton
        if ignore is not None:
            self.ignore = set(ignore)
        else:
            self.ignore = set()

        self.ground_parallel = ground_parallel if ground_parallel is not None else []
        self.parallel_in_3d = []

        self.cos = None

    def forward(self, y_pred3d, y_data, z, spine_j, writer=None, global_step=0):
        y_pred = y_pred3d[:, :2]
        rleg, lleg = spine_j

        Lcon2d = Lcount = 0
        if hasattr(self, "contact_2d"):
            for c2d in self.contact_2d:
                for (
                    (src_1, dst_1, t_1),
                    (src_2, dst_2, t_2),
                ) in itertools.combinations(c2d, 2):

                    a_1 = torch.lerp(y_data[src_1], y_data[dst_1], t_1)
                    a_2 = torch.lerp(y_data[src_2], y_data[dst_2], t_2)
                    a = a_2 - a_1

                    b_1 = torch.lerp(y_pred[src_1], y_pred[dst_1], t_1)
                    b_2 = torch.lerp(y_pred[src_2], y_pred[dst_2], t_2)
                    b = b_2 - b_1

                    lcon2d = ((a - b) ** 2).sum()
                    Lcon2d = Lcon2d + lcon2d
                    Lcount += 1

        if Lcount > 0:
            Lcon2d = Lcon2d / Lcount

        Ltan = Lpar = Lcos = Lcount = 0
        Lspine = 0
        for i, bone in enumerate(self.skeleton):
            if bone in self.ignore:
                continue

            src, dst = bone

            b = y_data[dst] - y_data[src]
            t = nn.functional.normalize(b, dim=0)
            n = torch.stack([-t[1], t[0]])

            if src == 10 and dst == 11:  # right leg
                a = rleg
            elif src == 13 and dst == 14:  # left leg
                a = lleg
            else:
                a = y_pred[dst] - y_pred[src]

            bone_name = f"{pose_estimation.KPS[src]}_{pose_estimation.KPS[dst]}"
            c = a - b
            lcos_loc = ltan_loc = lpar_loc = 0
            if self.cos is not None:
                if bone not in [
                    (1, 2),  # Neck + Right Shoulder
                    (1, 5),  # Neck + Left Shoulder
                    (9, 10),  # Hips + Right Upper Leg
                    (9, 13),  # Hips + Left Upper Leg
                ]:
                    a = y_pred[dst] - y_pred[src]
                    l2d = torch.norm(a, dim=0)
                    l3d = torch.norm(y_pred3d[dst] - y_pred3d[src], dim=0)
                    lcos = self.cos[i]

                    lcos_loc = (l2d / l3d - lcos) ** 2
                    Lcos = Lcos + lcos_loc
                    lpar_loc = ((a / l2d) * n).sum() ** 2
                    Lpar = Lpar + lpar_loc
            else:
                ltan_loc = ((c * t).sum()) ** 2
                Ltan = Ltan + ltan_loc
                lpar_loc = (c * n).sum() ** 2
                Lpar = Lpar + lpar_loc

            if writer is not None:
                writer.add_scalar(f"tan/{bone_name}", ltan_loc, global_step=global_step)
                writer.add_scalar(f"cos/{bone_name}", lcos_loc, global_step=global_step)
                writer.add_scalar(f"par/{bone_name}", lpar_loc, global_step=global_step)

            Lcount += 1

        if Lcount > 0:
            Ltan = Ltan / Lcount
            Lcos = Lcos / Lcount
            Lpar = Lpar / Lcount
            Lspine = Lspine / Lcount

        Lgr = Lcount = 0
        for (src, dst), value in self.ground_parallel:
            bone = y_pred[dst] - y_pred[src]
            bone = nn.functional.normalize(bone, dim=0)
            l = (torch.abs(bone[0]) - value) ** 2

            Lgr = Lgr + l
            Lcount += 1

        if Lcount > 0:
            Lgr = Lgr / Lcount

        Lstraight3d = Lcount = 0
        for (i, j), (k, l) in self.parallel_in_3d:
            a = z[j] - z[i]
            a = nn.functional.normalize(a, dim=0)
            b = z[l] - z[k]
            b = nn.functional.normalize(b, dim=0)
            lo = (((a * b).sum() - 1) ** 2).sum()
            Lstraight3d = Lstraight3d + lo
            Lcount += 1

            b = y_data[1] - y_data[8]
            b = nn.functional.normalize(b, dim=0)

        if Lcount > 0:
            Lstraight3d = Lstraight3d / Lcount

        return Ltan, Lcos, Lpar, Lspine, Lgr, Lstraight3d, Lcon2d


class MimickedSelfContactLoss(nn.Module):
    def __init__(self, geodesics_mask):
        super().__init__()
        """
        Loss that lets vertices in contact on presented mesh attract vertices that are close.
        """
        # geodesic distance mask
        self.register_buffer("geomask", geodesics_mask)

    def forward(
        self,
        presented_contact,
        vertices,
        v2v=None,
        contact_mode="dist_tanh",
        contact_thresh=1,
    ):

        contactloss = 0.0

        if v2v is None:
            # compute pairwise distances
            verts = vertices.contiguous()
            nv = verts.shape[1]
            v2v = verts.squeeze().unsqueeze(1).expand(
                nv, nv, 3
            ) - verts.squeeze().unsqueeze(0).expand(nv, nv, 3)
            v2v = torch.norm(v2v, 2, 2)

        # loss for self-contact from mimic'ed pose
        if len(presented_contact) > 0:
            # without geodesic distance mask, compute distances
            # between each pair of verts in contact
            with torch.no_grad():
                cvertstobody = v2v[presented_contact, :]
                cvertstobody = cvertstobody[:, presented_contact]
                maskgeo = self.geomask[presented_contact, :]
                maskgeo = maskgeo[:, presented_contact]
                weights = torch.ones_like(cvertstobody).to(verts.device)
                weights[~maskgeo] = float("inf")
                min_idx = torch.min((cvertstobody + 1) * weights, 1)[1]
                min_idx = presented_contact[min_idx.cpu().numpy()]

            v2v_min = v2v[presented_contact, min_idx]

            # tanh will not pull vertices that are ~more than contact_thres far apart
            if contact_mode == "dist_tanh":
                contactloss = contact_thresh * torch.tanh(v2v_min / contact_thresh)
                contactloss = contactloss.mean()
            else:
                contactloss = v2v_min.mean()

        return contactloss
