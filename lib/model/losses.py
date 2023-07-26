""" The code is based on https://github.com/apple/ml-gsn/ with adaption. """

import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
from lib.model.discriminator import StyleDiscriminator

from lib.flame.flame_deca.FLAME import FLAME
import pytorch3d

def hinge_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.relu(1.0 + fake_pred))
        d_loss_real = torch.mean(F.relu(1.0 - real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = -torch.mean(fake_pred)
    return d_loss

def logistic_loss(fake_pred, real_pred, mode):
    if mode == 'd':
        # Discriminator update
        d_loss_fake = torch.mean(F.softplus(fake_pred))
        d_loss_real = torch.mean(F.softplus(-real_pred))
        d_loss = d_loss_fake + d_loss_real
    elif mode == 'g':
        # Generator update
        d_loss = torch.mean(F.softplus(-fake_pred))
    return d_loss


def r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


class GANLoss(nn.Module):
    def __init__(
        self,
        opt,
        disc_loss='logistic',
    ):
        super().__init__()
        self.opt = opt

        input_dim = 3
        self.discriminator = StyleDiscriminator(input_dim, self.opt.img_res)

        if disc_loss == 'hinge':
            self.disc_loss = hinge_loss
        elif disc_loss == 'logistic':
            self.disc_loss = logistic_loss

    def forward(self, input, global_step, optimizer_idx, for_color=False):

        lambda_gan = self.opt.lambda_gan_color if for_color else self.opt.lambda_gan

        disc_in_real = input['real']
        disc_in_fake = input['fake']

        disc_in_real.requires_grad = True  # for R1 gradient penalty

        if optimizer_idx == 0:  # optimize generator
            loss = 0
            log = {}
            if lambda_gan > 0:
                logits_fake = self.discriminator(disc_in_fake)
                g_loss = self.disc_loss(logits_fake, None, mode='g')
                if for_color:
                    log["loss_train/g_loss_color"] = g_loss.detach()
                else:
                    log["loss_train/g_loss"] = g_loss.detach()
                loss += g_loss

            return loss, log

        if optimizer_idx == 1 :  # optimize discriminator
            logits_real = self.discriminator(disc_in_real)
            logits_fake = self.discriminator(disc_in_fake.detach().clone())

            disc_loss = self.disc_loss(fake_pred=logits_fake, real_pred=logits_real, mode='d')

            # lazy regularization so we don't need to compute grad penalty every iteration
            if (global_step % self.opt.d_reg_every == 0) and self.opt.lambda_grad > 0:
                grad_penalty = r1_loss(logits_real, disc_in_real)

                # the 0 * logits_real is to trigger DDP allgather
                # https://github.com/rosinality/stylegan2-pytorch/issues/76
                grad_penalty = self.opt.lambda_grad / 2 * grad_penalty * self.opt.d_reg_every + (0 * logits_real.sum())
            else:
                grad_penalty = torch.tensor(0.0).type_as(disc_loss)

            d_loss = disc_loss + grad_penalty #+ disc_recon_loss 

            if for_color:
                log = {
                    "loss_train/disc_loss_color": disc_loss.detach(),
                    "loss_train/r1_loss_color": grad_penalty.detach(),
                    "loss_train/logits_real_color": logits_real.mean().detach(),
                    "loss_train/logits_fake_color": logits_fake.mean().detach(),
                }
            else:
                log = {
                    "loss_train/disc_loss": disc_loss.detach(),
                    "loss_train/r1_loss": grad_penalty.detach(),
                    "loss_train/logits_real": logits_real.mean().detach(),
                    "loss_train/logits_fake": logits_fake.mean().detach(),
                }

            return d_loss*lambda_gan, log


class FlameLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.flame = FLAME()

        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')


    def sample_bases(self, pts_c, flame_params):
        """
        Input:
            pts_c: (bs*n*n_init, 3)
            flame_params: (bs, 169)

        Output:
            shapedirs: (bs*n*n_init, 3, 150)
            posedirs: (bs*n*n_init, 36, 3)
        """
        shape_params = flame_params[:,-150:-50]

        batch_size = shape_params.shape[0]
        device = shape_params.device
        pts_c = pts_c.reshape(batch_size,-1,3)

        # flame_cano_verts
        flame_cano_verts = self.flame.forward(shape_params=shape_params,
                                    expression_params=self.flame.canonical_exp.expand(batch_size,-1).to(device),
                                    pose_params=self.flame.canonical_pose.expand(batch_size,-1).to(device)).vertices.clone()

        # Find the nearest neighbor of each point of the model output (pts_c) to its corresponding flame model vertex.
        flame_distance, index_batch, _ = pytorch3d.ops.knn_points(pts_c, flame_cano_verts, K=1, return_nn=True)
        index_batch = index_batch.squeeze(-1)
        flame_distance = flame_distance.squeeze(-1).flatten(0,1)

        # sample flame bases according to the nearest neighbors
        shapedirs_gt = self.flame.shapedirs[index_batch].flatten(0,1)
        shapedirs_gt = shapedirs_gt[...,-50:]    # only use the expression bases
        posedirs_gt = self.flame.posedirs.reshape(36,5023,3).permute(1,0,2)[index_batch].flatten(0,1)    # self.posedirs: (36,15069)    posedirs: (bs,n*n_init,36,3)

        lbs_weights_gt = self.flame.lbs_weights[index_batch].flatten(0,1)

        return shapedirs_gt, posedirs_gt, lbs_weights_gt, flame_distance


    def get_blendshape_loss(self, shapedirs_gt, posedirs_gt, lbs_weights_gt, shapedirs, posedirs, lbs_weights, flame_distance):

        flame_distance_mask = flame_distance < 0.1
        if flame_distance_mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()

        shapedirs = shapedirs[flame_distance_mask]
        shapedirs_gt = shapedirs_gt[flame_distance_mask]
        loss = self.l2_loss(shapedirs, shapedirs_gt)/ float(shapedirs.shape[0]) *100

        posedirs = posedirs[flame_distance_mask]
        posedirs_gt = posedirs_gt[flame_distance_mask]
        loss += self.l2_loss(posedirs, posedirs_gt)/ float(posedirs.shape[0]) *100

        lbs_weights = lbs_weights[flame_distance_mask]
        lbs_weights_gt = lbs_weights_gt[flame_distance_mask]
        loss += self.l2_loss(lbs_weights, lbs_weights_gt)/ float(lbs_weights.shape[0]) *0.2

        return loss


    def forward(self, shapedirs, posedirs, lbs_weights, pts_c, flame_params):
        ## sample flame bases to obtain plausible ground truth bases
        shapedirs_gt, posedirs_gt, lbs_weights_gt, flame_distance = self.sample_bases(pts_c, flame_params)

        ## compute losses
        lbs_weights = lbs_weights.flatten(0,1)
        loss = self.get_blendshape_loss(shapedirs_gt, posedirs_gt, lbs_weights_gt,
                                        shapedirs, posedirs, lbs_weights, flame_distance)

        return loss