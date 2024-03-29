import os

import hydra
import torch
import wandb
import imageio
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import itertools

import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import cv2

from lib.model.flame import FLAMEServer
from lib.model.mesh import generate_mesh
from lib.model.sample import PointOnBones
from lib.model.generator import Generator
from lib.model.network import ImplicitNetwork
from lib.model.helpers import expand_cond, vis_images
from lib.utils.render import render_mesh_dict, weights2colors
from lib.model.deformer import skinning
from lib.model.ray_tracing import DepthModule
from lib.flame.flame_deca.FLAME import FLAME
from lib.model.losses import FlameLoss

class BaseModel(pl.LightningModule):

    def __init__(self, opt, meta_info, data_processor=None):
        super().__init__()

        self.flame = FLAME()
       
        self.opt = opt

        self.network = ImplicitNetwork(**opt.network)
        print(self.network)

        self.deformer = hydra.utils.instantiate(opt.deformer, opt.deformer)
        print(self.deformer)

        self.generator = Generator(opt.dim_shape)
        print(self.generator)

        self.flame_server = FLAMEServer(gender='neutral')

        self.sampler_bone = PointOnBones(self.flame_server.bone_ids)

        self.z_shapes = torch.nn.Embedding(meta_info.n_samples, opt.dim_shape)
        self.z_shapes.weight.data.fill_(0)

        self.z_details = torch.nn.Embedding(meta_info.n_samples, opt.dim_detail)
        self.z_details.weight.data.fill_(0)

        self.z_colors = torch.nn.Embedding(meta_info.n_samples, opt.dim_color)
        self.z_colors.weight.data.fill_(0)

        self.data_processor = data_processor

        self.flame_loss = FlameLoss()

        if opt.stage=='fine':
            self.norm_network = ImplicitNetwork(**opt.norm_network)
            print(self.norm_network)
            self.tex_network = ImplicitNetwork(**opt.tex_network)
            print(self.tex_network)

            if opt.use_gan:
                from lib.model.losses import GANLoss
                self.gan_loss = GANLoss(self.opt)
                print(self.gan_loss.discriminator)
            
            if opt.use_gan_color:
                from lib.model.losses import GANLoss
                self.gan_loss_color = GANLoss(self.opt)
                print(self.gan_loss_color.discriminator)

        self.render = DepthModule(**self.opt.ray_tracer)


    def configure_optimizers(self):

        grouped_parameters = self.parameters()
        
        def is_included(n): 
            if self.opt.stage =='fine':  # only train the z_details, z_colors, texture network, and normal network in the second stage
                if 'z_details' not in n and 'norm_network' not in n and 'z_colors' not in n and 'tex_network' not in n:
                    return False

            return True

        grouped_parameters = [
            {"params": [p for n, p in list(self.named_parameters()) if is_included(n)], 
            'lr': self.opt.optim.lr, 
            'betas':(0.9,0.999)},
        ]

        optimizer = torch.optim.Adam(grouped_parameters, lr=self.opt.optim.lr)

        if not self.opt.use_gan and not self.opt.use_gan_color:
            return optimizer
        elif self.opt.use_gan and not self.opt.use_gan_color:   # only use gan loss for normal
            optimizer_d = torch.optim.Adam(self.gan_loss.parameters(), 
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d
        elif not self.opt.use_gan and self.opt.use_gan_color:   # only use gan loss for color
            optimizer_d = torch.optim.Adam(self.gan_loss_color.parameters(), 
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d
        else:
            optimizer_d = torch.optim.Adam(itertools.chain(self.gan_loss.parameters(), self.gan_loss_color.parameters()),
                                            lr=self.opt.optim.lr_dis,
                                            betas=(0,0.99))
            return optimizer, optimizer_d

    def forward(self, pts_d, flame_tfs, flame_verts, cond, canonical=False, canonical_shape=False, eval_mode=True, fine=False, mask=None, only_near_flame=False):

        n_batch, n_points, n_dim = pts_d.shape

        outputs = {}        

        if mask is None:
            mask = torch.ones( (n_batch, n_points), device=pts_d.device, dtype=torch.bool)

        # Filter based on flame
        if only_near_flame:
            from kaolin.metrics.pointcloud import sided_distance

            flame_verts_scaled = flame_verts
            distance, _ = sided_distance(pts_d, flame_verts_scaled[:,::10])
            mask = mask & (distance<0.11)

        if not mask.any(): 
            return {'occ': -1000*torch.ones( (n_batch, n_points, 1), device=pts_d.device)}

        if canonical_shape:  # Input shape natural canonical points
            pts_c = pts_d 

            occ_pd, feat_pd = self.network(    # geometry network
                                    pts_c, 
                                    cond={'latent': cond['latent'],
                                         'thetas': cond['thetas']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)
        elif canonical:    # Input canonical points
            pts_c = self.deformer.query_cano(pts_d,          # Given canonical (with betas) point return its correspondence in the shape neutral space
                                            {'betas': cond['betas']}, 
                                            mask=mask)

            occ_pd, feat_pd = self.network(
                                    pts_c, 
                                    cond={'latent': cond['latent'],
                                         'thetas': cond['thetas']},
                                    mask=mask,
                                    val_pad=-1000,
                                    return_feat=True,
                                    spatial_feat=True,
                                    normalize=True)
        else:    # Input deformed points
            cond_tmp = cond.copy()
            cond_tmp['latent'] = cond['lbs']
            pts_c, others = self.deformer.forward(pts_d,     # Given deformed point return its caonical correspondence
                                        cond_tmp,
                                        flame_tfs,
                                        mask=mask,
                                        eval_mode=eval_mode)

            occ_pd, feat_pd = self.network(
                                        pts_c.reshape((n_batch, -1, n_dim)), 
                                        cond={'latent': cond['latent'],
                                             'thetas': cond['thetas']},
                                        mask=others['valid_ids'].reshape((n_batch, -1)),
                                        val_pad=-1000,
                                        return_feat=True,
                                        spatial_feat=True,
                                        normalize=True)

            occ_pd = occ_pd.reshape(n_batch, n_points, -1, 1)
            feat_pd = feat_pd.reshape(n_batch, n_points, -1, feat_pd.shape[-1])

            occ_pd, idx_c = occ_pd.max(dim=2)

            feat_pd = torch.gather(feat_pd, 2, idx_c.unsqueeze(-1).expand(-1, -1, 1, feat_pd.shape[-1])).squeeze(2)
            pts_c = torch.gather(pts_c, 2, idx_c.unsqueeze(-1).expand(-1,-1, 1, pts_c.shape[-1])).squeeze(2)
            valid_mask = torch.gather(others['valid_ids'], 2, idx_c)   # (bs,2250,1)
            outputs['valid_mask'] = valid_mask


        outputs['occ'] = occ_pd
        outputs['pts_c'] = pts_c
        outputs['weights'], outputs['shapedirs'], outputs['posedirs'] = self.deformer.query_weights(pts_c,  
                                                        cond={
                                                        'betas': cond['betas'],
                                                        'latent': cond['lbs']
                                                        })
        if fine:
            ## normal net
            outputs['norm'], feat_norm = self.norm_network(pts_c, 
                                                           cond={'latent': cond['detail'],
                                                                'thetas': cond['thetas']},
                                                           mask=mask,
                                                           return_feat=True,
                                                           input_feat=feat_pd,
                                                           val_pad=1)

            ## texture net
            condition = torch.cat([cond['color'], cond['thetas']], dim=-1)     # 64+65
            feature = torch.cat([feat_pd, feat_norm], dim=-1)
            outputs['color'] = self.tex_network(pts_c, 
                                                cond={'latent': condition,
                                                    'thetas': cond['thetas']},
                                                mask=mask,
                                                input_feat=feature,
                                                val_pad=1)

            flame_tfs = expand_cond(flame_tfs, pts_c)[mask]   # (4,5,4,4)->(4,65536,5,4,4)

            if not canonical:
                outputs['norm'][mask] = skinning(outputs['norm'][mask], outputs['weights'][mask], flame_tfs, inverse=False, normal=True)

            outputs['norm'][mask] = outputs['norm'][mask] / torch.linalg.norm(outputs['norm'][mask],dim=-1,keepdim=True)

        return outputs


    def forward_2d(self, flame_tfs, flame_verts, cond, eval_mode=True, fine=True, res=256):    

        yv, xv = torch.meshgrid([torch.linspace(-1, 1, res), torch.linspace(-1, 1, res)])
        pix_d = torch.stack([xv, yv], dim=-1).type_as(flame_tfs)
        pix_d = pix_d.reshape(1,res*res,2)

        def occ(x, mask=None):

            outputs = self.forward(x, flame_tfs, flame_verts, cond, eval_mode=eval_mode, mask=mask, fine=False, only_near_flame=True)

            if mask is not None:
                return outputs['occ'][mask].reshape(-1, 1)
            else:
                return outputs['occ']        

        pix_d = torch.stack([pix_d[...,0], -pix_d[...,1], torch.zeros_like(pix_d[...,0]) + 1], dim=-1)

        ray_dirs = torch.zeros_like(pix_d)
        ray_dirs[...,-1] = -1

        ##----render depth map---
        # extract mesh
        mesh_cano = self.extract_mesh(flame_verts, flame_tfs, cond, res_up=4)
        mesh_def  = self.deform_mesh(mesh_cano, flame_tfs, cond)
        mesh_def = trimesh.Trimesh(vertices=mesh_def['verts'].cpu().numpy(), faces=mesh_def['faces'].cpu().numpy())
        mesh = pyrender.Mesh.from_trimesh(mesh_def)
        # create scene
        scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
        # create nodes (camera, light, object) and place them into the scene
        camera = pyrender.OrthographicCamera(xmag=1, ymag=1, znear=0.01, zfar=2.0)
        nc = pyrender.Node(camera=camera, matrix=np.eye(4))
        scene.add_node(nc)
        light = pyrender.PointLight(color=np.ones(3))
        nl = pyrender.Node(light=light, matrix=np.eye(4))
        scene.add_node(nl, parent_node=nc)
        mesh_pose = np.array([    
                        [1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., -1.],
                        [0., 0., 0., 1.],
                    ]) 
        nm = pyrender.Node(mesh=mesh, matrix=mesh_pose)
        scene.add_node(nm, parent_node=nc)
        # render depth map
        r = pyrender.OffscreenRenderer(256, 256)
        _, depth = r.render(scene)
        d = depth.copy()
        d[d==0] = np.inf   # background
        d = torch.Tensor(d).to(ray_dirs.device).reshape(-1, res*res)
        ##---------------------------------------------

        pix_d[...,-1] += d*ray_dirs[...,-1]

        mask = ~d.isinf()

        outputs = self.forward(pix_d, flame_tfs, flame_verts, cond, eval_mode=eval_mode, fine=fine, mask=mask)

        outputs['mask'] = mask

        outputs['pts_c'][~mask, :] = 1

        img = outputs['pts_c'].reshape(res,res,3).data.cpu().numpy()
        mask = outputs['mask'].reshape(res,res,1).data.cpu().numpy()

        img_mask = np.concatenate([img,mask],axis=-1)

        return img_mask


    ## for rendering results
    def rendering_results(self, flame_tfs, flame_verts, cond, eval_mode=True, fine=True, res=256):

        mesh_cano = self.extract_mesh(flame_verts, flame_tfs, cond, res_up=4)   # canonical的mesh（根据training_step_fine函数中的第一个forward输入canonical_shape=True猜想这里应该用的就是canonical mesh的深度图）
        mesh_def  = self.deform_mesh(mesh_cano, flame_tfs, cond)

        return mesh_cano, mesh_def



    def prepare_cond(self, batch):

        cond = {}

        cond['betas'] = batch['flame_params'][:,-150:-50]     # body size    shape+exp

        z_shape = batch['z_shape']            
        cond['latent'] = self.generator(z_shape)
        cond['lbs'] = z_shape
        cond['detail'] = batch['z_detail']
        cond['color'] = batch['z_color']

        ## flame pose parameters  (bs,15)
        batch_size = batch['flame_params'].shape[0]
        default_neck_pose = torch.zeros([batch_size,3], dtype=batch['flame_params'].dtype).to(batch['flame_params'].device)
        default_eyball_pose = torch.zeros([batch_size,6], dtype=batch['flame_params'].dtype).to(batch['flame_params'].device)
        cond['full_pose'] = torch.cat([batch['flame_params'][:,4:7], default_neck_pose, batch['flame_params'][:,7:10], default_eyball_pose], dim=1) 

        ## pose_feature
        cond['pose_feature'] = batch['flame_pose_feature']

        ## expression parameters  (bs,50)
        cond['exp'] = batch['flame_params'][:,-50:]
        
        ## id/shape parameters  (bs,100)
        cond['shape'] = batch['flame_params'][:,-150:-50]

        ## pose and expression parameters   (bs,15+50)
        cond['thetas'] = torch.cat([batch['flame_params'][:,4:19], cond['exp']], dim=1)  # (bs,65)

        return cond
    

    def training_step_coarse(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0

        reg_shape = F.mse_loss(batch['z_shape'], torch.zeros_like(batch['z_shape']))
        self.log('reg_shape', reg_shape)
        loss = loss + self.opt.lambda_reg * reg_shape
        
        reg_lbs = F.mse_loss(cond['lbs'], torch.zeros_like(cond['lbs']))
        self.log('reg_lbs', reg_lbs)
        loss = loss + self.opt.lambda_reg * reg_lbs

        # occupancy loss
        outputs = self.forward(batch['pts_d'], batch['flame_tfs'],  batch['flame_verts'], cond, eval_mode=False, only_near_flame=False)
        loss_bce = F.binary_cross_entropy_with_logits(outputs['occ'], batch['occ_gt'])
        self.log('train_bce', loss_bce)
        loss = loss + loss_bce

        # lbs loss
        loss_blendshape = self.flame_loss(outputs['shapedirs'], outputs['posedirs'], outputs['weights'], outputs['pts_c'], batch['flame_params'])
        self.log('train_blendshape', loss_blendshape)
        loss = loss + loss_blendshape

        # Bootstrapping
        num_batch = batch['pts_d'].shape[0]
        if self.current_epoch < self.opt.nepochs_pretrain:

            # Bone occupancy loss
            if self.opt.lambda_bone_occ > 0:

                pts_c, _, occ_gt, _ = self.sampler_bone.get_points(self.flame_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                outputs = self.forward(pts_c, None, None, cond, canonical=True, only_near_flame=False)
                loss_bone_occ = F.binary_cross_entropy_with_logits(outputs['occ'], occ_gt.unsqueeze(-1))
                self.log('train_bone_occ', loss_bone_occ)
                loss = loss + self.opt.lambda_bone_occ * loss_bone_occ

            # Joint weight loss
            if self.opt.lambda_bone_w > 0:
                pts_c, w_gt, _ = self.sampler_bone.get_joints(self.flame_server.joints_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1))
                w_pd, _, _ = self.deformer.query_weights(pts_c, {'latent': cond['lbs'], 'betas': cond['betas']*0})
                loss_bone_w = F.mse_loss(w_pd, w_gt)
                self.log('train_bone_w', loss_bone_w)
                loss = loss + self.opt.lambda_bone_w * loss_bone_w

        # deshape loss
        pts_c_gt = self.flame_server.verts_c_deshaped.type_as(batch['pts_d']).expand(num_batch, -1, -1)   # (bs,5023,3)
        pts_c = self.deformer.query_cano(batch['flame_verts_cano'], {'betas': cond['betas']})     # Given canonical (with betas) point return its correspondence in the shape neutral space
        loss_disp = F.mse_loss(pts_c, pts_c_gt)
        self.log('train_disp', loss_disp)
        loss = loss + self.opt.lambda_disp * loss_disp

        return loss

    def training_step_fine(self, batch, batch_idx, optimizer_idx=None):
        
        cond = self.prepare_cond(batch)

        loss = 0
        
        outputs = self.forward(batch['cache_pts'], batch['flame_tfs_img'], None, cond, canonical_shape=True, mask=batch['cache_mask'], fine=True)

        self.gan_loss_input = {
            'real': batch['norm_img'],
            'fake': outputs['norm'].permute(0,2,1).reshape(-1,3,self.opt.img_res,self.opt.img_res)
        }

        self.gan_loss_input_color = {
            'real': batch['color_img'],
            'fake': outputs['color'].permute(0,2,1).reshape(-1,3,self.opt.img_res,self.opt.img_res)
        }

        ## photo loss (2D loss)
        mask_color = ~(self.gan_loss_input_color['real']==1)
        photo_loss_color = torch.nn.functional.mse_loss(self.gan_loss_input_color['fake']*mask_color, self.gan_loss_input_color['real']*mask_color)
        self.log('loss_train/photo_loss_color', photo_loss_color)
        loss += photo_loss_color

        mask = ~(self.gan_loss_input['real']==1)
        photo_loss = torch.nn.functional.mse_loss(self.gan_loss_input['fake']*mask, self.gan_loss_input['real']*mask)
        self.log('loss_train/photo_loss', photo_loss)
        loss += photo_loss


        if batch_idx%10 == 0 and self.trainer.is_global_zero:
            # normal
            img = vis_images(self.gan_loss_input)
            self.logger.experiment.log({"imgs":[wandb.Image(img)]})                  
            save_path = os.path.join(os.getcwd(), 'images')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), (255*img).astype(np.uint8)) 
            # color
            # gan_loss_input_color_vis = self.gan_loss_input_color.copy()
            # gan_loss_input_color_vis['real'] = gan_loss_input_color_vis['real'] * mask_color
            # gan_loss_input_color_vis['fake'] = gan_loss_input_color_vis['fake'] * mask_color
            # img_rgb = vis_images(gan_loss_input_color_vis)     # visualize with mask
            img_rgb = vis_images(self.gan_loss_input_color)
            self.logger.experiment.log({"imgs_rgb":[wandb.Image(img_rgb)]})                  
            imageio.imsave(os.path.join(save_path,'%04d_rgb.png'%self.current_epoch), (255*img_rgb).astype(np.uint8)) 

        ## gan loss on rendered images
        # normal
        if self.opt.use_gan:
            loss_gan, log_dict = self.gan_loss(self.gan_loss_input, self.global_step, optimizer_idx, for_color=False)
            for key, value in log_dict.items(): self.log(key, value)
            loss += self.opt.lambda_gan*loss_gan
        # color  
        if self.opt.use_gan_color:
            loss_gan_color, log_dict_color = self.gan_loss_color(self.gan_loss_input_color, self.global_step, optimizer_idx, for_color=True)
            for key, value in log_dict_color.items(): self.log(key, value)
            loss += self.opt.lambda_gan_color*loss_gan_color


        if optimizer_idx == 0 or (not self.opt.use_gan and not self.opt.use_gan_color):
            
            ## predicted normal vs gt normal loss, predicted color vs gt color loss   (3D loss)
            if self.opt.norm_loss_3d or self.opt.color_loss_3d:
                outputs = self.forward(batch['pts_surf'], batch['flame_tfs'],  batch['flame_verts'], cond, canonical=False, fine=True)

            if self.opt.norm_loss_3d:           
                loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_surf'])).mean() 
            else:
                loss_norm = (1 - torch.einsum('ijk, ijk->ij',outputs['norm'], batch['norm_img'].permute(0,2,3,1).flatten(1,2)))[batch['cache_mask']].mean()    
            self.log('loss_train/train_norm', loss_norm)
            loss += loss_norm

            if self.opt.color_loss_3d:           
                loss_color = torch.sum((outputs['color']-batch['color_surf'])**2, dim=-1)   # (4,2000,3)->(4,2000)
                loss_color = torch.sum(loss_color) / (outputs['color'].shape[0]*outputs['color'].shape[1]) #->(1,)
            else:
                loss_color = 0
            self.log('loss_train/train_color', loss_color)
            loss += self.opt.lambda_color * loss_color

            
            ## regularization trem
            reg_detail = torch.nn.functional.mse_loss(batch['z_detail'], torch.zeros_like(batch['z_detail']))
            self.log('loss_train/reg_detail', reg_detail)
            loss += self.opt.lambda_reg * reg_detail

            reg_color = torch.nn.functional.mse_loss(batch['z_color'], torch.zeros_like(batch['z_color']))
            self.log('loss_train/reg_color', reg_color)
            loss += self.opt.lambda_reg_color * reg_color

        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.flame_server, load_volume=self.opt.stage!='fine')

        batch['z_shape'] = self.z_shapes(batch['index'])
        batch['z_detail'] = self.z_details(batch['index'])
        batch['z_color'] = self.z_colors(batch['index'])

        if not self.opt.stage=='fine':
            loss = self.training_step_coarse(batch, batch_idx)
        else:
            loss = self.training_step_fine(batch, batch_idx, optimizer_idx=optimizer_idx)

        return loss
    
    def validation_step(self, batch, batch_idx):

        # Data prep
        if self.data_processor is not None:
            batch = self.data_processor.process(batch, self.flame_server)

        batch['z_shape'] = self.z_shapes(batch['index'])
        batch['z_detail'] = self.z_details(batch['index'])
        batch['z_color'] = self.z_colors(batch['index'])

        if batch_idx == 0 and self.trainer.is_global_zero:
            with torch.no_grad(): self.plot(batch)   

    def extract_mesh(self, flame_verts, flame_tfs, cond, res_up=3):

        def occ_func(pts_c):
            outputs = self.forward(pts_c, flame_tfs, flame_verts, cond, canonical=True, only_near_flame=False)
            return outputs['occ'].reshape(-1,1)

        mesh = generate_mesh(occ_func, flame_verts.squeeze(0),res_up=res_up)
        mesh = {'verts': torch.tensor(mesh.vertices).type_as(flame_verts), 
                'faces': torch.tensor(mesh.faces, device=flame_verts.device)}

        verts  = mesh['verts'].unsqueeze(0)

        outputs = self.forward(verts, flame_tfs, flame_verts, cond, canonical=True, fine=self.opt.stage=='fine', only_near_flame=False)
        
        mesh['weights'] = outputs['weights'][0].detach()#.clamp(0,1)[0]
        mesh['weights_color'] = torch.tensor(weights2colors(mesh['weights'].data.cpu().numpy()), device=flame_verts.device).float().clamp(0,1)
        mesh['pts_c'] = outputs['pts_c'][0].detach()

        mesh['shapedirs'] = outputs['shapedirs'].detach()
        mesh['posedirs'] = outputs['posedirs'].detach()
        
        if self.opt.stage=='fine':
            mesh['color'] = outputs['color'][0].detach()
            mesh['norm'] = outputs['norm'][0].detach()
        else:
            mesh['color'] = mesh['weights_color'] 

        return mesh

    def deform_mesh(self, mesh, flame_tfs, cond):
        import copy
        # mesh_deform = {key: mesh[key].detach().clone() for key in mesh}
        mesh = copy.deepcopy(mesh)

        flame_tfs = flame_tfs.expand(mesh['verts'].shape[0],-1,-1,-1)
        mesh['verts'] = self.flame.forward_pts(mesh['verts'], mesh['weights'], flame_tfs, cond['shape'], cond['exp'], cond['pose_feature'], shapedirs=mesh['shapedirs'], posedirs=mesh['posedirs'])
        
        if 'norm' in mesh:
            mesh['norm']  = skinning( mesh['norm'], mesh['weights'], flame_tfs, normal=True)
            mesh['norm'] = mesh['norm']/ torch.linalg.norm(mesh['norm'],dim=-1,keepdim=True)
            
        return mesh

    def plot(self, batch):

        # Plot pred surfaces
        for key in batch:
            if type(batch[key]) is list:
                batch[key] = batch[key][0]
            else:
                batch[key] = batch[key][[0]]

        cond = self.prepare_cond(batch)

        surf_pred_cano = self.extract_mesh(batch['flame_verts_cano'], batch['flame_tfs'], cond, res_up=3)
        surf_pred_def  = self.deform_mesh(surf_pred_cano, batch['flame_tfs'], cond)

        img_list = []
        if self.opt.stage == 'fine':
            img_list.append(render_mesh_dict(surf_pred_cano,mode='npt'))
            img_list.append(render_mesh_dict(surf_pred_def,mode='npt'))
        else:
            img_list.append(render_mesh_dict(surf_pred_cano,mode='nps'))
            img_list.append(render_mesh_dict(surf_pred_def,mode='nps'))

        img_all = np.concatenate(img_list, axis=1)

        self.logger.experiment.log({"vis":[wandb.Image(img_all)]})
        
        save_path = 'medias'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        imageio.imsave(os.path.join(save_path,'%04d.png'%self.current_epoch), img_all)        

    def sample_codes(self, n_sample, std_scale=1):
        device = self.z_shapes.weight.device

        mean_shapes = self.z_shapes.weight.data.mean(0)
        std_shapes = self.z_shapes.weight.data.std(0)
        mean_details = self.z_details.weight.data.mean(0)
        std_details = self.z_details.weight.data.std(0)
        mean_colors = self.z_colors.weight.data.mean(0)
        std_colors = self.z_colors.weight.data.std(0)

        z_shape = torch.randn(n_sample, self.opt.dim_shape, device=device)
        z_detail = torch.randn(n_sample, self.opt.dim_detail, device=device) 
        z_color = torch.randn(n_sample, self.opt.dim_color, device=device) 

        z_shape = z_shape*std_shapes*std_scale+mean_shapes
        z_detail = z_detail*std_details*std_scale+mean_details
        z_color = z_detail*std_colors*std_scale+mean_colors

        return z_shape, z_detail, z_color

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)