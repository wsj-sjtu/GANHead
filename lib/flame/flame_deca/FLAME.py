# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F
import hydra
from collections import namedtuple

from .lbs import *
import pytorch3d

## for debug lbs_weight
from lib.utils.render import weights2colors


ModelOutput = namedtuple('ModelOutput',
                         ['vertices','faces', 'joints', 'full_pose', 'betas',
                          'global_orient',
                          'body_pose', 'expression',
                          'left_hand_pose', 'right_hand_pose',
                          'jaw_pose', 'T', 'T_weighted', 'weights', 'pose_feature', 'transl'])
ModelOutput.__new__.__defaults__ = (None,) * len(ModelOutput._fields)



def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)
def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, 
                 flame_model_path='../../lib/flame/flame_model/generic_model.pkl',
                 n_shape=100,
                 n_exp=50,
                 flame_lmk_embedding_path='../../lib/flame/flame_model/landmark_embedding.npy',
                 ):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        factor = 6
        self.factor = factor
        self.bone_parents = to_np(flame_model.kintree_table[0])

        self.dtype = torch.float32
        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long))
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template) * factor, dtype=self.dtype))
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:n_shape], shapedirs[:,:,300:300+n_exp]], 2)
        self.register_buffer('shapedirs', shapedirs * factor)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs) * factor, dtype=self.dtype))
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype))
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype))

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long())
        self.register_buffer('lmk_bary_coords', torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long())
        self.register_buffer('full_lmk_bary_coords', torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype))

        neck_kin_chain = []; NECK_IDX=1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))


        self.canonical_pose = torch.zeros(1, 15).float()
        self.canonical_pose[:, 6] = 0.2     # canonical space: mouth open slightly
        self.canonical_exp = torch.zeros(1, n_exp).float()

        # translate v_template to the origin
        self.t = -(self.v_template[:,2].max() + self.v_template[:,2].min()) / 2
        self.v_template[:,2] = self.v_template[:,2] + self.t

        
    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:dd2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                       self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                       self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = expression_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        betas = torch.cat([shape_params, expression_params], dim=1)
        if pose_params.shape[-1]==15:
            full_pose = pose_params
        elif pose_params.shape[-1]==6:
            full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        else:
            raise ValueError('Incorrect pose_params dim!!!')
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)

        vertices, joints_flame, T_weighted, W, T, pose_feature = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        default_betas = torch.zeros([1,100], dtype=self.dtype)     

        output = ModelOutput(vertices=vertices,
                        faces=self.faces_tensor,
                        global_orient=pose_params[:,0:3],
                        body_pose=pose_params[:,3:6],
                        joints=joints_flame,
                        betas=default_betas,
                        full_pose=pose_params,
                        T=T, T_weighted=T_weighted, weights=W,
                        pose_feature=pose_feature,
                        transl=self.t)

        return output
    

    def forward_pts(self, pts_c, lbs_weights, tfs, shape_params, exp_params, pose_feature, shapedirs=None, posedirs=None, mask=None, dtype=torch.float32):
        """
        Input:
            pts_c: (bs*n*n_init, 3)
            lbs_weight: (bs*n*n_init, 5)
            tfs: (bs*n*n_init, 5, 4, 4)
            shape_params: (bs, 100)
            exp_params: (bs, 50)
            pose_feature: (bs, 36) 
        Output:
            pts_d: deformed points, (bs*n*n_init, 3)
        """
        assert len(pts_c.shape) == 2
        num_points = pts_c.shape[0]
        batch_size = shape_params.shape[0]
        n = num_points // batch_size
        device = shape_params.device

        # forward flame
        flame_cano_output = self.forward(shape_params=shape_params,
                                        expression_params=self.canonical_exp.expand(batch_size,-1).to(device),
                                        pose_params=self.canonical_pose.expand(batch_size,-1).to(device))
        flame_cano_verts = flame_cano_output.vertices.clone()
        flame_cano_tfs = flame_cano_output.T.clone().unsqueeze(1).expand(-1, n, -1, -1, -1).flatten(0, 1)      # (bs*n*n_init, 5, 4, 4)
        flame_cano_posefeature = flame_cano_output.pose_feature.clone().unsqueeze(1).expand(-1, n, -1).flatten(0, 1)     # (bs*n*n_init, 36)

        if shapedirs is None or posedirs is None:
            shapedirs, posedirs = self.sample_bases(pts_c, shape_params, flame_cano_verts)

        # expand
        canonical_exp = self.canonical_exp.expand(num_points, -1).to(device)
        exp_params = exp_params.unsqueeze(1).expand(-1, n, -1).flatten(0, 1)    # (bs, 50) -> (bs,1,50) -> (bs,n,50) -> (bs*n,50)
        pose_feature = pose_feature.unsqueeze(1).expand(-1, n, -1).flatten(0, 1)

        if mask is not None:
            pts_c = pts_c[mask]
            lbs_weights = lbs_weights[mask]
            tfs = tfs[mask]
            flame_cano_tfs = flame_cano_tfs[mask]
            flame_cano_posefeature = flame_cano_posefeature[mask]
            shapedirs = shapedirs[mask]
            posedirs = posedirs[mask]
            canonical_exp = canonical_exp[mask]
            exp_params = exp_params[mask]
            pose_feature = pose_feature[mask]

        shapedirs_exp = shapedirs[..., -50:]
        pts_c_original = inverse_pts(pts_c, 
                                    canonical_exp, flame_cano_tfs, flame_cano_posefeature, 
                                    shapedirs_exp, posedirs, 
                                    lbs_weights, dtype=dtype)

        # add expression and pose
        pts_d = forward_pts(pts_c_original, 
                            exp_params, tfs, pose_feature, 
                            shapedirs_exp, posedirs,
                            lbs_weights, dtype=dtype)
        
        return pts_d

    
    def sample_bases(self, pts_c, shape_params, flame_cano_verts):
        """
        Input:
            pts_c: (bs*n*n_init, 3)
            shape_params: (bs, 100)

        Output:
            shapedirs: (bs*n*n_init, 3, 150) 
            posedirs: (bs*n*n_init, 36, 3)
        """
        batch_size = shape_params.shape[0]
        pts_c = pts_c.reshape(batch_size,-1,3)
        
        flame_distance, index_batch, _ = pytorch3d.ops.knn_points(pts_c, flame_cano_verts, K=1, return_nn=True)
        index_batch = index_batch.squeeze(-1)

        shapedirs = self.shapedirs[index_batch].flatten(0,1)
        posedirs = self.posedirs.reshape(36,5023,3).permute(1,0,2)[index_batch].flatten(0,1)    # self.posedirs: (36,15069)    posedirs: (bs,n*n_init,36,3)

        return shapedirs, posedirs

    
    def inverse_skinning_pts(self, pts_p, lbs_weights, tfs, shape_params, dtype=torch.float32):
        num_points = pts_p.shape[0]
        batch_size = shape_params.shape[0]
        n = num_points // batch_size
        device = shape_params.device

        flame_cano_output = self.forward(shape_params=shape_params,
                                expression_params=self.canonical_exp.expand(batch_size,-1).to(device),
                                pose_params=self.canonical_pose.expand(batch_size,-1).to(device))
        flame_cano_tfs = flame_cano_output.T.clone().unsqueeze(1).expand(-1, n, -1, -1, -1).flatten(0, 1)      # (bs*n*n_init, 5, 4, 4)

        pts_c_original = inverse_skinning_pts(pts_p, tfs, lbs_weights, dtype=dtype)
        pnts_c = forward_skinning_pts(pts_c_original, flame_cano_tfs, lbs_weights, dtype=dtype, mask=None)
        
        return pnts_c
