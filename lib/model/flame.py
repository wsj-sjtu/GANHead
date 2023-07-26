import torch
import hydra
import numpy as np

from lib.flame.flame_deca.FLAME import FLAME

class FLAMEServer(torch.nn.Module):

    def __init__(self, gender='male', betas=None):
        super().__init__()

        self.flame = FLAME()

        self.prev_input = None
        self.prev_output = None

        self.bone_parents = self.flame.bone_parents.astype(int)
        self.bone_parents[0] = -1

        self.bone_ids = []
        for i in range(5): self.bone_ids.append([self.bone_parents[i], i])

        param_canonical = torch.zeros((1, 169),dtype=torch.float32)
        param_canonical[0, 0] = 1
        param_canonical[0, 10] = 0.2  # openmouth slightly

        # if betas is not None:
        #     param_canonical[0,-10:] = betas
        self.param_canonical = param_canonical
        flame_output = self.forward(param_canonical, absolute=True)

        self.verts_c = flame_output['flame_verts']
        self.joints_c = flame_output['flame_jnts']
        self.tfs_c = flame_output['flame_tfs']
        self.tfs_c_inv = self.tfs_c.squeeze(0).inverse()
        self.weights_c = flame_output['flame_weights']

        param_canonical_deshaped = param_canonical.detach().clone()
        param_canonical_deshaped[0,-150:-50] = 0
        flame_output_deshaped = self.forward(param_canonical_deshaped, absolute=True)
        self.verts_c_deshaped = flame_output_deshaped['flame_verts']
        self.joints_c_deshaped = flame_output_deshaped['flame_jnts']
        self.tfs_c_deshaped = flame_output_deshaped['flame_tfs']

    def forward(self, flame_params, displacement=None, v_template=None, absolute=False):
        """return flame output from params

        Args:
            flame_params [B, 60]: flame parameters scale(1), transl(3), pose(15), shape(100), exp(50)
            displacement [B, 5023] (optional): per vertex displacement to represent wrinkle and hair. Defaults to None.

        Returns:
            verts: vertices [B,5023]
            tf_mats: bone transformations [B,5,4,4]
            weights: lbs weights [B,5]
        """

        if flame_params.shape[1] == 160:    # pose without eye pose and neck pose
            scale, transl, pose, betas, exp = torch.split(flame_params, [1, 3, 6, 100, 50], dim=1)
        elif flame_params.shape[1] == 169:
            scale, transl, pose, betas, exp = torch.split(flame_params, [1, 3, 15, 100, 50], dim=1)
        else:
            raise ValueError('Incorrect flame_params dimension!!!')

        if v_template is not None:
            betas = 0*betas

        flame_output = self.flame.forward(shape_params=betas,
                                          expression_params=exp, 
                                          pose_params=pose)

        output = {}

        verts = flame_output.vertices.clone()
        verts = verts * (scale.unsqueeze(1)) + transl.unsqueeze(1)

        tf_mats = flame_output.T.clone()
        tf_mats[:, :, :3, :] *= scale.unsqueeze(1).unsqueeze(1)
        tf_mats[:, :, :3, 3] += transl.unsqueeze(1)

        if not absolute:
            param_canonical = self.param_canonical.expand(flame_params.shape[0], -1).clone()
            param_canonical[:,-150:-50] = flame_params[:,-150:-50] 
            output_cano = self.forward(param_canonical.type_as(betas), v_template=v_template, absolute=True)

            output_cano = { k+'_cano': v for k, v in output_cano.items() }
            output.update(output_cano)


        joints = flame_output.joints.clone()
        joints = joints * scale.unsqueeze(1) + transl.unsqueeze(1)

        output.update({'flame_verts': verts.float(),
                        'flame_tfs': tf_mats,
                        'flame_weights': flame_output.weights.float(),
                        'flame_jnts': joints.float(),
                        'flame_pose_feature': flame_output.pose_feature.clone()
                        })
        return output