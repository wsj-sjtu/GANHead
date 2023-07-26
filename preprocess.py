import os
import torch
import kaolin
import pandas
import imageio
import numpy as np
from tqdm import tqdm
import csv

from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import TexturesVertex

import sys
sys.path.append('.')

from lib.utils.render import render_pytorch3d, Renderer
from lib.utils.uv import Uv2Attr
from lib.flame.flame_deca.FLAME import FLAME


class ScanProcessor():

    def __init__(self, args):

        self.scan_folder  = args.scan_folder
        self.flame_folder = args.flame_folder

        self.scan_list = sorted(os.listdir(self.scan_folder))

        self.output_folder = args.output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.flame = FLAME(flame_model_path='lib/flame/flame_model/generic_model.pkl',
                           flame_lmk_embedding_path='lib/flame/flame_model/landmark_embedding.npy')

        self.renderer = Renderer(256)

    def process(self, name):

        batch = {}

        scan_name = name

        scan_path = os.path.join(self.scan_folder,scan_name, scan_name+'.obj')

        output_folder = os.path.join(self.output_folder, scan_name)
        if not os.path.exists(output_folder): os.makedirs(output_folder)

        batch['scan_name'] = scan_name

        pickle_path = os.path.join(self.flame_folder, scan_name+'_flame.pkl')
        file = pandas.read_pickle(pickle_path)

        ## flame_param: (1,169)
        flame_param = np.concatenate([np.ones( (1,1)),   # scale  (1,1)
                                    np.zeros( (1,3)),    # transl (1,3)
                                    file['pose'].reshape(1,-1),      # (1,15)    
                                    file['betas'].reshape(1,-1)], axis=1)[0]  # (1,150)
        batch['flame_params'] = flame_param

        scan_verts, scan_faces, aux = load_obj(scan_path, 
                                                device=torch.device("cuda:0"),
                                                load_textures=True)

        scan_faces = scan_faces.verts_idx.long()


        ## load texture
        scan_uvs = aux.verts_uvs     # uv_coords
        uv_img = aux.texture_images[batch['scan_name']].to(scan_uvs.device)
        uv_img = torch.flip(uv_img, (0,))
        uv_size = uv_img.shape[0]
        uv_reader = Uv2Attr(torch.round(scan_uvs.unsqueeze(0) * uv_size), size=uv_size)
        scan_colors = uv_reader(uv_img.unsqueeze(0).permute(0,3,1,2), bilinear=True).permute(0, 2, 1).contiguous()
        batch['scan_colors'] = scan_colors[0].cpu().numpy()


        ## preprocess faceverse scans
        # flip and translate to origin
        scan_verts[:,1] = -scan_verts[:,1]
        scan_verts[:,2] = -scan_verts[:,2]
        t = (scan_verts[:,2].max() + scan_verts[:,2].min()) / 2
        scan_verts[:,2] = scan_verts[:,2] - t
        # add scale and transl to align with flame model
        flame_scale = torch.Tensor(file['scale']).to(scan_verts.device)
        flame_transl = torch.Tensor(file['transl']).to(scan_verts.device)
        flame_t = torch.zeros_like(flame_transl)    
        flame_t[:,2] = self.flame.t
        scan_verts = self.flame.factor / flame_scale * (scan_verts - flame_transl) + flame_t

        batch['scan_verts'] = scan_verts.data.cpu().numpy()
        batch['scan_faces'] = scan_faces.data.cpu().numpy()


        ## sample points from the scan surface
        d = scan_verts[:,2].max()/2
        scan_face_idx = scan_verts[:,2] > d     # rough facial area indexes
        scan_verts_face = scan_verts[scan_face_idx]
        scan_verts_other = scan_verts[~scan_face_idx]

        num_verts_face, num_dim = scan_verts_face.shape
        num_verts_other, _ = scan_verts_other.shape
        random_idx_face = torch.randint(0, num_verts_face, [70000, 1], device=scan_verts.device)   # 70000 points in the facial area
        random_idx_other = torch.randint(0, num_verts_other, [30000, 1], device=scan_verts.device)   # 30000 points in other areas of the head

        pts_surf_face = torch.gather(scan_verts_face, 0, random_idx_face.expand(-1, num_dim))
        pts_surf_other = torch.gather(scan_verts_other, 0, random_idx_other.expand(-1, num_dim))
        pts_surf = torch.cat([pts_surf_face, pts_surf_other], dim=0)    # (100000,3)

        pts_surf += 0.01 * torch.randn(pts_surf.shape, device=scan_verts.device)
        pts_bbox = torch.rand(pts_surf.shape, device=scan_verts.device) * 2 - 1
        pts_d = torch.cat([pts_surf, pts_bbox],dim=0)
        occ_gt = kaolin.ops.mesh.check_sign(scan_verts[None], scan_faces, pts_d[None]).float().unsqueeze(-1)
        
        batch['pts_d'] = pts_d.data.cpu().numpy()
        batch['occ_gt'] = occ_gt[0].data.cpu().numpy()

        np.savez(os.path.join(output_folder, 'occupancy.npz'), **batch)


        ## get surface normals and colors
        texture = TexturesVertex(verts_features=scan_colors) 
        meshes = Meshes(verts=[scan_verts], faces=[scan_faces], textures=texture)
        verts, normals, colors = sample_points_from_meshes(meshes, num_samples=100000, return_textures=True, return_normals=True)

        batch_surf = {}
        batch_surf['surface_points'] = verts[0].data.cpu().numpy()
        batch_surf['surface_normals'] = normals[0].data.cpu().numpy()
        batch_surf['surface_colors'] = colors[0].data.cpu().numpy()

        # normalize colors
        cmin = batch_surf['surface_colors'].min()
        cmax = batch_surf['surface_colors'].max()
        batch_surf['surface_colors'] = (batch_surf['surface_colors']-cmin) / (cmax - cmin)
        batch_surf['surface_colors'] = (batch_surf['surface_colors'] - 0.5) / 0.5

        np.savez(os.path.join(output_folder, 'surface.npz'), **batch_surf)


        ## get multiview 2D normal maps and RGB images
        n_views = 9

        output_image_folder = os.path.join(output_folder, 'multi_view_256')
        if not os.path.exists(output_image_folder): os.makedirs(output_image_folder)

        for i in range(n_views):

            ## rotate and render mesh
            flame_params = flame_param.copy()
            flame_params[4:7] = np.array([0,2*np.pi/n_views*i,0])
            flame_param_tensor = torch.Tensor(flame_params).unsqueeze(0)
            flame_outputs = self.flame.forward(shape_params=flame_param_tensor[:,-150:-50],
                                        expression_params=flame_param_tensor[:,-50:], 
                                        pose_params=flame_param_tensor[:,4:19])
            flame_tfs = flame_outputs.T.clone()[0]

            # rotate mesh
            texture_new = texture
            v_h = torch.ones_like(scan_verts[:,0:1])
            v_h = torch.cat([scan_verts, v_h], dim=-1)
            v_r = torch.einsum('ij,nj->ni',flame_tfs[0].cuda().float(),v_h)[:,0:3]
            meshes_new = Meshes(verts=[v_r], faces=[scan_faces], textures=texture_new)

            # render normal image
            image = render_pytorch3d(meshes_new, mode='n', renderer_new=self.renderer)
            imageio.imwrite(os.path.join(output_image_folder, '%04d_normal.png'%i), image)
            # render rgb image
            image_rgb = render_pytorch3d(meshes_new, mode='t', renderer_new=self.renderer)
            imageio.imwrite(os.path.join(output_image_folder, '%04d_rgb.png'%i), image_rgb)

        return 


def split(a, n):
    k, m = divmod(len(a), n)
    return [ a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n) ]
    

if __name__ == '__main__':


    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--scan_folder', type=str, default="data/faceverse/faceverse_dataset/", help="Folder of raw scans.")
    parser.add_argument('--flame_folder', type=str, default="data/faceverse/flame_params/", help="Folder of fitted flame parameters of raw scans.")
    parser.add_argument('--output_folder', type=str, default="data/faceverse/faceverse_processed/", help="Output folder.")
    parser.add_argument('--id', type=int, default=0)
    parser.add_argument('--tot', type=int, default=1)

    args = parser.parse_args()

    processor = ScanProcessor(args)

    task = split(list(range(len(processor.scan_list))) , args.tot)[args.id]

    name_list = os.listdir(processor.scan_folder)


    ## preprocess data
    for i in tqdm(task):
        batch = processor.process(name_list[i])


    ## prepare csv file
    data_csv = []
    csv_head = ['id', 'dataset']
    csv_path = './lib/dataset/faceverse.csv'

    if not os.path.exists(csv_path):

        for i in range(len(name_list)):
            data_csv.append({'id':name_list[i], 'dataset':'faceverse'})

        # save csv
        with open(csv_path, 'w', encoding='utf-8', newline='') as file_obj:
            dictWriter = csv.DictWriter(file_obj, csv_head)
            dictWriter.writeheader()
            dictWriter.writerows(data_csv)