
import os
import pytorch_lightning as pl
import hydra
import torch
import numpy as np
from lib.ganhead_model import BaseModel
from tqdm import trange, tqdm
from lib.model.helpers import split,rectify_pose
from lib.dataset.datamodule import DataModule, DataProcessor



@hydra.main(config_path="config", config_name="config")
def main(opt):

    print(opt.pretty())
    pl.seed_everything(42, workers=True)
    torch.set_num_threads(10) 

    datamodule = DataModule(opt.datamodule)
    datamodule.setup(stage='fit')
    meta_info = datamodule.meta_info
    data_processor = DataProcessor(opt.datamodule)

    checkpoint_path = os.path.join('./checkpoints', 'last.ckpt')
    
    model = BaseModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        opt=opt.model, 
        meta_info=meta_info,
        data_processor=data_processor,
    ).cuda()

    # prepare latent codes

    batch_list = []

    output_folder = 'cache_img_dvr_pyrender'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    task = split( list(range( meta_info.n_samples)), opt.agent_tot)[opt.agent_id]

    for index in tqdm(task):


        scan_info = meta_info.scan_info.iloc[index]
        f = np.load(os.path.join(meta_info.dataset_path, scan_info['id'], 'occupancy.npz') )

        batch = {'index': torch.tensor(index).long().cuda().reshape(1),
                'flame_params': torch.tensor(f['flame_params']).float().cuda()[None,:],
                'scan_name': scan_info['id']
                }
        
        batch_list.append(batch)

    with torch.no_grad():

        for i, batch in enumerate(tqdm(batch_list)):

            batch['z_shape'] = model.z_shapes(batch['index'])
            batch['z_detail'] = model.z_details(batch['index'])
            batch['z_color'] = model.z_colors(batch['index'])

            batch_flame = data_processor.process_flame({'flame_params': batch['flame_params']}, model.flame_server)
            batch.update(batch_flame)
            
            cond = model.prepare_cond(batch)
            scan_name = batch['scan_name']

            outputs_list = []
            flame_param_list = []

            n = 9
            for k in trange(n):

                flame_params = batch['flame_params'][0].data.cpu().numpy()
                flame_thetas = rectify_pose(flame_params[4:10], np.array([0,2*np.pi/n*k,0]))
                
                flame_params[4:10] = flame_thetas
                flame_param_list.append(flame_params.copy())

                flame_output = model.flame_server(torch.tensor(flame_params[None]).cuda().float(), absolute=False)

                img_mask = model.forward_2d(flame_output['flame_tfs'], 
                                        flame_output['flame_verts_cano'],
                                        cond, 
                                        # angle_id = k,
                                        eval_mode=True, 
                                        fine=False)

                outputs_list.append(img_mask)

            outputs_all = np.stack(outputs_list, axis=0)
            flame_all = np.stack(flame_param_list, axis=0)

            np.save(os.path.join(output_folder,'%s.npy'%scan_name),outputs_all)
            np.save(os.path.join(output_folder,'%s_pose.npy'%scan_name),flame_all)


if __name__ == '__main__':
    main()