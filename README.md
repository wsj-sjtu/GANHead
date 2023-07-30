# GANHead: Towards Generative Animatable Neural Head Avatars
## [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_GANHead_Towards_Generative_Animatable_Neural_Head_Avatars_CVPR_2023_paper.pdf) | [Project Page](https://wsj-sjtu.github.io/GANHead/) | [Video](https://www.youtube.com/watch?v=Cg0ubzo7DXk)

<img src="assets/teaser.png" /> 

Official implentation of CVPR 2023 paper [*GANHead: Towards Generative Animatable Neural Head Avatars*](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_GANHead_Towards_Generative_Animatable_Neural_Head_Avatars_CVPR_2023_paper.pdf).

## Getting Started
### Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/wsj-sjtu/GANHead.git
cd GANHead
conda env create -f env.yml
conda activate ganhead
python setup.py install
```

2. Download the [FLAME model](https://flame.is.tue.mpg.de/download.php) (FLAME 2020), and put the generic_model.pkl file to `/lib/flame/flame_model/`.

3. Install [kaolin](https://kaolin.readthedocs.io/en/latest/notes/installation.html) for fast occupancy query from meshes:
```
git clone https://github.com/NVIDIAGameWorks/kaolin
cd kaolin
python setup.py develop
```

4. If [PyTorch3D](https://github.com/facebookresearch/pytorch3d) can not be installed correctly, install it manually following their [instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

### Run demos
1. Download pretrained models and test motion sequence (taken from [IMavatar data](https://drive.google.com/file/d/1Hzv41ZkpMK1X9h9Z-B54S-Nn1GcMveb8/view)) from [Google drive](https://drive.google.com/file/d/1ztHGU9GdCoMkAlQB7ynoztIFewVaILlQ/view?usp=drive_link) and organize them according to the fllowing structure:
```
GANHead
│
└─── data
│    └─── yufeng_1812
└─── outputs
     └─── stage1_model
     └─── stage2_model
```




## Training
### Prepare Data
First download [FaceVerse dataset](https://github.com/LizhenWangT/FaceVerse-Dataset) following their instructions, and the fitted FLAME parameters form [Google drive](https://drive.google.com/file/d/1W-r6H573sKW_euG1zjiEgqsSaO_aVvld/view?usp=drive_link). Organize all files into the following structure:
```
GANHead
│
└─── data
     │
     └─── faceverse
          │
          └─── faceverse_dataset
          |    └─── 001_01, 001_02, ...
          └─── flame_params
               └─── 001_01_flame.pkl, 001_02_flame.pkl, ...
```
Next, run the pre-processing script to get ground truth occupancy, surface color and normal, and rendered multiview images and normal maps:
```
python preprocess.py --tot 1 --id 0
```
You can run multiple instances of the script in parallel by simply specifying `--tot` to be the number of total instances and `--id` to be the rank of current instance. For example, divide the task into 4 instances and run the first instance on GPU0: `CUDA_VISIBLE_DEVICES=0 python preprocess.py --tot 4 --id 0`.

### Training
Our model is trained in two stages following the steps below:
1. Stage1: train the geometry network and deformation module.
```
python train.py expname=coarse datamodule=faceverse
```
This step takes around 12 hours on 4 NVIDIA 3090 GPUs.

2. Precompute
```
python precompute.py expname=coarse datamodule=faceverse agent_tot=1 agent_id=0
```
Multiple instances of the script can also be runned in parallel by specifying `--agent_tot` and `--agent_id`.

3. Stage2: train the normal and texture networks.
```
python train.py expname=fine datamodule=faceverse +experiments=fine starting_path='./outputs/coarse/checkpoints/last.ckpt'
```
This step takes around 12 hours on 4 NVIDIA 3090 GPUs.

## Inference
### Sampling
### Face reenactment
### Raw scan fitting

## Acknowledgements
This project is built upon [gdna](https://github.com/xuchen-ethz/gdna). Some code snippets are also borrowed from [DECA](https://github.com/yfeng95/DECA) and [IMavatar](https://github.com/zhengyuf/IMavatar). Thanks for these great projects. We thank all the authors for their great work and repos.

## Citation
If you find our code or paper useful, please cite as:
```
@inproceedings{wu2023ganhead,
  title={GANHead: Towards Generative Animatable Neural Head Avatars},
  author={Wu, Sijing and Yan, Yichao and Li, Yunhao and Cheng, Yuhao and Zhu, Wenhan and Gao, Ke and Li, Xiaobo and Zhai, Guangtao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={437--447},
  year={2023}
}
