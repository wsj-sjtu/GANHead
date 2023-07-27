# GANHead: Towards Generative Animatable Neural Head Avatars
## [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_GANHead_Towards_Generative_Animatable_Neural_Head_Avatars_CVPR_2023_paper.pdf) | [Project Page](https://wsj-sjtu.github.io/GANHead/) | [Video](https://www.youtube.com/watch?v=Cg0ubzo7DXk)
Official implentation of CVPR 2023 paper [*GANHead: Towards Generative Animatable Neural Head Avatars*](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_GANHead_Towards_Generative_Animatable_Neural_Head_Avatars_CVPR_2023_paper.pdf).

## Installation
1. Clone the repository and set up a conda environment with all dependencies as follows:
```
git clone https://github.com/wsj-sjtu/GANHead.git
cd GANHead
conda env create -f env.yml
conda activate ganhead
python setup.py install
```

2. Download FLAME models

## Training
### Prepare Data
First download [FaceVerse dataset](https://github.com/LizhenWangT/FaceVerse-Dataset) following their instructions, and the fitted FLAME parameters form [here](https://drive.google.com/file/d/1W-r6H573sKW_euG1zjiEgqsSaO_aVvld/view?usp=drive_link). Organize all files into the following structure:
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
python preprocess.py --tot 1 --id 0 --scan_folder <folder_to_raw_scans> --flame_folder <folder_to_flame_parameters> --output_folder <>
```

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
