# GANHead: Towards Generative Animatable Neural Head Avatars
## [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_GANHead_Towards_Generative_Animatable_Neural_Head_Avatars_CVPR_2023_paper.pdf) | [Project Page](https://wsj-sjtu.github.io/GANHead/) | [Video](https://www.youtube.com/watch?v=Cg0ubzo7DXk)
Official implentation of CVPR 2023 paper [*GANHead: Towards Generative Animatable Neural Head Avatars*](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_GANHead_Towards_Generative_Animatable_Neural_Head_Avatars_CVPR_2023_paper.pdf).
### Code Comming Soon! 😉


## Training
### Prepare Data
First download [FaceVerse dataset](https://github.com/LizhenWangT/FaceVerse-Dataset) following their instructions.
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
