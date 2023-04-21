# FSVOD

This work is from **Qi Fan**  (HKUST, fanqics@gmail.com) are open-sourced here. 

[FSVOD paper](https://arxiv.org/abs/2104.14805): few-shot video object detection with [FSVOD-500 dataset](https://drive.google.com/drive/folders/1DDQ81A8yVj7D8vLUS01657ATr2sK1zgC?usp=sharing) and [FSYTV-40 dataset](https://drive.google.com/drive/folders/1a1PpfAxeYL7AbxYViDDnx7ACFtRohVL5?usp=sharing). 



## Step 1: Installation
You only need to install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). We recommend the Pre-Built Detectron2 (Linux only) version with pytorch 1.7. I use the Pre-Built Detectron2 with CUDA 10.1 and pytorch 1.7 and you can run this code to install it.

```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.7/index.html
```

## Step 2: Training and Evaluation

Run `sh all.sh` in the root dir. (This script uses `4 GPUs`. You can change the GPU number. If you use 2 GPUs with unchanged batch size (8), please [halve the learning rate](https://github.com/fanq15/FewX/issues/6#issuecomment-674367388).)

```
cd FSVOD
sh all.sh
```


## Citing FewX
If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

  ```
  @inproceedings{fan2021fsvod,
    title={Few-Shot Video Object Detection},
    author={Fan, Qi and Tang, Chi-Keung and Tai, Yu-Wing},
    booktitle={ECCV},
    year={2022}
  }
  ```

## Special Thanks
[Detectron2](https://github.com/facebookresearch/detectron2), [AdelaiDet](https://github.com/aim-uofa/AdelaiDet), [centermask2](https://github.com/youngwanLEE/centermask2)
