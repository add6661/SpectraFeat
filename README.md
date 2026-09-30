# SpectraFeat
SpectraFeat: Spectral Filtering and Orientation-Gated Attention for Robust Local Feature Matching

# SpectraFeat implementation
Pytorch implementation of SpectraFeat


## Requirements

Please use Python 3.9 and Pytorch (>= 1.1.0). Other dependencies should be easily installed through pip or conda.

## Explanation

To train SpectraFeat as described in the paper, you will need MegaDepth dataset and COCO_20k subset of COCO2017 dataset. As mentioned in the paper *[XFeat: Accelerated Features for Lightweight Image Matching](https://arxiv.org/abs/2404.19174)*, you can obtain the full COCO2017 training data at https://cocodataset.org/.
However, we [make available](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) a subset of COCO for convenience. We simply selected a subset of 20k images according to image resolution. Please check COCO [terms of use](https://cocodataset.org/#termsofuse) before using the data.

To reproduce the training setup from the paper, please follow the steps:
1. Download [COCO_20k](https://drive.google.com/file/d/1ijYsPq7dtLQSl-oEsUOGH1fAy21YLc7H/view?usp=drive_link) containing a subset of COCO2017;
2. Download MegaDepth dataset. You can follow [LoFTR instructions](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md#download-datasets), we use the same standard as LoFTR. Then put the megadepth indices inside the MegaDepth root folder following the standard below:
```bash
{megadepth_root_path}/train_data/megadepth_indices #indices
{megadepth_root_path}/MegaDepth_v1 #images & depth maps & poses
```
3. Finally you can call training
```bash
python -m modules.training.train --training_type SpectraFeat_default  --megadepth_root_path <path_to>/MegaDepth --synthetic_root_path <path_to>/coco_20k  --ckpt_save_path /path/to/ckpts 
```




