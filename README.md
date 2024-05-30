# Parameter Efficient Point Cloud Prompt Tuning for Unified Point Cloud Understanding

<!-- 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/point-bert-pre-training-3d-point-cloud/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=point-bert-pre-training-3d-point-cloud)

Created by [Xumin Yu](https://yuxumin.github.io/)\*, [Lulu Tang](https://github.com/lulutang0608)\*, [Yongming Rao](https://raoyongming.github.io/)\*, [Tiejun Huang](http://www.ai.pku.edu.cn/info/1139/1243.htm), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)

[[arXiv]](https://arxiv.org/abs/2111.14819) [[Project Page]](https://point-bert.ivg-research.xyz/) [[Models]](#pretrained-models) -->

This repository contains PyTorch implementation for __Parameter Efficient Point Cloud Prompt Tuning for Unified Point Cloud Understanding__.

The prevailing paradigm in 3D vision involves fully fine-tuning all the backbone parameters of pre-trained models. However, this approach poses challenges due to the large number of parameters requiring tuning, resulting in unexpected storage demands. To address these issues and alleviate the computational cost and storage burden associated with full fine-tuning, we propose Point Cloud Prompt Tuning (PCPT) as an effective method for large Transformer models in point cloud processing. PCPT offers a powerful and efficient solution to mitigate the costs associated with full fine-tuning. Drawing inspiration from recent advancements in efficient tuning of large-scale language models and 2D vision models, PCPT leverages less than 0.05 % of trainable parameters, while keeping the pre-trained parameters of the Transformer backbone unchanged. To evaluate the effectiveness of PCPT, extensive experiments were conducted on four discriminative datasets (ModelNet40, few-shot ModelNet40, ScanObjectNN, ShapeNetPart) and four generation datasets (PCN, MVP, ShapeNet55, and ShapeNet34/Unseen21). The results demonstrate that the task-specific prompts utilized in PCPT enable the Transformer model to adapt effectively to the target domains, yielding results comparable to those obtained through other full fine-tuning methods. This highlights the versatility of PCPT across various domains and tasks.

![intro](fig/teaser.png)

![intro](fig/pipeline.png)

![intro](fig/pcpt.png)

<!-- ## 🔥News
- **2021-11-26** Code and Pretrained weight released. -->
<!-- 
## Pretrained Models

|model| dataset | config | url|
| :---: | :---: | :---: |  :---: | 
| dVAE |ShapeNet |  [config](cfgs/ShapeNet55_models/dvae.yaml)|  [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c76274f9afb34cdbb57e/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1tiO5nYmkQ8jesPNPVWrdYQ)(code:26d3) |
|Point-BERT| ShapeNet |  [config](cfgs/Mixup_models/Point-BERT.yaml)|  [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/202b29805eea45d7be92/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1O0_9MYVTGcPcbBccfXZuog)(code:jvtg) |

|model| dataset  | Acc. | Acc. (vote) | config | url|
| :---:| :---: | :---: |  :---: | :---: | :---: |
| Transformer| ModelNet | 92.67 | 93.24 |  [config](cfgs/ModelNet_models/PointTransformer.yaml) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/9be5d9dcbaeb48adb360/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1Ykb9UwOeZRDXGwYZpO_AAg )(code:tqow) |
| Transformer| ModelNet | 92.91 | 93.48 |[config](cfgs/ModelNet_models/PointTransformer_4096point.yaml) | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/121b2651374e4ab1ade6/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1n2GIrOX93hpO5pgmV0Q1vw)(code:tcin) | 
| Transformer| ModelNet | 93.19 |93.76 | [config](cfgs/ModelNet_models/PointTransformer_8192point.yaml) |  [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/3ee8e437e07f4dc49738/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1o5DsEbPPA85dvuVuim8Y1Q)(code:k343) |
| Transformer| ScanObjectNN |88.12| -- |[config](cfgs/ScanObjectNN_models/PointTransformer_objectonly.yaml)| [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/60260a3cbd8940f5bf0d/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1ZH0mLlJKmtB22xUUALP-dQ)(code:f0km) | 
| Transformer| ScanObjectNN |87.43| -- | [config](cfgs/ScanObjectNN_models/PointTransformer_objectbg.yaml) |[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/c66c28c771e24cd588ad/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1nPiiDnV3qDmDqD17FW5rUg)(code:k3cb) | 
| Transformer| ScanObjectNN |83.07| -- | [config](cfgs/ScanObjectNN_models/PointTransformer_hardest.yaml) |[Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/2edb5b2810dc4bd9b796/?dl=1) / [BaiDuYun](https://pan.baidu.com/s/1ehxb9QPB2nkKYJixEMZeuw)(code:rxsw) |  -->


## Usage

### Requirements

- PyTorch >= 1.7.0
- python >= 3.7
- CUDA >= 9.0
- GCC >= 4.9 
- torchvision
- timm
- open3d
- tensorboardX

```
pip install -r requirements.txt
```

#### Building Pytorch Extensions for Chamfer Distance, PointNet++ and kNN

*NOTE:* PyTorch >= 1.7 and GCC >= 4.9 are required.

```
# Chamfer Distance
bash install.sh
# PointNet++
pip install "git+git://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Dataset

We use **ShapeNet** for the training of dVAE and the pre-training of models. And finetuning the models on **ModelNet**, **ScanObjectNN**, **ShapeNetPart**, **PCN**, **MVP**, **ShapeNet55/21**

The details of used datasets can be found in [DATASET.md](./DATASET.md).


### dVAE
To train a dVAE by yourself, simply run: 
```
bash scripts/train.sh <GPU_IDS>\
    --config cfgs/ShapeNet55_models/dvae.yaml \
    --exp_name <name>
```

Visualize the reconstruction results of a pre-trained dVAE, run: (default path: `./vis`)
```
bash ./scripts/test.sh <GPU_IDS> \
    --ckpts <path>\
    --config cfgs/ShapeNet55_models/dvae.yaml\
    --exp_name <name>
```

### Pre-training
To pre-train the models on ShapeNet, simply run:
(complete the `ckpt` in `cfgs/Mixup_models/PCPT-deep.yaml` first )
```
bash ./scripts/dist_train_BERT.sh <NUM_GPU> <port>\
    --config cfgs/Mixup_models/PCPT-deep.yaml \
    --exp_name pcpt_deep_pretrain 
    [--val_freq 10]
```
*val_freq* controls the frequence to evaluate the Transformer on ModelNet40 with LinearSVM.

### Fine-tuning on downstream tasks
We finetune our PCPT on 5 downstream tasks: Classfication on ModelNet40, Few-shot learning on ModelNet40, Transfer learning on ScanObjectNN, Part segmentation on ShapeNetPart, and Point Cloud Completion on PCN/MVP/ShapeNet55/ShapeNetUnseen21.

#### ModelNet40
To finetune a pre-trained PCPT model on ModelNet40, simply run:
```
# 1024 points
bash ./scripts/train_BERT.sh <GPU_IDS> \
    --config cfgs/ModelNet_models/PointTransformer.yaml\
    --finetune_model\
    --ckpts <path>\
    --exp_name <name>
# 4096 points
bash ./scripts/train_BERT.sh <GPU_IDS>\
    --config cfgs/ModelNet_models/PointTransformer_4096point.yaml\ 
    --finetune_model\ 
    --ckpts <path>\
    --exp_name <name>
# 8192 points
bash ./scripts/train_BERT.sh <GPU_IDS>\
    --config cfgs/ModelNet_models/PointTransformer_8192point.yaml\ 
    --finetune_model\ 
    --ckpts <path>\
    --exp_name <name>
```

To evaluate a model finetuned on ModelNet40, simply run:
```
bash ./scripts/test_BERT.sh <GPU_IDS>\
    --config cfgs/ModelNet_models/PointTransformer.yaml \
    --ckpts <path> \
    --exp_name <name>
```

#### Few-shot Learning on ModelNet40
We follow the few-shot setting in the previous work.

First, generate your own few-shot learning split or use the same split as us (see [DATASET.md](./DATASET.md)).
```
# generate few-shot learning split
cd datasets/
python generate_few_shot_data.py
# train and evaluate the PCPT
bash ./scripts/train_BERT.sh <GPU_IDS> \
    --config cfgs/Fewshot_models/PointTransformer.yaml \
    --finetune_model \
    --ckpts <path> \
    --exp_name <name> \
    --way <int> \
    --shot <int> \
    --fold <int>
```

#### ScanObjectNN
To finetune a pre-trained PCPT model on ScanObjectNN, simply run:
```
bash ./scripts/train_BERT.sh <GPU_IDS>  \
    --config cfgs/ScanObjectNN_models/PointTransformer_hardest.yaml \
    --finetune_model \
    --ckpts <path> \
    --exp_name <name>
```

To evaluate a model on ScanObjectNN, simply run:
```
bash ./scripts/test_BERT.sh <GPU_IDS>\
    --config cfgs/ScanObjectNN_models/PointTransformer_hardest.yaml \
    --ckpts <path> \
    --exp_name <name>
```

#### Part Segmentation

To finetune a pre-trained PCPT model on ShapeNetPart
```
cd segmentation
python train_partseg.py \
    --model PointTransformer \
    --gpu <GPU_IDS> \
    --pretrain_weight <path> \
    --log_dir <name> 
```
To evaluate a model on ShapeNetPart, simply run:
```
python test_partseg.py \
    --gpu <GPU_IDS> \
    --log_dir <name> 
```

#### Point Cloud Completion
```
bash scripts/run_OAE_pcn.sh <GPU_IDS> --config cfgs/ShapeNet55_models/completion.yaml --exp_name <name> 
```
```
bash scripts/test_pcn.sh <GPU_IDS> --config /data/feiben/Point_Cloud_Prompt_Tunning/cfgs/ShapeNet55_models/completion.yaml --ckpt <path> --mode easy/median/hard --exp_name <name> 
```

### Visualization
Part Segmentation Results on ShapeNetPart

![results](fig/seg_vis.png)

Point Cloud Completion Results on PCN, MVP, ShapeNet55 and ShapeNetUnseen21

![results](fig/PCN.png)

![results](fig/MVP.png)

![results](fig/55.png)

![results](fig/21.png)

## License
MIT License

<!-- 
## Citation
If you find our work useful in your research, please consider citing: 
```
@inproceedings{yu2021pointbert,
  title={Point-BERT: Pre-Training 3D Point Cloud Transformers with Masked Point Modeling},
  author={Yu, Xumin and Tang, Lulu and Rao, Yongming and Huang, Tiejun and Zhou, Jie and Lu, Jiwen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
# Point_Cloud_Prompt_Tunning
# Point_Cloud_Prompt_Tunning -->
