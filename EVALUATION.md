# Pretrained models

To load pretrained model you need to define `model_name` (size), `dataset` it was trained on, `train_type` whether it is pretrain-only (`ptn`), large-margin finetuning (`ft_lm`), finetuning with cutmix+mixup augmentations (`ft_mix`). 

```python
import torch

model_name='M' # ~b3-b4 size
train_type='ft_mix'
dataset='vb2+vox2+cnc'

model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name=model_name, 
                       train_type=train_type, 
                       dataset=dataset)
```

All models configurations with corresponding metrics can be found in following table:

| Model name (size)   |  Train dataset  |    Train type   |   Vox1-O EER(%) |   Vox1-E EER(%) |   Vox1-H EER(%) |   SITW EER(%) |   VOICES EER(%) |   CN-Celeb EER(%) |
|:-------------|:-------------|:-------------|-------------:|-------------:|-------------:|-----------:|-------------:|---------------:|
| M            | vb2+vox2+cnc | ft_mix       |            0.835 |            0.745 |            1.284 |          1.203 |            2.703 |              7.474 |
| M            | vb2          | ptn          |            1.319 |            1.128 |            2.000 |          1.482 |            4.116 |              9.012 |
| S            | vb2+vox2+cnc | ft_mix       |            0.936 |            0.874 |            1.510 |          1.310 |            2.774 |              8.043 |
| S            | vb2          | ptn          |            1.542 |            1.408 |            2.505 |          1.781 |            3.987 |              9.592 |
| b0           | vox2         | ft_lm        | 1.16 | 1.25 | 2.20 |          - |            - |              - |
| b0           | vox2         | ptn          |            - |            - |            - |          - |            - |              - |
| b1           | vox2         | ft_lm        | 0.85 | 0.97 | 1.73 |          - |            - |              - |
| b1           | vox2         | ptn          |            - |            - |            - |          - |            - |              - |
| b2           | vox2         | ft_lm        | 0.57 | 0.76 | 1.32 |          - |            - |              - |
| b2           | vox2         | ptn          |            - |            - |            - |          - |            - |              - |
| b3           | vox2         | ft_lm        | 0.50 | 0.73 | 1.33 |          - |            - |              - |
| b3           | vox2         | ptn          |            - |            - |            - |          - |            - |              - |
| b4           | vox2         | ft_lm        | 0.51 | 0.68 | 1.26 |          - |            - |              - |
| b4           | vox2         | ptn          |            - |            - |            - |          - |            - |              - |
| b5           | vox2         | ft_lm        | 0.43 | 0.61 | 1.08 |          - |            - |              - |
| b5           | vox2         | ptn          |            - |            - |            - |          - |            - |              - |
| b6           | vox2         | ft_lm        | 0.40 | 0.55 | 1.05 |          - |            - |              - |
| b6           | vox2         | ptn          |            - |            - |            - |          - |            - |              - |

## Paper metrics

| Model | Params | GMACs | LM | AS-Norm | Vox1-O EER(%) | Vox1-E EER(%) | Vox1-H EER(%) |
|-------|--------|-------|----|---------|---------------|---------------|---------------|
| **⬦ ReDimNet-B0** | **1.0M** | **0.43** | ✓ | ✗ | 1.16 | 1.25 | 2.20 |
| **⬥ ReDimNet-B0** | | | ✓ | ✓ | **1.07** | **1.18** | **2.01** |
| NeXt-TDNN-l (C=128,B=3)| 1.6M | 0.29* | ✗ | ✓ | 1.10 | 1.24 | 2.12 |
| NeXt-TDNN (C=128,B=3)| 1.9M | 0.35* | ✗ | ✓ | 1.03 | 1.17 | 1.98 |
| **⬦ ReDimNet-B1** | **2.2M** | **0.54** | ✓ | ✗ | 0.85 | 0.97 | 1.73 |
| **⬥ ReDimNet-B1** | | | ✓ | ✓ | **0.73** | **0.89** | **1.57** |
| ECAPA (C=512) | 6.4M | 1.05 | ✗ | ✓ | 0.94 | 1.21 | 2.20 |
| NeXt-TDNN-l (C=256,B=3)| 6.0M | 1.13* | ✗ | ✓ | 0.81 | 1.04 | 1.86 |
| CAM++ | 7.2M | 1.15 | ✓ | ✗ | 0.71 | 0.85 | 1.66 |
| NeXt-TDNN (C=256,B=3)| 7.1M | 1.35* | ✗ | ✓ | 0.79 | 1.04 | 1.82 |
| **⬦ ReDimNet-B2** | **4.7M** | **0.90** | ✓ | ✗ | 0.57 | 0.76 | 1.32 |
| **⬥ ReDimNet-B2** | | | ✓ | ✓ | **0.52** | **0.74** | **1.27** |
| ECAPA (C=1024) | 14.9M | 2.67 | ✓ | ✗ | 0.98 | 1.13 | 2.09 |
| DF-ResNet56 | 4.5M | 2.66 | ✗ | ✓ | 0.96 | 1.09 | 1.99 |
| Gemini DF-ResNet60 | 4.1M | 2.50* | ✗ | ✓ | 0.94 | 1.05 | 1.80 |
| **⬦ ReDimNet-B3** | **3.0M** | **3.00** | ✓ | ✗ | 0.50 | 0.73 | 1.33 |
| **⬥ ReDimNet-B3** | | | ✓ | ✓ | **0.47** | **0.69** | **1.23** |
| ResNet34 | 6.6M | 4.55 | ✓ | ✗ | 0.82 | 0.93 | 1.68 |
| Gemini DF-ResNet114 | 6.5M | 5.00 | ✗ | ✓ | 0.69 | 0.86 | 1.49 |
| **⬦ ReDimNet-B4** | **6.3M** | **4.80** | ✓ | ✗ | 0.51 | 0.68 | 1.26 |
| **⬥ ReDimNet-B4** | | | ✓ | ✓ | **0.44** | **0.64** | **1.17** |
| Gemini DF-ResNet183 | 9.2M | 8.25 | ✗ | ✓ | 0.60 | 0.81 | 1.44 |
| DF-ResNet233 | 12.3M | 11.17 | ✗ | ✓ | 0.58 | 0.76 | 1.44 |
| **⬦ ReDimNet-B5** | **9.2M** | **9.87** | ✓ | ✗ | 0.43 | 0.61 | 1.08 |
| **⬥ ReDimNet-B5** | | | ✓ | ✓ | **0.39** | **0.59** | **1.05** |
| ResNet293 | 23.8M | 28.10 | ✓ | ✗ | 0.53 | 0.71 | 1.30 |
| ECAPA2 | 27.1M | 187.00* | ✓ | ✗ | 0.44 | 0.62 | 1.15 |
| **⬦ ReDimNet-B6** | **15.0M** | **20.27** | ✓ | ✗ | 0.40 | 0.55 | 1.05 |
| **⬥ ReDimNet-B6** | | | ✓ | ✓ | **0.37** | **0.53** | **1.00** |

\* - means values have been estimated.