# Self-Distilled-Hashing for Deep Image Retrieval

Official Pytorch implementation of "Self-Distilled Hashing for Deep Image Retrieval"

*This repository is anonymized for double-blind review.*

## Overall training procedure of DHD

<p align="center"><img src="figures/framework.png" width="900"></p>


## Requirements

Prepare requirements by following command.
```
pip install -r requirements.txt
```

## Train DHD models
### Prepare datasets
We use public benchmark datasets: ImageNet, NUS-WIDE, MS COCO.  
Image file name and corresponding labels are provided in ```./data```.

Example
- Train DHD model with ImageNet, AlexNet backbone, 64-bit, temperature scaling with 0.2
- ```python main_DHD.py --dataset=imagenet --encoder=AlexNet --N_bits=64 --temp=0.2``` 

```python main_DHD.py --help``` will provide detailed explanation of each argument.

## Model ZOO
| Method  | Dataset | Backbone| Bits | mAP| Dataset | Backbone| Bits | mAP|
| ------------- | ------------- | ------------- | ------------- | ------------- |------------- | ------------- | ------------- | ------------- |
| DHD  | ImageNet | AlexNet  | 64 | 0.721 |ImageNet | AlexNet  | 64 | 0.721 |
| DHD  | ImageNet | ResNet  | 64 | 0.901 |
| DHD  | ImageNet | ViT  | 64 | 0.944 |
| DHD  | ImageNet | DeiT  | 64 | 0.948 |
| DHD  | ImageNet | SwinT  | 64 | 0.956 |

| Method  | Dataset | Backbone| Bits | mAP|
| ------------- | ------------- | ------------- | ------------- | ------------- |
| DHD  | NUS-WIDE | AlexNet  | 64 | 0.820 |
| DHD  | NUS-WIDE | ResNet  | 64 | 0.850 |
| DHD  | NUS-WIDE | ViT  | 64 | 0.870 |
| DHD  | NUS-WIDE | DeiT  | 64 | 0.867 |
| DHD  | NUS-WIDE | SwinT  | 64 | 0.875 |

| Method  | Dataset | Backbone| Bits | mAP|
| ------------- | ------------- | ------------- | ------------- | ------------- |
| DHD  | MS COCO | AlexNet  | 64 | 0.792 |
| DHD  | MS COCO | ResNet  | 64 | 0.889 |
| DHD  | MS COCO| ViT  | 64 | 0.939 |
| DHD  | MS COCO | DeiT  | 64 | 0.925 |
| DHD  | MS COCO | SwinT  | 64 | 0.945 |

## Retrieval Results
ImageNet
<p align="center"><img src="figures/Imagenet_results.png" width="900"></p>
NUS-WIDE
<p align="center"><img src="figures/Nuswide_results.png" width="900"></p>
MS COCO
<p align="center"><img src="figures/Mscoco_results.png" width="900"></p>


## Self-distilled hashing with other methods.
We will provide self-distilled hashing learning to improve previous deep hashing algorithms (TBD) .


