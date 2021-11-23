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
| Method  | Backbone | Dataset | Bits | Dataset | Bits | Dataset | Bits | 
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| DHD  | AlexNet |  ImageNet | 16 / 32 / 64 | NUS-WIDE | 16 / 32 / 64 | MS COCO | 16 / 32 / 64 |
| DHD  | ResNet |  ImageNet | 16 / 32 / 64 | NUS-WIDE | 16 / 32 / 64 | MS COCO | 16 / 32 / 64 |
| DHD  | ViT |  ImageNet | 16 / 32 / 64 | NUS-WIDE | 16 / 32 / 64 | MS COCO | 16 / 32 / 64 |
| DHD  | DeiT |  ImageNet | 16 / 32 / 64 | NUS-WIDE | 16 / 32 / 64 | MS COCO | 16 / 32 / 64 |
| DHD  | SwinT |  ImageNet | 16 / 32 / 64 | NUS-WIDE | 16 / 32 / 64 | MS COCO | 16 / 32 / 64 |

## Retrieval Results
ImageNet
<p align="center"><img src="figures/Imagenet_results.png" width="900"></p>
NUS-WIDE
<p align="center"><img src="figures/Nuswide_results.png" width="900"></p>
MS COCO
<p align="center"><img src="figures/Mscoco_results.png" width="900"></p>


## Self-distilled hashing with other methods.
We will provide self-distilled hashing learning to improve previous deep hashing algorithms.


