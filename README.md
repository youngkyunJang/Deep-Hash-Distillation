# Deep Hash Distillation for Image Retrieval

Official Pytorch implementation of "Deep Hash Distillation for Image Retrieval"

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

## Retrieval Results with Different Backbone
S: Swin Transformer, R: ResNet, A: AlexNet

ImageNet
<p align="center"><img src="figures/Imagenet_results.png" width="900"></p>
NUS-WIDE
<p align="center"><img src="figures/Nuswide_results.png" width="900"></p>
MS COCO
<p align="center"><img src="figures/Mscoco_results.png" width="900"></p>


