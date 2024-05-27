# Deep Hash Distillation for Image Retrieval (ECCV2022)

Official Pytorch implementation of "Deep Hash Distillation for Image Retrieval" Accepted to ECCV2022 - <a href="https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740345.pdf">DHD</a> 

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

Datasets can be downloaded here:
<a href="https://drive.google.com/file/d/1TAjFKnOEse4xU_ScZOM8NgQLGexebmRn/view?usp=share_link">NUS-WIDE</a> / <a href="https://drive.google.com/file/d/1EsRZP3YsLbkbJ9rNXA4x5BFkHVFIGlQP/view?usp=share_link">MS COCO</a>

For ImageNet, please download through official website <a href="https://www.image-net.org/download.php">ImageNet</a> and follow our data configuration.

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


## Citation
```
@inproceedings{DHD,
  title={Deep Hash Distillation for Image Retrieval},
  author={Young Kyun Jang, Geonmo Gu, Byungsoo Ko, Isaac Kang, Nam Ik Cho},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

