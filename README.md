# Self-Distilled-Hashing for Deep Image Retrieval

Official Pytorch implementation of "Self-Distilled Hashing"

*This repository is anonymized for double-blind review.*

## Overall training procedure of DHD

<p align="center"><img src="Figure_framework.png" width="900"></p>


## Requirements

Prepare requirements by following command.
```
pip install -r requirements.txt
```

## Train DHD models
### Prepare datasets
We use public benchmark datasets: ImageNet, NUS-WIDE, MS COCO.  
Image file name and corresponding labels are provided in ```./data```.

DHD
- Train DHD model with ImageNet, AlexNet backbone, 64-bit, temperature scaling with 0.2
- ```python main_DHD.py --dataset=imagenet --encoder=AlexNet --N_bits=64 --temp=0.2``` 

```python main_DHD.py --help``` will provide detailed explanation of each argument.
