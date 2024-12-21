# UAFER
Official implementation and checkpoints for paper "CEPrompt: Cross-Modal Emotion-Aware Prompting for Facial Expression Recognition" (accepted to Neurocomputing 2024) [![paper](https://img.shields.io/badge/Paper-87CEEB)](https://doi.org/) <be> 

---
### Abstract <br>
Facial Expression Recognition (FER) remains a challenging task due to unconstrained conditions like variations in illumination, pose, and occlusion. Current FER approaches mainly focus on learning discriminative features through local attention and global perception of visual encoders, while neglecting the rich semantic information in the text modality. Additionally, these methods rely solely on the softmax-based activation layer for predictions, resulting in overconfident decision-making that hampers the effective handling of uncertain samples and relationships. Such insufficient representations and overconfident predictions degrade recognition performance, particularly in unconstrained scenarios. To tackle these issues, we propose an end-to-end FER framework called UA-FER, which integrates vision-language pre-training (VLP) models with evidential deep learning (EDL) theory to enhance recognition accuracy and robustness. Specifically, to identify multi-grained discriminative regions, we propose the Multi-granularity Feature Decoupling (MFD) module, which decouples global and local facial representations based on image-text affinity while distilling the universal knowledge from the pre-trained VLP models. Additionally, to mitigate misjudgments in uncertain visual-textual relationships, we introduce the Relation Uncertainty Calibration (RUC) module, which corrects these uncertainties using EDL theory. In this way, the model enhances its ability to capture emotion-related discriminative representations and tackle uncertain relationships, thereby improving overall recognition accuracy and robustness. Extensive experiments on in-the-wild and in-the-lab datasets demonstrate that our UA-FER outperforms the state-of-the-art models.

<p align="center">
<img src="https://github.com/HaoliangZhou/UAFER/blob/master/uafer.png" width=100% height=100% 
class="center">
</p>

## Installation
1. Installation the package requirements
```
pip install -r requirements.txt
```

2. Download pretrained VLP(ViT-B/16) model from [OpenAI CLIP](https://github.com/openai/CLIP).

---
## Data Preparation
Following [CEPrompt](https://github.com/HaoliangZhou/CEPrompt?tab=readme-ov-file#data-preparation), the data lists are reorganized as follow:
1. The downloaded [RAF-DB](http://www.whdeng.cn/RAF/model1.html) are reorganized as follow:
```
data/
├─ RAF-DB/
│  ├─ basic/
│  │  ├─ EmoLabel/
│  │  │  ├─ images.txt
│  │  │  ├─ image_class_labels.txt
│  │  │  ├─ train_test_split.txt
│  │  ├─ Image/
│  │  │  ├─ aligned/
│  │  │  ├─ aligned_224/  # reagliend by MTCNN
```
2. The downloaded [AffectNet](http://mohammadmahoor.com/affectnet/) are reorganized as follow:
```
data/
├─ AffectNet/
│  ├─ affectnet_info/
│  │  ├─ images.txt
│  │  ├─ image_class_labels.txt
│  │  ├─ train_test_split.txt
│  ├─ Manually_Annotated_Images/
│  │  ├─ 1/
│  │  │  ├─ images
│  │  │  ├─ ...
│  │  ├─ 2/
│  │  ├─ ./
```
3. The structure of three data-load and -split txt files are reorganized as follow:
```
% (1) images.txt:
idx | imagename
1 train_00001.jpg
2 train_00002.jpg
.
15339 test_3068.jpg

% (2) image_class_labels.txt:
idx | label
1 5
2 5
.
15339 7

% (3) train_test_split.txt:
idx | train(1) or test(0)
1 1
2 1
.
15339 0
```

---
## Training
### Train and Eval the UAFER
```
python3 train.py \  
--dataset ${DATASET} \ 
--data-path ${DATAPATH}
```
### You can also run the script
```
bash train.sh
```


---
## Cite Our Work
If you find our work helps, please cite our paper.
```
@ARTICLE{Zhou2024UAFER,
  author={Zhou, Haoliang and Huang, Shucheng and Xu, Yuqiao},
  journal={Neurocomputing}, 
  title={UA-FER: Uncertainty-aware Representation Learning for Facial Expression Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}
}

``` 

---
## Contact
For any questions, welcome to create an issue or email to <a href="mailto:haoliangzhou6@gmail.com">haoliangzhou6@gmail.com</a>.




