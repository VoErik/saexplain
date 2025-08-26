# SAExplain

This repository contains the code for my Masters thesis.

## Components

tbc

## Quickstart

tbc

## Datasets

### Overview

This repository uses a mixture of publicly available dermatological datasets that encompass both clinical photographs
as well as dermoscopic images. An overview of the supported datasets can be found in the following table:

| Dataset                    | Description                                                                           | Modality    | Image Count | Link                                                                                                   | License |
|:---------------------------|:--------------------------------------------------------------------------------------|:------------|:------------|:-------------------------------------------------------------------------------------------------------|:--------|
| **DDI**                    | Curated to address racial bias; balanced across Fitzpatrick skin tones (I-VI).        | Clinical    | 656         | [Link](https://ddi-dataset.github.io)                                                                  | [License](https://ddi-dataset.github.io/) |
| **Fitzpatrick17k**         | Focuses on skin tone diversity by labeling clinical images with Fitzpatrick types.    | Clinical    | 16,577      | [Link](https://github.com/mattgroh/fitzpatrick17k)                                                     | CC BY-NC-SA 3.0 |
| **HAM10000**               | Benchmark for pigmented skin lesion classification into 7 categories.                 | Dermoscopic | 10,015      | [Link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)                | [License](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T&version=4.0&selectTab=termsTab) |
| **ISIC 2020**              | Large-scale challenge dataset for melanoma detection with patient-level data.         | Dermoscopic | 33,126      | [Link](https://challenge2020.isic-archive.com)                                                         | CC-BY-NC |
| **MRA-MIDAS**              | Prospectively-recruited with paired dermoscopic & clinical photos of the same lesion. | Multimodal  | ~3,800      | [Link](https://aimi.stanford.edu/datasets/mra-midas-Multimodal-Image-Dataset-for-AI-based-Skin-Cancer) | [License](https://stanfordaimi.azurewebsites.net/datasets/f4c2020f-801a-42dd-a477-a1a8357ef2a5#:~:text=Stanford%20University%20Dataset%20Research%20Use%20Agreement) | 
| **SCIN**                   | Crowdsourced "in-the-wild" images of common skin conditions from the public.          | Clinical    | ~10,000     | [Link](https://github.com/google-research-datasets/scin)                                               | [License](https://github.com/google-research-datasets/scin/blob/main/LICENSE) |
| **SKINCON-DDI**            | Concept-annotated variant of the DDI dataset.                                         | Clinical    | 656         | [Link](https://skincon-dataset.github.io)                                                              | [License](https://ddi-dataset.github.io/) |
| **SKINCON-Fitzpatrick17k** | Concept-annotated subset of Fitzpatrick17k .                                          | Clinical    | 3280        | [Link](https://skincon-dataset.github.io)                                                              | CC BY-NC-SA 3.0 |


#### 📥 Accessing the Datasets
This repository provides dataset classes and loaders for convenience. **However, it does not host or distribute the 
dataset images themselves.**

You are responsible for downloading each dataset directly from its official source using the links provided in the 
overview table.

Both the **DDI** and **MRA-MIDAS** datasets require you to access them via the Stanford AIMI Datasets Portal.

⚠️ **A Note on Licensing**

**Each dataset is governed by its own license. You must review and comply with the terms of use for each dataset before 
incorporating it into your work.**

---

### DDI

The DDI (Diverse Dermatological Images) dataset is a curated collection created specifically to address the lack of 
diversity in dermatology AI. It is the first publicly available, expertly-validated dataset that is intentionally 
balanced across the Fitzpatrick skin type scale. All cases are pathologically confirmed, making it an essential 
resource for auditing and mitigating racial bias in machine learning models.

This repository directly integrates the SKINCON variant of the DDI dataset into the loader.

### Fitzpatrick17k

The Fitzpatrick17k dataset contains 16,577 clinical images of various skin diseases, carefully annotated with 
Fitzpatrick skin types (I-VI). Its primary purpose is to facilitate the development and fair evaluation of 
dermatological AI models that perform equitably across a diverse spectrum of skin tones, addressing a critical gap in 
medical imaging datasets.

This repository includes loaders for both the basic Fitzpatrick17k dataset, as well as the more densely labeled 
SKINCON variant.



### HAM10000

The HAM10000 ("Human Against Machine with 10,000 training images") dataset is a collection of 10,015 multi-source 
dermoscopic images of common pigmented skin lesions. It covers seven important diagnostic categories, including 
melanoma and benign keratosis. A large portion of the cases are confirmed by histopathology, making it a standard 
benchmark for skin lesion classification tasks.



### SCIN
The SCIN dataset is a large, crowdsourced collection of over 10,000 "in-the-wild" images of common skin conditions, 
contributed by the general public. It includes self-reported demographic data and dermatologist-provided labels. 
This dataset is valuable for training models on a wider variety of image qualities, angles, and presentations than are 
typically found in controlled clinical settings.

### ISIC2020

The ISIC 2020 (International Skin Imaging Collaboration) dataset was released for a public challenge focused on 
melanoma detection. It is a large-scale collection containing over 33,000 high-quality dermoscopic images from more 
than 2,000 patients. Its inclusion of patient-level metadata makes it a premier resource for developing and testing 
algorithms for skin cancer diagnosis.



### MRA-MIDAS
The MRA-MIDAS dataset is a large, publicly available collection designed to mirror real-world clinical scenarios for 
skin cancer detection. It is unique as it was prospectively recruited and contains systematically paired dermoscopic 
and clinical photographs of the same skin lesion. The dataset includes extensive, patient-level clinical metadata and 
features a wide range of lesion diagnoses that are confirmed by histopathology. Its multimodal nature makes it a 
crucial resource for training and validating robust AI models that can generalize better than those trained on more 
curated, retrospective datasets.


## Models

### Feature Extractors

| Model               | Architecture | Pretraining Method             | Pretraining Dataset            | Source                                                      |
|:--------------------|:-------------|:-------------------------------|:-------------------------------|:------------------------------------------------------------|
| **CLIP**            | ViT          | Contrastive (Image-Text)       | DataComp-XL                    | [Link](https://github.com/mlfoundations/open_clip)          |
| **MAE (Timm)**      | ViT          | Self-Supervised (Masking)      | ImageNet                       | [Link](https://github.com/huggingface/pytorch-image-models) |
| **MAE**             | ViT          | Self-Supervised (Masking)      | Fitzpatrick17k, SCIN, Ham10000 | -                                                           |
| **DINOv2**          | ViT          | Self-Supervised (Distillation) | LVD-142M                       | [Link](https://github.com/facebookresearch/dinov2)          |
| **DINOv3**          | ViT          | Self-Supervised (Distillation) | LVD-1689M                      | [Link](https://github.com/facebookresearch/dinov3)          |

### Sparse Autoencoders

tbc

## Results

### 📊 Linear Probing the Feature Embeddings
**Comparison of mean accuracy and trainable parameter counts across different model architectures.** Models are evaluated on three classification schemes derived from the Fitzpatrick17k dataset: three-partition (*c=3*), nine-partition (*c=9*), and the original 114 disease labels (*c=114*). All accuracy values represent the average of five independent runs.

*Note: All ResNet models were pretrained on ImageNet-1k and subsequently fine-tuned on Fitzpatrick17k. The CLIP model is a ViT-L-14 pretrained on DataComp-1B. The Masked Autoencoder (MAE) models were pretrained on either a combination of dermatology datasets (Fitzpatrick17k, SCIN, and Ham10000; ⋆) or ImageNet-1k (†). For CLIP, DINO and MAE, performance was measured using linear probes trained for 50 epochs.*

| **Model**       | **Three Partition ↑** | **Nine Partition ↑** | **Disease ↑**   | ***Trainable Param.*** |
|:----------------|:----------------------|:---------------------|:----------------|:-----------------------|
| ResNet18        | 0.85 ± 0.00           | 0.78 ± 0.02          | 0.54 ± 0.00     | 1.169x10<sup>7</sup>   |
| ResNet50        | *0.86 ± 0.00*         | 0.81 ± 0.01          | 0.56 ± 0.01     | 2.556x10<sup>7</sup>   |
| CLIP-L (0-Shot) | 0.34 ± 0.00           | 0.06 ± 0.00          | 0.01 ± 0.00     | 0                      |
| CLIP-L (LP)     | 0.83 ± 0.00           | 0.77 ± 0.00          | 0.53 ± 0.00     | 3.230x10<sup>4</sup>   |
| MAE-S† (LP)     | 0.79 ± 0.00           | 0.73 ± 0.00          | 0.34 ± 0.00     | 3.230x10<sup>4</sup>   |
| MAE-S† (FT)     | **0.87 ± 0.00**       | **0.82 ± 0.00**      | 0.57 ± 0.00     | 8.640x10<sup>7</sup>   |
| MAE-S⋆ (LP)     | 0.76 ± 0.00           | 0.68 ± 0.00          | 0.22 ± 0.00     | 3.230x10<sup>4</sup>   |
| MAE-S⋆ (FT)     | 0.81 ± 0.00           | 0.74 ± 0.00          | 0.38 ± 0.00     | 8.640x10<sup>7</sup>   |
| DINOv2-L (LP)   | 0.82 ± 0.00           | 0.76 ± 0.00          | 0.51 ± 0.00     | 3.230x10<sup>4</sup>   |
| DINOv3-L (LP)   | 0.83 ± 0.00           | 0.79 ± 0.00          | *0.59 ± 0.00*   | 3.230x10<sup>4</sup>   |
| DINOv3-H (LP)   | 0.84 ± 0.00           | *0.81 ± 0.00*        | **0.60 ± 0.00** | 3.230x10<sup>4</sup>   |

## Dependencies

tbc

