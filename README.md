# Swin-BathyUNet: A Swin-Transformer Enhanced U-Net for Shallow Water Bathymetry Using Remote Sensing Imagery and Multi-view Stereo-derived DSMs with Missing Data
![unetswin9](https://github.com/user-attachments/assets/e5198507-7491-4aff-9ac6-74764ecc5f5c)


This repository contains the code of the paper "P. Agrafiotis, Ł. Janowski, D. Skarlatos and B. Demir, "MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters," IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium, Athens, Greece, 2024, pp. 249-253, doi: 10.1109/IGARSS53475.2024.10641355."<br />

## Abstract of the respective [paper](https://arxiv.org/abs/2405.15477)
Accurate, detailed, and high-frequent bathymetry is crucial for shallow seabed areas facing intense climatological and anthropogenic pressures. Current methods utilizing optical remote sensing images to derive bathymetry primarily rely on either Structure-from-Motion and Multi-View Stereo (SfM-MVS) with refraction correction or Spectrally Derived Bathymetry (SDB). However, SDB methods often require extensive manual fieldwork or costly reference data while SfM-MVS approaches face challenges even after refraction correction. These include depth data gaps and noise in environments with homogeneous visual textures, which hinder the creation of accurate and complete Digital Surface Models (DSMs) of the seabed. To address these challenges, this work introduces a methodology that combines the high-fidelity 3D reconstruction capabilities of the SfM-MVS methods with state-of-the-art refraction correction techniques, along with the spectral analysis capabilities of a new deep learning-based method for bathymetry prediction. This integration enables a synergistic approach where SfM-MVS derived DSMs with data gaps, are used as training data to generate complete bathymetric maps. **In this context, we propose Swin-BathyUNet, a deep learning model that combines U-Net with Swin Transformer self-attention layers and a cross-attention mechanism, tailored specifically for SDB. Swin-BathyUNet is designed to improve bathymetric accuracy by capturing long-range spatial relationships and can also function as a standalone solution for standard bathymetric mapping with various training depth data, independent of SfM-MVS output**. Additionally, a boundary-sensitive weighted loss function is introduced to prioritize relevant depth labels. Experimental results at two test sites in the Mediterranean and Baltic Seas demonstrate the effectiveness of the proposed approach through extensive experiments showcasing improvements in bathymetric accuracy, detail, coverage, and noise reduction in the predicted DSM. The code will be made publicly available upon acceptance.


Download the paper from: [arXiv](https://arxiv.org/abs/2405.15477)

## Citation

If you find this repository useful, please consider giving a star ⭐.<br />
If you use the code in this repository or the dataset please cite:

>P. Agrafiotis, Ł. Janowski, D. Skarlatos and B. Demir, "MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters," IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium, Athens, Greece, 2024, pp. 249-253, doi: 10.1109/IGARSS53475.2024.10641355.
```
@INPROCEEDINGS{10641355,
  author={Agrafiotis, Panagiotis and Janowski, Łukasz and Skarlatos, Dimitrios and Demir, Begüm},
  booktitle={IGARSS 2024 - 2024 IEEE International Geoscience and Remote Sensing Symposium}, 
  title={MAGICBATHYNET: A Multimodal Remote Sensing Dataset for Bathymetry Prediction and Pixel-Based Classification in Shallow Waters}, 
  year={2024},
  volume={},
  number={},
  pages={249-253},
  doi={10.1109/IGARSS53475.2024.10641355}}
```
<br />

# Getting started

## Clone the repo

`git clone https://github.com/pagraf/MagicBathyNet.git`

## Installation Guide
The requirements are easily installed via Anaconda (recommended):

`conda env create -f environment.yml`

After the installation is completed, activate the environment:

`conda activate magicbathynet`

## Train and Test the models
To train and test the **bathymetry** models use **Dual_Attention-Based_Bathymetry.ipynb**.

## Pre-trained Deep Learning Models
We provide code and model weights for the following deep learning models that have been pre-trained on MagicBathyNet for pixel-based classification and bathymetry tasks:

### Learning-based Bathymetry
| Model Name | Modality | Area | Pre-Trained PyTorch Models                                                                                                                | 
| ----------- |----------| ---- |----------------------------------------------------------------------------------------------------------------------------------------------|
| Modified U-Net for bathymetry | Aerial | Agia Napa | [bathymetry_aerial_an.zip](https://drive.google.com/file/d/1-qUlQMHdZDZKkeQ4RLX54o4TK6juwOqD/view?usp=sharing) |
| Modified U-Net for bathymetry | Aerial | Puck Lagoon         | [bathymetry_aerial_pl.zip](https://drive.google.com/file/d/1SN8YH-WZIdR4e5Zl0uQK4OM62z_WNCks/view?usp=sharing)            |

To achieve the results presented in the paper, use the parameters and the specific train-evaluation splits provided in the dataset. Parameters can be found [here](https://drive.google.com/file/d/1gkIG99WFI6LNP7gsRvae9FZWU3blDPgv/view?usp=sharing) while train-evaluation splits are included in the dataset.

## Example testing results
Example aerial patch (GSD=0.25m) of the Agia Napa area (left), refraction corrected SfM-MVS priors used for training, and predicted bathymetry obtained by the Dual Attention Network (right). 

![img_410](https://github.com/user-attachments/assets/85e891e3-70d0-46f1-bdbc-23df9fc6128c)
![sfm_410](https://github.com/user-attachments/assets/3f35fa83-4ec4-4eea-b714-3d49f62de928)
![inference_410](https://github.com/user-attachments/assets/dfbf6552-6eb0-4ef7-b3c3-ccd41f439c6c)


Example Sentinel-2 patch (GSD=10m) of the Agia Napa area (left), refraction corrected SfM-MVS priors used for training, and predicted bathymetry obtained by the Dual Attention Network (right). 

![img_410_s2](https://github.com/user-attachments/assets/2f02c6d4-4079-4ef6-b0a6-6c0bf678191c)
![sfm_410](https://github.com/user-attachments/assets/3f35fa83-4ec4-4eea-b714-3d49f62de928)
![depth_410_s2](https://github.com/user-attachments/assets/0c090aab-62a5-42c1-80bb-4be95556ff73)

For more information on the results and accuracy achieved read our [paper](https://www.magicbathy.eu/). 

## Authors
Panagiotis Agrafiotis [https://www.user.tu-berlin.de/pagraf/](https://www.user.tu-berlin.de/pagraf/)

## Feedback
Feel free to give feedback, by sending an email to: agrafiotis@tu-berlin.de
<br />
<br />

# Funding
This work is part of **MagicBathy project funded by the European Union’s HORIZON Europe research and innovation programme under the Marie Skłodowska-Curie GA 101063294**. Work has been carried out at the [Remote Sensing Image Analysis group](https://rsim.berlin/). For more information about the project visit [https://www.magicbathy.eu/](https://www.magicbathy.eu/).
