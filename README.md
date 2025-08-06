# Deep learning-based bathymetry retrieval without in-situ depths using remote sensing imagery and SfM-MVS DSMs with data gaps

This repository contains the code of the paper 
>Panagiotis Agrafiotis, Begüm Demir,
Deep learning-based bathymetry retrieval without in-situ depths using remote sensing imagery and SfM-MVS DSMs with data gaps,
ISPRS Journal of Photogrammetry and Remote Sensing,
Volume 225,
2025,
Pages 341-361,
ISSN 0924-2716,
https://doi.org/10.1016/j.isprsjprs.2025.04.020.
<br />


DOI of GitHub Repository [![DOI](https://zenodo.org/badge/880392000.svg)](https://doi.org/10.5281/zenodo.15630607)
<br />
## Abstract of the respective paper [![Elsevier Paper](https://img.shields.io/static/v1?label=Elsevier&message=Paper&color=FF6600)](https://doi.org/10.1016/j.isprsjprs.2025.04.020) [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2504.11416) [![MagicBathy](https://img.shields.io/badge/MagicBathy-Project-red.svg)](https://www.magicbathy.eu)
Accurate, detailed, and high-frequent bathymetry is crucial for shallow seabed areas facing intense climatological and anthropogenic pressures. Current methods utilizing airborne or satellite optical imagery to derive bathymetry primarily rely on either Structure-from-Motion and Multi-View Stereo (SfM-MVS) with refraction correction or Spectrally Derived Bathymetry (SDB). However, SDB methods often require extensive manual fieldwork or costly reference data, while SfM-MVS approaches face challenges even after refraction correction. These include depth data gaps and noise in environments with homogeneous visual textures, which hinder the creation of accurate and complete Digital Surface Models (DSMs) of the seabed. To address these challenges, this work introduces a methodology that combines the high-fidelity 3D reconstruction capabilities of the SfM-MVS methods with state-of-the-art refraction correction techniques, along with the spectral analysis capabilities of a new deep learning-based method for bathymetry prediction. This integration enables a synergistic approach where SfM-MVS derived DSMs with data gaps are used as training data to generate complete bathymetric maps. In this context, we propose Swin-BathyUNet that combines U-Net with Swin Transformer self-attention layers and a cross-attention mechanism, specifically tailored for SDB. Swin-BathyUNet is designed to improve bathymetric accuracy by capturing long-range spatial relationships and can also function as a standalone solution for standard SDB with various training depth data, independent of the SfM-MVS output. Experimental results in two completely different test sites in the Mediterranean and Baltic Seas demonstrate the effectiveness of the proposed approach through extensive experiments that demonstrate improvements in bathymetric accuracy, detail, coverage, and noise reduction in the predicted DSM.


## Citation

If you find this repository useful, please consider giving a star ⭐.
<br />

If you use the code in this repository please cite:

>Panagiotis Agrafiotis, Begüm Demir,
Deep learning-based bathymetry retrieval without in-situ depths using remote sensing imagery and SfM-MVS DSMs with data gaps,
ISPRS Journal of Photogrammetry and Remote Sensing,
Volume 225,
2025,
Pages 341-361,
ISSN 0924-2716,
https://doi.org/10.1016/j.isprsjprs.2025.04.020.
<br />

```
@article{AGRAFIOTIS2025341,
title = {Deep learning-based bathymetry retrieval without in-situ depths using remote sensing imagery and SfM-MVS DSMs with data gaps},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {225},
pages = {341-361},
year = {2025},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2025.04.020},
url = {https://www.sciencedirect.com/science/article/pii/S0924271625001522},
author = {Panagiotis Agrafiotis and Begüm Demir},
}
```

If you use the dataset please cite:

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

# Architecture Overview
Swin-BathyUNet combines U-Net with Swin Transformer self-attention layers and a cross-attention mechanism, tailored specifically for SDB.

![unetswin10](https://github.com/user-attachments/assets/abbba1fb-d56b-400c-9edb-e7a08535a6f0)



# Getting started

## Downloading the dataset

For downloading the dataset and a detailed explanation of it  (in case you don't have your own data or wish to compare), please visit the MagicBathy Project website at [https://www.magicbathy.eu/magicbathynet.html](https://www.magicbathy.eu/magicbathynet.html). When entering the Zenodo repository, download Version v2 of MagicBathyNet, which, in addition to v1, contains the SfM-MVS files.

## Clone the repo

`git clone https://github.com/pagraf/Swin-BathyUNet.git`

## Installation Guide
The requirements are easily installed via Anaconda (recommended):

`conda env create -f environment.yml`

After the installation is completed, activate the environment:

`conda activate magicbathynet`

## Train and Test the models
To train and test the model use **Swin-BathyUNet_MVS_SDB_Intergration_pub.ipynb**.
 
## Example testing results
Example aerial patch (GSD=0.25m) of the Agia Napa area (left), refraction corrected SfM-MVS priors used for training, and predicted bathymetry obtained by the Dual Attention Network (right). 

![img_410](https://github.com/user-attachments/assets/85e891e3-70d0-46f1-bdbc-23df9fc6128c)
![sfm_410](https://github.com/user-attachments/assets/3f35fa83-4ec4-4eea-b714-3d49f62de928)
![inference_410](https://github.com/user-attachments/assets/dfbf6552-6eb0-4ef7-b3c3-ccd41f439c6c)


Example Sentinel-2 patch (GSD=10m) of the Agia Napa area (left), refraction corrected SfM-MVS priors used for training, and predicted bathymetry obtained by the Dual Attention Network (right). 

![img_410_s2](https://github.com/user-attachments/assets/2f02c6d4-4079-4ef6-b0a6-6c0bf678191c)
![sfm_410](https://github.com/user-attachments/assets/3f35fa83-4ec4-4eea-b714-3d49f62de928)
![depth_410_s2](https://github.com/user-attachments/assets/0c090aab-62a5-42c1-80bb-4be95556ff73)

For more information on the results and accuracy achieved read our [paper](https://doi.org/10.1016/j.isprsjprs.2025.04.020). 

## Authors
Panagiotis Agrafiotis [https://www.user.tu-berlin.de/pagraf/](https://www.user.tu-berlin.de/pagraf/)

## Feedback
Feel free to give feedback, by sending an email to: agrafiotis@tu-berlin.de
<br />
<br />

# Funding
This work is part of **MagicBathy project funded by the European Union’s HORIZON Europe research and innovation programme under the Marie Skłodowska-Curie GA 101063294**. Work has been carried out at the [Remote Sensing Image Analysis group](https://rsim.berlin/). For more information about the project visit [https://www.magicbathy.eu/](https://www.magicbathy.eu/).
