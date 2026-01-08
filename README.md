# Olfactory Stimulus Classification Using Correlation Coefficient Based EEG Redundant Channel Elimination

[![Paper](https://img.shields.io/badge/Paper-JICS_2025-blue.svg)](https://doi.org/10.7472/jksii.2025.26.6.51)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the paper:
**"Olfactory Stimulus Classification Using Correlation Coefficient Based EEG Redundant Channel Elimination"**, published in *Journal of Internet Computing and Services (JICS)*, 2025.

## Abstract
While multivariate feature extraction methods, such as the Wavelet-Spatial Domain Feature (WSDF), demonstrate high classification performance in olfactory electroencephalogram (EEG) analysis, their reliance on all available channels results in significant computational costs.

To address this, we propose a novel pipeline that integrates a **Pearson correlation-based channel selection method** as a preprocessing step to the WSDF algorithm.
* **Efficiency:** Reduces feature extraction time by up to **58%**.
* **Performance:** Maintains or slightly improves classification accuracy compared to using all channels.
* **Method:** Systematically identifies and eliminates redundant channels by analyzing inter-channel correlations.

## Dataset
We used a public olfactory EEG dataset. Please download the dataset from [https://ieee-dataport.org/documents/olfactory-eeg-datasets-eegdot-and-eegdoc] and structure it as follows
```bash
data/
├── Sub. 1_A_001.fdt
├── Sub. 1_A_001.set
└── ...
```

## Usage
Train the model and evaluate performance
```bash
python .\src\main.py
```

## Results
<img width="1236" height="297" alt="image" src="https://github.com/user-attachments/assets/915fe4d5-3de4-46a0-8f20-0e0cac20208f" />
<img width="1359" height="329" alt="image" src="https://github.com/user-attachments/assets/893bae15-0ebf-473a-be9e-d0a7590bb0fa" />
<img width="1395" height="452" alt="image" src="https://github.com/user-attachments/assets/86f62efc-247f-4b13-8436-bcd65a836492" />


## Requirements
This code was tested on with `Python 3.11`
```bash
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
mne
pywavelets
```

```bash
pip install -r requirements.txt
```

##  citation
If you find this code useful for your research, please cite our paper:

**[APA Style]**
Kim, H., Shin, W., & Nam, C. (2025). Olfactory Stimulus Classification Using Correlation Coefficient Based EEG Redundant Channel Elimination. Journal of Internet Computing and Services, 26(6), 51-62. DOI: 10.7472/jksii.2025.26.6.51.

**[IEEE Style]**
H. Kim, W. Shin, C. Nam, "Olfactory Stimulus Classification Using Correlation Coefficient Based EEG Redundant Channel Elimination," Journal of Internet Computing and Services, vol. 26, no. 6, pp. 51-62, 2025. DOI: 10.7472/jksii.2025.26.6.51.

**[ACM Style]**
Hyun-il Kim, Woong-sik Shin, and Choon-sung Nam. 2025. Olfactory Stimulus Classification Using Correlation Coefficient Based EEG Redundant Channel Elimination. Journal of Internet Computing and Services, 26, 6, (2025), 51-62. DOI: 10.7472/jksii.2025.26.6.51.
