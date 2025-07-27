# 🧠MIRepNet: A pipeline and foundation model for EEG-based motor imagery classification
We will release our code after accepted.

## Overview
We introduced RepMI, the first EEG foundation model specifically designed for MI paradigm. A high-quality EEG data pipeline was developed, featuring a neurophysiologically informed channel template that aligns heterogeneous EEG electrode layouts into a unified spatial framework. Furthermore, an efficient pretraining strategy combining self-supervised masked token reconstruction and supervised MI classification was proposed, enabling rapid adaptation to novel subjects and tasks via minimal downstream fine-tuning. Extensive evaluations on five downstream MI tasks encompassing 47 subjects demonstrated the efficacy and robustness of RepMI, consistently surpassing state-of-the-art specialized and generalized EEG models. Our results underscore the significant advantage and practical necessity of paradigm-specific EEG foundation modeling. 
## Contributions
- We introduce MIRepNet, the first paradigm-specific foundation model specifically tailored explicitly for MI tasks. By capturing MI-specific neurophysiological features, MIRepNet effectively learns generalizable representations for MI decoding.
- We propose a high-quality EEG preprocessing pipeline comprising subject screening, a unified channel-template-based spatial alignment, frequency filtering, temporal resampling, and distribution alignment. This approach addresses challenges arising from heterogeneous EEG headset configurations, ensuring data consistency across diverse datasets.
- We develop an efficient pretraining approach combining masked token reconstruction and supervised MI classification. This strategy enables the model to acquire robust, generalizable temporal-spatial EEG representations.
-  Extensive experiments on five public MI datasets including 47 downstream subjects demonstrate that RepMI achieves state-of-the-art decoding accuracy (see Fig.~\ref{fig:radar} (b)). Moreover, RepMI requires significantly fewer calibration trials (fewer than 30 trials per class) and rapidly converges in a few epochs, highlighting its practical utility and effectiveness.
## Architecture of RepMI
![High-quality_Data_Construction](https://github.com/staraink/RepMI/blob/main/High-quality_Data_Construction.jpg)

![RepMI](https://github.com/staraink/RepMI/blob/main/RepMI.jpg)

## Results

![results](https://github.com/staraink/RepMI/blob/main/results.jpg)

## Datasets Download

In this paper, seven datasets are used for pretraining, and five downstream datasets are employed to validate RepMI.

* [BNCI2014002](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_002.html#moabb.datasets.BNCI2014_002)
* [PhysionetMI](https://moabb.neurotechx.com/docs/generated/moabb.datasets.PhysionetMI.html#moabb.datasets.PhysionetMI)
* [Dreyer2023](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Dreyer2023.html#moabb.datasets.Dreyer2023)
* [Weibo2014](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Weibo2014.html#moabb.datasets.Weibo2014)
* [Zhou2016](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Zhou2016.html#moabb.datasets.Zhou2016)
* [Lee2019](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html#moabb.datasets.Lee2019_MI)
* [Cho2017](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html#moabb.datasets.Cho2017)
* [BNCI2014001](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html#moabb.datasets.BNCI2014_001)
* [BNCI2015001](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2015_001.html#moabb.datasets.BNCI2015_001)
* [BNCI2014004](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_004.html#moabb.datasets.BNCI2014_004)
* [AlexMI](https://moabb.neurotechx.com/docs/generated/moabb.datasets.AlexMI.html#moabb.datasets.AlexMI)
* [BNCI2014001-4](https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html#moabb.datasets.BNCI2014_001)

The introduction of these datasets are summarized as follows:

![Datasets](https://github.com/staraink/RepMI/blob/main/Datasets.jpg)

## Baselines
In this paper, I have implemented nine EEG specialist models and five generalized models ⬇

### (1) EEG specialist models
* [CSP-LDA](https://ieeexplore.ieee.org/abstract/document/4408441): Optimizing Spatial Filters for Robust EEG Single-trial Analysis (IEEE Signal Processing Magazine 2007)
* [ShallowConv](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730): Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization (HBM 2017)
* [DeepConv](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730): Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization (HBM 2017)
* [EEGNet]([http://proceedings.mlr.press/v70/long17a.html](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta?casa_token=gbHBznN-MjgAAAAA:umQc5RN4DQ_zFDAhU5yIF4lR3D1gs5ZCv0nbdtqnL-skW7K8EphRQLuRV-L-q2pFNMB3NnahCP8uXKPvwdXvPjFdcqGR)): EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–computer Interfaces (JNE 2018)
* [IFNet](https://ieeexplore.ieee.org/abstract/document/10070810): IFNet: An Interactive Frequency Convolutional Neural Network for Enhancing Motor Imagery Decoding From EEG (TNSRE 2023)
* [ADFCNN](https://ieeexplore.ieee.org/abstract/document/10356088): ADFCNN: Attention-Based Dual-Scale Fusion Convolutional Neural Network for Motor Imagery Brain–Computer Interface (TNSRE 2023)
* [Conformer](https://ieeexplore.ieee.org/abstract/document/9991178): EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization (TNSRE 2022)
* [FBCNet](https://arxiv.org/abs/2104.01233): FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface (Arxiv 2021)
* [EDPNet](https://scholar.google.cz/scholar?hl=zh-CN&as_sdt=0%2C5&q=EDPNet%3A+An+Efficient+Dual+Prototype+Network+for+Motor+Imagery+EEG+Decoding&btnG=): EDPNet: An Efficient Dual Prototype Network for Motor Imagery EEG Decoding (Arxiv 2024)

### (2) EEG generalized models
* [BIOT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html): BIOT: Biosignal Transformer for Cross-data Learning in the Wild (NIPS 2023)
* [BENDR](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full): BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data (Front.hum.neurosci 2021)
* [LaBraM](https://openreview.net/forum?id=QzTpTRVtrP): Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI (ICLR 2024)
* [CBraMod](https://openreview.net/forum?id=NPNUHgHF2w): CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding (ICLR 2025)
* [EEGPT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html): EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals (NIPS 2024)


## Contact
For any questions or collaborations, please feel free to reach out via liudingkun@hust.edu.cn / zhu_chen@hust.edu.cn or open an issue in this repository.
