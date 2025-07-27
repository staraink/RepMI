# Do BCIs Really Need One Generalized Model? RepMI: Toward Paradigm-Specific EEG Representations for Motor Imagery
We will release our code after accepted.

## Overview
We introduced RepMI, the first EEG foundation model specifically designed for MI paradigm. A high-quality EEG data pipeline was developed, featuring a neurophysiologically informed channel template that aligns heterogeneous EEG electrode layouts into a unified spatial framework. Furthermore, an efficient pretraining strategy combining self-supervised masked token reconstruction and supervised MI classification was proposed, enabling rapid adaptation to novel subjects and tasks via minimal downstream fine-tuning. Extensive evaluations on five downstream MI tasks encompassing 47 subjects demonstrated the efficacy and robustness of RepMI, consistently surpassing state-of-the-art specialized and generalized EEG models. Our results underscore the significant advantage and practical necessity of paradigm-specific EEG foundation modeling. 
## Contributions
- We introduce RepMI, the first paradigm-specific foundation model specifically tailored explicitly for MI tasks. By capturing MI-specific neurophysiological features, RepMI effectively learns generalizable representations for MI decoding.
- We propose a high-quality EEG preprocessing pipeline comprising subject screening, a unified channel-template-based spatial alignment, frequency filtering, temporal resampling, and distribution alignment. This approach addresses challenges arising from heterogeneous EEG headset configurations, ensuring data consistency across diverse datasets.
- We develop an efficient pretraining approach combining masked token reconstruction and supervised MI classification. This strategy enables the model to acquire robust, generalizable temporal-spatial EEG representations.
-  Extensive experiments on five public MI datasets including 47 downstream subjects demonstrate that RepMI achieves state-of-the-art decoding accuracy (see Fig.~\ref{fig:radar} (b)). Moreover, RepMI requires significantly fewer calibration trials (fewer than 30 trials per class) and rapidly converges in a few epochs, highlighting its practical utility and effectiveness.
## Architecture of RepMI

## Results

![results](https://github.com/staraink/RepMI/blob/main/results.jpg)

## Datasets Download

## Baselines

## Contact
For any questions or collaborations, please feel free to reach out via liudingkun@hust.edu.cn / zhu_chen@hust.edu.cn or open an issue in this repository.
