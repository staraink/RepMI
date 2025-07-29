# **🧠MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification**
![issues](https://img.shields.io/github/issues/staraink/MIRepNet)
![forks](https://img.shields.io/github/forks/staraink/MIRepNet?style=flat&color=orange)
![stars](https://img.shields.io/github/stars/staraink/MIRepNet?style=flat&color=red)
![license](https://img.shields.io/github/license/staraink/MIRepNet)

## :speech_balloon: Annoucement
- [2025.07.29] 🚩 **News**  The manuscript of MIRepNet can be found in [MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification](https://arxiv.org/abs/2507.20254).

- [2025.07.26] We propose **MIRepNet**, the first EEG foundation model tailored explicitly for motor imagery (MI), achieving **SOTA** performance across five public datasets and significantly outperforming existing specialized and generalized EEG models, even with fewer than 30 training trials per class.


## 📌 Abstract
Brain-computer interfaces (BCIs) enable direct communication between the brain and external devices. Recent EEG foundation models aim to learn generalized representations across diverse BCI paradigms. However, these approaches overlook fundamental paradigm-specific neurophysiological distinctions, limiting their generalization ability. Importantly, in practical BCI deployments, the specific paradigm such as motor imagery (MI) for stroke rehabilitation or assistive robotics, is generally determined prior to data acquisition. To address these issues, we propose MIRepNet, the first EEG foundation model explicitly tailored for the MI paradigm. MIRepNet comprises a high-quality EEG preprocessing pipeline incorporating a neurophysiologically-informed channel template, adaptable to EEG headsets with arbitrary electrode configurations. Furthermore, we introduce a hybrid pretraining strategy that combines self-supervised masked token reconstruction and supervised MI classification, facilitating rapid adaptation and accurate decoding on novel downstream MI tasks with fewer than 30 trials per class. Extensive evaluations across five public MI datasets demonstrate that MIRepNet consistently achieves state-of-the-art performance, significantly outperforming both specialized and generalized EEG models. We will release our code soon.

![RepMI](asset/RepMI.jpg)

## 🚀  Contributions
- 🧩 We introduce MIRepNet, the first paradigm-specific foundation model specifically tailored explicitly for MI tasks. By capturing MI-specific neurophysiological features, MIRepNet effectively learns generalizable representations for MI decoding.
- 🛠️ We propose a high-quality EEG preprocessing pipeline comprising subject screening, a unified channel-template-based spatial alignment, frequency filtering, temporal resampling, and distribution alignment. This approach addresses challenges arising from heterogeneous EEG headset configurations, ensuring data consistency across diverse datasets.
- 🎯 We develop an efficient pretraining approach combining masked token reconstruction and supervised MI classification. This strategy enables the model to acquire robust, generalizable temporal-spatial EEG representations.
- 📊 Extensive experiments on five public MI datasets including 47 downstream subjects demonstrate that RepMI achieves state-of-the-art decoding accuracy. Moreover, RepMI requires significantly fewer calibration trials (fewer than 30 trials per class) and rapidly converges in a few epochs, highlighting its practical utility and effectiveness.

## 💻 Deployment

### Environment Install
<details>
<summary>Install on Environment</summary> <br/> 
  
To configure the code environment（```python>=3.8,torch>=2.2.0```), use the following command:

```bash
git clone https://github.com/yourusername/MIRepNet.git
cd MIRepNet
conda create -n MIRepNet python>=3.8
conda activate MIRepNet
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

</details>

### Pretrained Model
MIRepNet should be placed into `./weight/MIRepNet.pth`.

### Finetune
To run the code, use the following command:
```bash
python finetune.py --dataset_name BNCI2014004 --model_name MIRepNet
```

Code for other datasets will be released.
## 📈 Results

![results](asset/Results.jpg)

## 📥 Datasets Download

In this paper, seven datasets are used for pretraining, and five downstream datasets are employed to validate RepMI. All datasets can be found in [MOABB](https://moabb.neurotechx.com/docs/dataset_summary.html#motor-imagery).

The introduction of these datasets is summarized as follows:

![Datasets](asset/Datasets.jpg)

## 🔍 Baselines
In this paper, I have implemented nine EEG specialist models and five generalized models ⬇

### 🧑‍🔬 EEG specialist models
* [CSP-LDA](https://ieeexplore.ieee.org/abstract/document/4408441): Optimizing Spatial Filters for Robust EEG Single-trial Analysis (IEEE Signal Processing Magazine 2007)
* [ShallowConv](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730): Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization (HBM 2017)
* [DeepConv](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730): Deep Learning with Convolutional Neural Networks for EEG Decoding and Visualization (HBM 2017)
* [EEGNet](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta?casa_token=gbHBznN-MjgAAAAA:umQc5RN4DQ_zFDAhU5yIF4lR3D1gs5ZCv0nbdtqnL-skW7K8EphRQLuRV-L-q2pFNMB3NnahCP8uXKPvwdXvPjFdcqGR): EEGNet: A Compact Convolutional Neural Network for EEG-based Brain–computer Interfaces (JNE 2018)
* [IFNet](https://ieeexplore.ieee.org/abstract/document/10070810): IFNet: An Interactive Frequency Convolutional Neural Network for Enhancing Motor Imagery Decoding From EEG (TNSRE 2023)
* [ADFCNN](https://ieeexplore.ieee.org/abstract/document/10356088): ADFCNN: Attention-Based Dual-Scale Fusion Convolutional Neural Network for Motor Imagery Brain–Computer Interface (TNSRE 2023)
* [Conformer](https://ieeexplore.ieee.org/abstract/document/9991178): EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization (TNSRE 2022)
* [FBCNet](https://arxiv.org/abs/2104.01233): FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface (Arxiv 2021)
* [EDPNet](https://scholar.google.cz/scholar?hl=zh-CN&as_sdt=0%2C5&q=EDPNet%3A+An+Efficient+Dual+Prototype+Network+for+Motor+Imagery+EEG+Decoding&btnG=): EDPNet: An Efficient Dual Prototype Network for Motor Imagery EEG Decoding (Arxiv 2024)

### 🌐 EEG generalized models
* [BIOT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html): BIOT: Biosignal Transformer for Cross-data Learning in the Wild (NIPS 2023)
* [BENDR](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.653659/full): BENDR: Using Transformers and a Contrastive Self-Supervised Learning Task to Learn From Massive Amounts of EEG Data (Front.hum.neurosci 2021)
* [LaBraM](https://openreview.net/forum?id=QzTpTRVtrP): Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI (ICLR 2024)
* [CBraMod](https://openreview.net/forum?id=NPNUHgHF2w): CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding (ICLR 2025)
* [EEGPT](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html): EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals (NIPS 2024)


## 📩 Contact
For any questions or collaborations, please feel free to reach out via `liudingkun@hust.edu.cn` / `zhu_chen@hust.edu.cn` or open an issue in this repository.

## Citation
If you find our repo or MIRepNet useful for your research, please cite us:
```
@misc{liu2025MIRepNet,
  title         = {MIRepNet: A Pipeline and Foundation Model for EEG-Based Motor Imagery Classification}, 
  author        = {Dingkun Liu and Zhu Chen and Jingwei Luo and Shijie Lian and Dongrui Wu},
  year          = {2025},
  eprint        = {2507.20254},
  archivePrefix = {arXiv},
}
```
