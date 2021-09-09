# Cross-subject Movement Intention Decoding using Transfer Learning 

## Introduction

### Background of Brain-Computer Interface 

A brain-computer interface (BCI) can be defined as a system that translates the brain activity patterns of a user into messages or commands for an interactive application. For instance, a BCI can enable a user to move a cursor to the left or to the right of a computer screen by imagining left or right hand movements. EEG-based BCIs promise to revolutionize many applications areas: enable severely motor-impaired users to control assistive technologies, e.g. text input systems or wheelchairs, as rehabilitation devices for stroke patients, as new gaming input devices, or to design adaptive human-computer interfaces that can react to the user's mental states. 

(Invasive and Non-invasive)

EEG signals are highly user-specific and most current BCI systems are calibrated specifically for each user. 

---

### Background of Machine Learning Techniques

Most machine learning pipelines and BCIs not only use a classifier but also apply feature extraction/selection teniques to represent EEG signals in a compact and relevant manner. EEG signals are typically filtered both in the time domain (band-pass filter) and spatial domain (spatial filter) before features are extracted from the resulting signals. While there are many ways in which EEG signals can be represented, the two most common types of features used to represent EEG signals are **frequency band power features** and **time point features**.





---

### Methodology 

1. Notation Definition 

2. Network architecture 

3. Weight Decay 

L2正则化可能是最广泛使用的对参数化机器学习模型进行正则化的技术。

泛化性和灵活性之间的基本权衡被描述为bias-variance tradeoff。

---

### Background of Deep Learning on BCI 

---

### Background of Transfer Learning 

---

### Strategies of Transfer Learning 

1. 预训练网络直接用于新任务

2. 预训练+微调

3. 预训练网络作为新任务的特征提取器，参考DeCAF

4. 预训练提取特征加分类器构建

---

### Experiments & Results

---

### Discussion 
