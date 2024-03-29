## How to transfer in deep learning models?

把学习到的特征分为general features（一般来说，网络的前几层）和specific features（网络的后几层）.对于一个神经网络来说：

1. 哪几层学到的是general features? 哪几层学到的是specific features？

2. (previous works) 如何决定迁移哪些层? 如何固定哪些层？

3. 如何fine-tune?

如何网络的浅层学到的特征是通用的，在一个网络中，task-specific 的层如何适配?

MMD: 计算源域和目标域的距离，把该项加入网络的损失进行训练。

DDC： 分类层前一层加一层适配层，adaption layer. 

DaNN: 多适配了几层特征，多核MMD替换单核MD。 

---

## Transfer Learning for Motor Imagery Based Brain-Computer Interfaces: A Complete Pipeline 

Insights:

1. Data alignment component before spatial filtering is very important to the TL performance, for both (ML and DL, offline and online, cross-subject and cross-session). 

2. **Binary classification** 

---

## Make the plans in the timeline. (What would you do in the comming 5 months, what should you achieve in the next and next stage? ) 

---

Introduction: 

**Problems**

An MI-based BCI usually needs a long calibration session for a new subject. 

Transfer learning has been widely used in motor imagery (MI) based brain-computer interfaces to reduce the calibration effort for a new subject. 

**Insights**: It is very important to specifically add a data alignment component before spatial filtering to make the data from different subjects more consistent, and hence to facilitate subsequential TL.



Top-left of motor cortex for right-hand MI, 

Top-right of motor cortex for left-hand MI, 

Top-central of motor cortex for feet MI. 

---

Previous Work: 

Previous work in Transfer Learning: 

1. Dai et al. proposed transfer kernel CSP to integrate kernel CSP and transfer kernel learning for **EEG trial filtering**.

2. Chen et al., feature selection approaches, optimized both the class separability and the domain similarity. 

3. Band-pass filtering is performed on **both the source and target domain data**. 

4. Data alignment, which aligns EEG trials from the source domains and the target domain so that their distributions are more consistent. 

5. Spatial filtering, where TL can be used to design better spatial filters, especially when the amount of target domain labeled data is small. 

6. Feature engineering, where TL may be used to extract or select more informative features. 

7. Classification, where TL can be used to design better classifiers or regression models, especially when there are no or very few target domain labeled data. 

---

Proposed Methods and Its Originality 

---

Future Works: 

---

P1: Previous work : transfer learning, generalized decoder

P2: Why convolutional neural network? What is the originality? What is the best-performance baseline?

P3: 2-class datasets? 重点是验证算法而不是刷performance

P4: Transfer Learning framework!
