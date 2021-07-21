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
