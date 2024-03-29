
DeepConvNets Training:

- Trial-wise training 

For each trial, the trial signal is used as input and the corresponding trial label asa target to train the ConvNet. 

- Cropped training

The cropped training strategy uses crops, i.e., sliding input windows within the trial, which leads to many more training examples for the network than the trial-wise training strategy. 

Overall, this resulted in 625 crops and therefore 625 label predictions per trial. The mean of these 625 predictions is used as the final prediction for the trial during the test phase. --> Cropped training increases the training set size by a factor of 625. 

The cropped training method leads to a new hyperparameter: the number of crops that are processed at the same time. 

The larger the number of crops, the larger the speedup one can get. 

---



---

[可借鉴的训练过程](https://robintibor.github.io/braindecode/notebooks/Trialwise_Decoding.html)

---

loss function 大幅改进！
```
criterion = nn.CrossEntropyLoss()

deep4model.compile(loss=criterion, optimizer=optimizer, iterator_seed=1, )
```

---

绘制multiclass baseline 的条状图

[python 保存矢量图](https://blog.csdn.net/Poul_henry/article/details/88294297)

[绘制条形图](https://zhuanlan.zhihu.com/p/25128216)

---

[Read gdf](https://www.programmersought.com/article/31122630428/)

---

single_sbj_training 

learning rate = 0.01

weight_decay = 0.001

batch_size = 4

```
optimizer = AdamW(deep4model.parameters(), lr=1 * 0.01, weight_decay=0.001)

#deep4model.compile(loss=F.nll_loss, optimizer=optimizer, iterator_seed=1, )
deep4model.compile(loss=criterion, optimizer=optimizer, iterator_seed=1, )

deep4model.fit(train_data, train_label, epochs=100, batch_size=4, scheduler='cosine',
              validation_data=(val_data, val_label), remember_best_column='valid_loss', )
              # loss function? 

history8 = deep4model.epochs_df
print(history8)

```
output:

```
train_loss  valid_loss  train_misclass  valid_misclass   runtime
0     0.158286    1.630520        0.034146        0.500000  0.000000
1     0.282955    2.227727        0.107317        0.558824  0.700789
2     0.135450    2.089952        0.029268        0.514706  0.659522
3     0.167929    1.703305        0.048780        0.544118  0.666153
4     0.383309    2.449876        0.156098        0.632353  0.658703
5     0.145946    2.034413        0.053659        0.588235  0.658547
6     0.185438    2.484074        0.078049        0.529412  0.659072
7     0.082264    2.068612        0.024390        0.500000  0.658974
8     0.127770    2.505341        0.053659        0.500000  0.658822
9     0.089598    1.738926        0.034146        0.573529  0.667188
10    0.090617    2.174409        0.043902        0.500000  0.658924
11    0.041898    2.108466        0.009756        0.529412  0.658856
12    0.380520    3.317204        0.121951        0.529412  0.658538
13    0.101602    1.690527        0.048780        0.470588  0.659323
14    0.283455    1.880919        0.087805        0.500000  0.659170
15    0.200497    2.467774        0.073171        0.573529  0.675409
16    0.100796    1.987552        0.043902        0.514706  0.659257
17    0.087119    2.106706        0.029268        0.514706  0.658949
18    0.064717    2.207149        0.029268        0.529412  0.660533
19    0.047756    2.158039        0.009756        0.544118  0.657942
20    0.112315    2.461140        0.048780        0.544118  0.663732
21    0.012422    2.068894        0.000000        0.558824  0.664397
22    0.007268    2.245999        0.000000        0.558824  0.664045
23    0.023146    2.331896        0.004878        0.514706  0.658610
24    0.029908    1.987853        0.004878        0.500000  0.658893
25    0.016562    1.916895        0.000000        0.529412  0.659140
26    0.012147    1.942908        0.000000        0.514706  0.658729
27    0.022965    1.962652        0.009756        0.470588  0.659063
28    0.006852    1.782545        0.000000        0.544118  0.658696
29    0.055030    2.312063        0.019512        0.544118  0.659083
30    0.021451    1.850484        0.000000        0.514706  0.670523
31    0.292671    2.556932        0.121951        0.573529  0.669550
32    0.112969    1.930143        0.043902        0.426471  0.669354
33    0.093279    1.787467        0.029268        0.441176  0.668126
34    0.126454    2.435751        0.043902        0.500000  0.666178
35    0.116715    1.684855        0.053659        0.470588  0.667808
36    0.035295    2.138998        0.009756        0.500000  0.667725
37    0.015844    2.150216        0.000000        0.455882  0.666980
38    0.020877    2.797687        0.004878        0.529412  0.667346
39    0.010170    2.009787        0.000000        0.500000  0.667501
40    0.022160    2.035425        0.004878        0.441176  0.668107
41    0.012502    1.929241        0.000000        0.485294  0.670255
42    0.005019    1.896852        0.000000        0.500000  0.670152
43    0.010341    1.877459        0.000000        0.500000  0.670142
44    0.004078    2.002730        0.000000        0.500000  0.670272
45    0.004259    2.005555        0.000000        0.500000  0.667169
46    0.002579    1.710731        0.000000        0.470588  0.668494
47    0.003273    1.788825        0.000000        0.470588  0.674854
48    0.002558    1.773003        0.000000        0.514706  0.669987
49    0.002302    1.795718        0.000000        0.426471  0.669674
50    0.006932    1.738347        0.000000        0.411765  0.668988
51    0.003176    1.550689        0.000000        0.426471  0.680277
52    0.003894    1.649565        0.000000        0.470588  0.673846
53    0.002150    1.581338        0.000000        0.441176  0.670968
54    0.001593    1.538299        0.000000        0.455882  0.673005
55    0.002000    1.478102        0.000000        0.426471  0.674397
```

