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

