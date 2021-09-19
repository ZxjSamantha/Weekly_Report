DaNN -> DDC -> DAN 

---

### Chap 9 深度迁移学习的网络结构

最基础的结构的输入数据只包含一个来源，如果不添加新的约束，网络本身无法得知输入数据来自源域还是目标域

1. 单流结构

e.g., 知识蒸馏 knowledge distillation

2. 双流结构

对于前L层，可以选择共享部分层、微调部分层。**需要探索！！！**

瓶颈层


---

### Chap 8 如何让深度网络自己学会在何时、何处迁移

- 预训练模型可以获得大量任务的通用表现特征general features， 能否直接将预训练模型作为特征提取器，从新任务中提取特征， 从而进行后续的迁移学习。-> 特征嵌入 + 模型构建

- 预训练模型的应用方法：

  - 预训练网络直接应用于任务

  - 预训练+微调

  - 预训练网络作为新任务的特征提取器

  - 预训练提取特征加分类器构建

深度迁移学习的核心问题是研究深度网络的可迁移性，以及如何利用深度网络来完成迁移任务。

---

## Feature-based transfer & Model-based transfer