# Numpy, Tensorflow, PyTorch

---

```
import numpy as np
np.random.seed(0)

N, D = 3, 4

x = np.random.randn(N, D)
# assignment of x, y, z, a, b, c 

grad_c = 1.0
grad_b = grad_c * np.ones((N, D))

```

Tensorflow two-layer FC ReLU network:

```
N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
w1 = tf.placeholder(tf.float32, shape=(D, H))
w2 = tf.placeholder(tf.float32, shape=(H, D))

h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis = 1))

grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

with tf.Session() as sess:
    values = {x: np.random.randn(N, D), 
              w1: np.random.randn(D, H), 
              w2: np.random.randn(H, D), 
              y: np.random.randn(N, D), }
    out = sess.run([loss, grad_w1, grad_w2], 
                    feed_dict=values)
    loss_val, grad_w1_val, grad_w2_val = out
```

Problem: copying weights between CPU/GPU each step 
