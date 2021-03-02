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
    # Run the graph: feed in the numpy arrays for x, y, w1, and w2; get numpy arrays for loss gras_w1, and grad_w2
    out = sess.run([loss, grad_w1, grad_w2], 
                    feed_dict=values)
    loss_val, grad_w1_val, grad_w2_val = out
```

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
    learning_rate = 1e-5
    for t in range(50):
        out = sess.run([loss, grad_w1, grad_w2], 
                    feed_dict=values)
        loss_val, grad_w1_val, grad_w2_val = out
        values[w1] -= learning_rate * grad_w1_val
        values[w2] -= learning_rate * grad_w2_val

```

Problem 1: copying weights between CPU/GPU each step 
Solution: Change w1 and w2 from **placeholder** (fed on each call) to **Variable** (persists in the graph between calls) 

```
N, D, H = 64, 1000, 100

x = tf.placeholder(tf.float32, shape=(N, D))
y = tf.placeholder(tf.float32, shape=(N, D))
#w1 = tf.placeholder(tf.float32, shape=(D, H))
w1 = tf.Variable(tf.random_normal((D, H)))
#w2 = tf.placeholder(tf.float32, shape=(H, D))
w2 = tf.Variable(tf.random_normal((H, D)))
# This is not initializing them, but to tell tensorflow how 

h = tf.maximum(tf.matmul(x, w1), 0)
y_pred = tf.matmul(h, w2)
diff = y_pred - y
loss = tf.reduce_mean(tf.reduce_sum(diff ** 2, axis = 1))
grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])

learning_rate = 1e-5
new_w1 = w1.assign(w1 - learning_rate * grad_w1)
new_w2 = w2.assign(w2 - learning_rate * grad_w2)
# Add dummy graph node that depends on updates
updates = tf.group(new_w1, new_w2)

with tf.Session() as sess:
   sess.run(tf.global_variables_initializer())
   # Run graph once to initialize w1 and w2
   values = {x: np.random.randn(N, D), 
             y: np.random.randn(N, D), }
   
   # Run many times to train 
   for t in range(50):
        loss_val, = sess.run([loss, updates], 
                             feed_dict=values)

```
Problem 2: loss not going down! Assign calls not actually being executed. 

Tensorflow: Optimizer





















