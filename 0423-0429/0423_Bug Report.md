Bug in 0419 is about htnet_model.py, the Hilbert transformation

```
block1 = Lambda(lambda inputs: inputs[0] - inputs[1])(X1, X2)
```
It is called in run_nn_models.py:

model = htnet()

```
input1 = Input(shape = (1, Chans, Samples))
```

Training of the model is done in run_nn_models.py

Modification in run_nn_models.py:

line 44: `Samples = X_train.shape[-1]` -> `Samples = X_train.shape`
