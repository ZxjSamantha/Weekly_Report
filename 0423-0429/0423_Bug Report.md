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


line 79: if val == relative_power 时，do_log = True

In htnet()

```
if useHilbert:
    if compute_val == 'relative_power':
    	X1 = Lambda(apply_hilbert_tf, arguments = {'do_log':True, 'compute_val':'power'})(block1)
	X2 = AveragePooling2D((1, X1.shape[-1]//base_split), data_format = 'channels_last')(X1)
        X2 = Lambda(lambda x: tf.tile(x[...,:1],tf.constant([1,1,1,Samples], dtype=tf.int32)))(X2)
	block1 = Lambda(lambda inputs: inputs[0] - inputs[1])([X1, X2])
```



---

New bug:
```
tensorflow.python.framework.errors_impl.InvalidArgumentError:  Default AvgPoolingOp only supports NHWC on device type CPU
	 [[node model/average_pooling2d/AvgPool (defined at /home/xinjie/xinjie/HTNet/HTNet_generalized_decoding/run_nn_models.py:67) ]] [Op:__inference_train_function_1732]
```

nvidia failed

[Solution](https://qiita.com/ell/items/be3d3527b723f70f888d)

重启！
