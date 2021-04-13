Traceback (most recent call last):
  File "train_decoders.py", line 97, in <module>
    do_log = True if val == 'power_log' else False
  File "/home/xinjie/xinjie/HTNet/HTNet_generalized_decoding/run_nn_models.py", line 455, in run_nn_models
    accs_lst, last_epoch_tmp = cnn_model(X_train, Y_train,X_validate,
  File "/home/xinjie/xinjie/HTNet/HTNet_generalized_decoding/run_nn_models.py", line 45, in cnn_model
    model = htnet(nb_classes, Chans = X_train.shape[2], Samples = X_train.shape[-1], 
  File "/home/xinjie/xinjie/HTNet/HTNet_generalized_decoding/htnet_model.py", line 86, in htnet
    block1 = Lambda(lambda inputs: inputs[0]-inputs[1])([X1, X2])
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/keras/engine/base_layer.py", line 922, in __call__
    outputs = call_fn(cast_inputs, *args, **kwargs)
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/keras/layers/core.py", line 888, in call
    result = self.function(inputs, **kwargs)
  File "/home/xinjie/xinjie/HTNet/HTNet_generalized_decoding/htnet_model.py", line 86, in <lambda>
    block1 = Lambda(lambda inputs: inputs[0]-inputs[1])([X1, X2])
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/ops/math_ops.py", line 984, in binary_op_wrapper
    return func(x, y, name=name)
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/ops/gen_math_ops.py", line 10102, in sub
    _, _, _op, _outputs = _op_def_library._apply_op_helper(
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/framework/op_def_library.py", line 742, in _apply_op_helper
    op = g._create_op_internal(op_type_name, inputs, dtypes=None,
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py", line 593, in _create_op_internal
    return super(FuncGraph, self)._create_op_internal(  # pylint: disable=protected-access
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 3319, in _create_op_internal
    ret = Operation(
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 1816, in __init__
    self._c_op = _create_c_op(self._graph, node_def, inputs,
  File "/home/xinjie/anaconda3/envs/tf/lib/python3.8/site-packages/tensorflow/python/framework/ops.py", line 1657, in _create_c_op
    raise ValueError(str(e))
ValueError: Dimensions must be equal, but are 20 and 501 for '{{node lambda_2/sub}} = Sub[T=DT_FLOAT](lambda/Identity, lambda_1/Identity)' with input shapes: [?,1,94,20], [?,1,94,501].
