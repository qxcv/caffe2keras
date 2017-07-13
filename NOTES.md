# Conversion notes

## Keras 2.0.6 weight shapes

I suspect the shape of weights can change from release to release, since it
broke on the upgrade from 1.2.X to 2.0.Y. The following shapes are known to be
valid for 2.0.6. Bias shape is only mentioned when not obvious.

| Layer          | How shape is calculated |
|----------------|-------------------------|
| `Conv{1,2,3}D` | [`kernel_size + (input_dim, filters)`](https://github.com/fchollet/keras/blob/2.0.6/keras/layers/convolutional.py#L128). That means `height*width*in*out` for `Conv2D`. |
| Dense | [`(input_dim, units)`](https://github.com/fchollet/keras/blob/2.0.6/keras/layers/core.py#L821) |

## Convolution layer output sizes: Caffe vs. Keras

Here is [Caffe's code for computing a convolution layer's output
shape](https://github.com/BVLC/caffe/blob/93bfcb53120416255d6d7261b638f0b38ff9e9bf/src/caffe/layers/conv_layer.cpp#L7-L22).
It does something like this for each output dimension (independently):

```python
dilated_extent = dilation * (kernel_shape - 1) + 1
# Original was C++; just using // to show that it rounds down
out_shape = (input_shape + 2 * pad - dilated_extent) // stride + 1
```

In contrast,
[this is what Keras does](https://github.com/fchollet/keras/blob/2.0.6/keras/utils/conv_utils.py#L90-L116)
to each axis:

```python
# real dilated_extent is kenel_shape + (kernel_shape - 1) * (dilation - 1),
# this is equivalent
dilated_extent = dilation * (kernel_shape - 1) + 1
if padding_mode in {'same', 'causal'}:
    padded_shape = input_shape
elif padding_mode == 'valid':
    padded_shape = input_shape - dilated_extent + 1
elif padding_mode == 'full':
    padded_shape = input_shape + dilated_extent - 1
out_shape = (padded_shape + stride - 1) // stride
```

Two important differences in the two implementations: firstly, padding is
handled by specification of a padding "mode" instead of a number of pixels to
pad by. Secondly, the default mode is 'valid', in which case the code above can
be simplified to:

```python
dilated_extent = dilation * (kernel_shape - 1) + 1
out_shape = (input_shape - dilated_extent + stride) // stride
```

As far as I can tell, the Keras' code for `'valid'`` mode is equivalent to
Caffe's code when `pad = 0`.`
