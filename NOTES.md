# Conversion notes

## Keras 2.0.6 weight shapes

I suspect the shape of weights can change from release to release, since it
broke on the upgrade from 1.2.X to 2.0.Y. The following shapes are known to be
valid for 2.0.6. Bias shape is only mentioned when not obvious.

| Layer          | How shape is calculated |
|----------------|-------------------------|
| `Conv{1,2,3}D` | [`kernel_size + (input_dim, filters)`](https://github.com/fchollet/keras/blob/2.0.6/keras/layers/convolutional.py#L128) |
| Dense | [`(input_dim, units)`](https://github.com/fchollet/keras/blob/2.0.6/keras/layers/core.py#L821) |
