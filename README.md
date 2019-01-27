# I don't use Caffe or Keras any longer and am not planning to update this code. @ysh329 has compiled a [fantastic list of alternative software that you can use instead](https://github.com/ysh329/deep-learning-model-convertor). Happy converting! :smile:
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

## Caffe to Keras converter

**Note:** This converter has been adapted from code in [Marc Bolaños fork of
Caffe](https://github.com/MarcBS/keras). See acks for code provenance.

This is intended to serve as a conversion module for Caffe models to Keras
models. It only works
with [Ye Olde Caffe Classic™](https://github.com/BVLC/caffe) (which isn't really
a thing, but which probably should be a thing to prevent confusion with
the [Caffe 2](https://caffe2.ai/)).

Please be aware that this module is not regularly maintained. Thus, some layers
or parameter definitions introduced in newer versions of either Keras or Caffe
might not be compatible with the converter. Pull requests welcome!

### Conversion

In order to convert a model you just need the `.caffemodel` weights and the
`.prototxt` deploy file. You will need to include the input image dimensions as
a header to the `.prototxt` network structure, preferably as an `Input` layer:

```
layer {
  name: "image"
  type: "Input"
  top: "image"
  input_param {shape {dim: 1, dim: 3, dim: 128, dim: 128}}
}
```

Given the differences between Caffe and Keras when applying the max pooling
operation, in some occasions the max pooling layers must include a `pad: 1`
value even if they did not include them in their original `.prototxt`.

The module `caffe2keras` can be used as a command line interface for converting
any model the following way:

```
python -m caffe2keras models/train_val_for_keras.prototxt models/bvlc_googlenet.caffemodel keras-output-model.h5
```

To use the produced model from Keras, simply load the output file (i.e.
`keras-output-model.h5`) using `keras.models.load_model`.

### Acknowledgments

This code is yet another iteration of a tool which many people have contributed
to. Previous authors:

- Marc Bolaños ([email](mailto:marc.bolanos@ub.edu), [Github](https://github.com/MarcBS))
- Pranav Shyam ([Github](https://github.com/pranv))
- Antonella Cascitelli ([Github](https://github.com/lenlen))
