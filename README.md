## Caffe to Keras converter

**Note:** This converter has been adapted from code in [Marc Bolaños fork of
Caffe](https://github.com/MarcBS/keras). See acks for code provenance.

This is intended to serve as a conversion module for Caffe models to Keras
models.

Please, be aware that this module is not regularly maintained. Thus, some layers
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
