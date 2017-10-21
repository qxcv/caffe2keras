#!/usr/bin/env python
"""Run this with ``python -m caffe2keras <args>``"""

import argparse

parser = argparse.ArgumentParser(
    description='Converts a Caffe model to a Keras model')
parser.add_argument(
    '--debug', action='store_true', default=False, help='enable debug mode')
parser.add_argument('prototxt', help='network definition path')
parser.add_argument('caffemodel', help='network weights path')
parser.add_argument('destination', help='path for output model')
parser.add_argument('--code-file', help='generate python code here')
parser.add_argument('--weights-file', help='put model weights here')


def main():
    args = parser.parse_args()

    # lazy import so that we parse args before initialising TF/Theano
    from caffe2keras import convert

    if args.code_file is not None:
        cgen = convert.CodeGenerator(args.code_file)
    else:
        cgen = convert.CodeGenerator(None)

    print("Converting model...")
    model = convert.caffe_to_keras(args.prototxt, args.caffemodel, cgen=cgen, debug=args.debug)
    print("Finished converting model")

    cgen.close()

    # Save converted model structure
    print("Storing model...")
    model.save(args.destination)
    print("Finished storing the converted model to " + args.destination)
    if args.weights_file:
        model.save_weights(args.weights_file)
        print("Saved weights to " + args.weights_file)
    if args.code_file:
        print("Saved code for model to " + args.code_file)


if __name__ == '__main__':
    main()
