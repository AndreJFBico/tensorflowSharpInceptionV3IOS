# TensorflowSharp InceptionV3 experiment

Note requires the compiled C api tensorflow libraries in:
```
https://github.com/AndreJFBico/tensorflow/tree/r1.1
```
When adding the core and protobuf .a libraries to the project don't forget to change the libtensorflow-core.linkwith.cs 

This experiment runs writes the classification results to the console output. 

Tested with debug mode on the device and simulator.

# Notes

Xamarin iOS: Version: 10.10.0.36

Xcode version: 8.3.2

Mono: 5.0.1.1

Tensorflow: r1.1

Tensorflow Sharp: July 2017