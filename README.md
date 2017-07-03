# TensorflowSharp InceptionV3 experiment

Note requires the compiled C api tensorflow libraries in:
```
https://github.com/AndreJFBico/tensorflow/tree/r1.1
```
When adding the core and protobuf .a libraries to the project don't forget to change the libtensorflow-core.linkwith.cs 

This experiment runs on Debug mode in the simulator and writes the classification results to the console output. Not working on device for now.
