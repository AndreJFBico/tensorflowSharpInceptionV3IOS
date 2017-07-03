# TensorflowSharp InceptionV3 experiment

Note requires the compiled C api tensorflow libraries in:
```
https://github.com/AndreJFBico/tensorflow/tree/r1.1
```
When adding the core and protobuf .a libraries to the project don't forget to change the libtensorflow-core.linkwith.cs 

This experiment runs on Debug mode in the simulator and writes the classification results to the console output. Not working on the device it gives a compiler error:
```
System.TypeInitializationException: The type initializer for 'TensorFlow.TFBuffer' threw an exception. ---> System.ExecutionEngineException: Attempting to JIT compile method '(wrapper native-to-managed) TensorFlow.TFBuffer:FreeBlock (intptr,intptr)' while running in aot-only mode.
```

# Notes

Xamarin iOS: Version: 10.10.0.36

Xcode version: 8.3.2

Mono: 5.0.1.1

Tensorflow: r1.1

Tensorflow Sharp: June 2017