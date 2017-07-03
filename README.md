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