# TensorFlow Java Benchmarks

This package contains JMH benchmarks of several popular deep learning architectures using models from [TF-Hub](tfhub.dev).
The models need to be downloaded and uncompressed into the `models` folder.

- BERT, a transformer encoder which is quadratic in the number of tokens. Uses this [BERT model](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4).
- ELMO, a multi-layer LSTM. Uses this [ELMO model](https://tfhub.dev/google/elmo/3).
- ResNet50, a CNN. Uses this [ResNet50 model](https://tfhub.dev/tensorflow/resnet_50/classification/1).

The benchmarks are designed to be run on multiple platforms, and by supplying either the CPU or GPU jars to
determine where the computation is performed. They are used to measure the performance of TF-Java and monitor
for regressions or other performance issues.
