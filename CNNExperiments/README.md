# Experiment with CNN

Implement a CNN architecture with the following components: 
- Convolution.
- Activation Function (ReLU).
- Max Pooling.
- Fully Connected.

Then remove each and experiment with those architecture on MNIST dataset.

| Model	| Total time | Total Param | Loss | Accuracy | Precision | Recall | F1-Score |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |   
| `Conv+Pool+ReLU+FC` | 15s| 73.2 K| 1.5 | 0.962 | 0.963 | 0.962 | 0.96 |
| `Conv+Pool+FC` | 15s| 73.2 K | 1.54 | 0.95 | 0.95 | 0.95 | 0.95 |
| `Conv+ReLU+FC` | 45s| 13.5 M | 1.5 | 0.97 | 0.97 | 0.97 | 0.97 |
| `Conv+Pool+ReLU` | 15s| 54.7 K | 1.6 | 0.86 | 0.79 | 0.86 | 0.82 |