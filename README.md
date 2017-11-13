# Natural Language Classification via Neural Networks in Keras

Welcome! This repo contains the code to classify natural language sentences via Neural Networks, all coded in the Keras library.

The way the system is coded allows for the architecture to easily be changed in order to prototype and test new models. The current code is a CNN+GRU system.

This is the file structure:

- `train.py` should be used for training the system. It will save an `h5` file with the pretrained model once complete.
- `samplefunc.py` should be used for running inference against the system. It will use the saved `h5` file from the training script.
