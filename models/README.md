# Example: Unsupervised ConvLSTM Model

## Moving MNIST
Different varities of this dataset exist; you may find one at http://www.cs.toronto.edu/~nitish/unsupervised_video/ or in the ConvLSTM implementation at http://home.cse.ust.hk/~xshiab/.

## Tasks
Here we show a Caffe + ConvLSTM implementation of a standalone Future Predictor, as in [1].

### Future Predictor
> File: encode-decode.prototxt

We implement the Prediction network from [1]. An encoding network reads in features x over T=10 timesteps. The resulting hidden state and cell state are copied to a decoding network, which generates T-1=9 outputs. Finally, the last output of encoding and all outputs of decoding are concatenated and passed through a traditional convolutional layer, before applying cross-entropy loss.

## Literature
> [1]: X. Shi, Zh. Chen, H. Wang, D. Yeung, W. Wong, W. Woo: "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting". NIPS 2015.