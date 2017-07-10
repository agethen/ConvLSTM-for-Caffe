# Example: Unsupervised ConvLSTM Model

## Moving MNIST
Different varities of this dataset exist; you may find one at http://www.cs.toronto.edu/~nitish/unsupervised_video/ or in the ConvLSTM implementation at http://home.cse.ust.hk/~xshiab/.

## Tasks
Here we show a Caffe + ConvLSTM implementation of a standalone Future Predictor, as in [1].

### Future Predictor
> File: encode-decode.prototxt

We implement the 1-Layer Prediction network from [1]. An encoding network reads in features x over T=10 timesteps. The resulting hidden state and cell state are copied to a decoding network, which generates T-1=9 outputs. Finally, the last output of encoding and all outputs of decoding are concatenated and passed through a traditional convolutional layer, before applying cross-entropy loss.

The input data, provided by HDF5 files, is expected to be in range [0..1]. To inspect the result, extract the blob `out_sigm`, and multiply values by 255.

Note that the 64x64 input bitmaps are reshaped to 16x16x16 in the following manner (following the Theano ConvLSTM code available):

```
# Input shape = 10 x 16 x 64 x 64 (T x B x H x W)
def patchify( img, shape, size = 4 ):
  # shape = [10, 16, 64, 64]
  img   = numpy.reshape( shape[0], shape[1], shape[2]/size, size, shape[3]/size, size )
  img_T = numpy.transpose( img, [0,1,3,5,2,4] )
  return numpy.reshape( img_T, (shape[0], shape[1], size*size, shape[2]/size, shape[3]/size) )

# Input shape: 10 x 16 x 16 x 16 x 16 (T x B x C x H x W)
def reconstruct( img, shape, size = 4 ):
  # shape = [10, 16, 64, 64]
  img = numpy.reshape( img, (shape[0], orgshape[1], size, size, shape[2]/size, orgshape[3]/size) )
  img_T = numpy.transpose( img, (0,1,4,2,5,3) )
  return numpy.reshape( img_T, (shape[0], shape[1], shape[2], shape[3]) )
```


## Literature
> [1]: X. Shi, Zh. Chen, H. Wang, D. Yeung, W. Wong, W. Woo: "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting". NIPS 2015.