# ConvLSTM layer

Implementation based on Jeff Donahue's LSTM implementation for Caffe. 

## Installation
Requires a recent version of caffe (or alternatively, the "recurrent" branch of Jeff Donahue's caffe github repository). 
Clone with `git clone -b recurrent-layer <address>`

Then, simply copy the files in include/ and src/ to their corresponding directories.

### Patching the proto file
You need to merge the protobuffer defintion in patch.proto with src/caffe/proto/caffe.proto.
To make this job easier, I have written a small patcher in python, see patch_proto.py. 

* Note: **I do not take any responsibility for files broken by the patcher!** Merge the files manually!
* It does create a backup file!
* Note: The patcher is more of a quick hack. Applying a patch more than once will destroy caffe.proto


If you do not want to use the patcher, you will have to manually merge the two files: Extend the block "LayerParameter" accordingly, and add the other blocks to the end of the file.

### Makefile.config
We provide a working configuration file with this repository, see 
> Makefile.config

It was tested with g++5 and Cuda 7.5.

### Building
Once everything is prepared, run make clean && make to recompile caffe.

### Notes on Compiling

Requires C++11 to compile. Set CUSTOM_CXX := g++ -std=c++11 in Makefile.config.
We have observed some bugs when compiling with g++-5 (which is not technically supported with CUDA 7.5). 
To avoid these problems, add: -D__STRICT_ANSI__ -D_MWAITXINTRIN_H_INCLUDED to the compiler line. 

Furthermore, a bug seems to appear in crop_layer.cu when using C++11 and Cuda. You can find a simple fix in fixes/.

## Usage
Note that Jeff's implementation expects data of shape T x N x ..., where T are the number of timesteps and N the number of independent streams, e.g., videos. 

That means the data needs to be interleaved: `<video1_t1>, <video2_t1>, <video1_t2>, <video2_t2>,` etc..

### Specifying a ConvLSTM layer
Use "lstm_convolution_param" to specify the details of the convolutional layer inside a ConvLSTM layer. It can have the following parameters:
- "type": Whether this is a input-to-hidden or hidden-to-hidden operations, or both ("input", "hidden", "all").
- This means you can specify up to two lstm_convolution_param per layer (one "hidden" plus one "input", or simply one "all")!
- Default Conv.-Params, such as "kernel_size", "num_outputs", "pad", etc.
- Certain features may not be available

### Example
For an example, please refer to the models/ directory!

### Feedback
If you find any bugs or have other feedback, please let me know! Thanks :)
