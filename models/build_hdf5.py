import h5py
import numpy

FILENAME = 'seq.h5'

# Edit the shape here:
T = 19
N = 16

# My sequence indicators
indicators = numpy.ones( (T,N) )
for i in range(0,N):
  indicators[0][i] = 0

# Open an HDF5 file
h5file = h5py.File( FILENAME, 'w' )

# Set Sequence indicator
dataset = h5file.create_dataset( 'sequence', shape = indicators.shape, dtype = indicators.dtype )
dataset[:] = indicators

h5file.close()
