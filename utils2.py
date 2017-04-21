from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, TimeDistributedDense, Input, merge, Flatten
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import WeightRegularizer, l2
from keras.callbacks import EarlyStopping
from keras import backend as K
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import scipy.io 
import matplotlib.pyplot as plt
import matplotlib

import theano
import theano.tensor as T
from theano import shared
from theano import function


#frames the signal @x into frames of size @frame_size separated by @hop samples
def get_frames( x, frame_size, hop ):
    start_idx = 0
    end_idx = frame_size
    frames = []
    limit = x.shape[1]
    
    while end_idx <= limit:
        frames.append( x[:, start_idx:end_idx] )
        start_idx = start_idx + hop
        end_idx = start_idx + frame_size
    
    return np.float_(frames)

#evaluates @model over @x musing frames of length @length
def long_prediction( model, x, length ):
    frames = get_frames( x, length, length )
    print frames.shape
    y = np.argmax( model.predict( np.expand_dims( frames[0], axis=2 ) ), axis=2 )
    print y.shape
    
    for i in range( 1, frames.shape[0] ):
        new_y = np.argmax( model.predict( np.expand_dims( frames[i], axis=2 ) ), axis=2 )
        y = np.hstack( (y, new_y) )
    
    return y

#generates @n_seed samples from @model using opt@seed as the first len(seed) samples.
def get_samples( model, timesteps, sample_length, seed, hop, n_gen=2500  ):
    synth = np.zeros( (n_gen, sample_length) )
    start_index = 0
    end_index = timesteps
    
    if seed.shape[0] != timesteps:
        print 'ERROR: seed must be exactly ' + str(timesteps)+ 'samples long. Got: ' + str( seed.shape[0] )
        
    synth[0:end_index] = seed
    synth = np.expand_dims(synth, axis=0)
    print synth.shape
    
    for i in range( timesteps, n_gen-1, hop ):
        pred = model.predict( synth[:, start_index:end_index] )
        synth[ :, end_index:end_index+hop ] = pred[ :, -hop: ] #take must be negative
        start_index = start_index + hop
        end_index = end_index + hop
    
    return synth

def mg_gen( a, b, tau, n=11000, lag=85, skip=1000 ):
    N = n + lag + 1
    x = 0.1 * np.ones( N )
    
    for t in np.arange( tau, N - 1 ):
        x[t+1] = ( 1.0 - b ) * x[t] + a * ( x[t-tau] / ( 1.0 + np.power( x[t-tau], 10.0 ) ) )

    return x[skip:]
