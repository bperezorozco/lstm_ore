import scipy.io
import numpy as np

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
    
    frames = np.float32(frames)
    
    return np.swapaxes( frames, 1, 2 )

def long_prediction( model, x, length, timesteps ):
    seed = x
    for i in range( length ):
        y = model.predict( np.expand_dims( seed, axis= 0 ) )
        x = np.vstack( (x, y[0, -1, 0]) )
        seed = x[-timesteps:]
    
    return x

def prepare_data( pdata, timesteps, sample_length, hop, return_full=False, mean_norm=True ):
    sample_length = 1
    hop = 1

    if mean_norm:
        data = ( pdata - pdata[0].mean() ) / pdata[0].std()

    X_train = get_frames( data[:, 0:-1], timesteps, hop )
    y_train = get_frames( data[:, 1:], timesteps, hop )

    if not return_full:
	y_train = y_train[:, -1, :]

    return ( X_train, y_train )

def prepare_attractor_data( data, timesteps, lags, hop, return_full=False, mean_norm=True ):
    sample_length = len(lags)
    
    if mean_norm:
        data = ( data - data[0].mean() ) / data[0].std()

    tmp = np.roll( data[:, lags[-1]:], lags[0] )
    for i in range( sample_length - 1 ):
        lag = np.roll( data, lags[i+1] )
        tmp = np.concatenate( (tmp, lag[:, lags[-1]:]), axis=0 )
    
    X_train = get_frames( tmp[:, 0:-1], timesteps, hop )
    y_train = get_frames( tmp[:, 1:], timesteps, hop )

    if not return_full:
	y_train = y_train[:, -1, :]

    return ( X_train, y_train )
