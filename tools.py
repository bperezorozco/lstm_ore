import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

DEBUG = True
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
    y = np.argmax( model.predict( np.expand_dims( frames[0], axis=2 ) ), axis=2 )

    if DEBUG:
        print frames.shape
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
    if DEBUG:   
       print synth.shape
    
    for i in range( timesteps, n_gen-1, hop ):
        pred = model.predict( synth[:, start_index:end_index] )
        synth[ :, end_index:end_index+hop ] = pred[ :, -hop: ] #take must be negative
        start_index = start_index + hop
        end_index = end_index + hop
    
    return synth

def get_full_prediction( model, weights_file, seed, n_pred=300, feed_uncertainty=False ):
    if weights_file != '':
        model.load_weights( weights_file )
        
    preds = []
    seed_binned = [ x.argmax() for x in seed[0] ]
    n_bins = seed.shape[-1]

    for i in range(n_pred):
        new_pred = model.predict( seed )
        preds.append( new_pred )
        
        new_pred_idx = new_pred.argmax()
        onehot_newpred = np.zeros( (1, 1, n_bins) )
        
        if feed_uncertainty:
            onehot_newpred[0, 0, :] = new_pred
        else:
            onehot_newpred[0, 0, new_pred_idx] = 1
            
        seed = np.concatenate( ( seed[:, 1:, :],  onehot_newpred), axis=1 )
    
    return preds

def get_full_prediction_aux( model, weights_file, seed, seed_aux, bins, n_pred=300, feed_uncertainty=False ):
    if weights_file != '':
        model.load_weights( weights_file )
        
    preds = []
    seed_binned = [ x.argmax() for x in seed[0] ]
    n_bins = seed.shape[-1]

    for i in range(n_pred):
        new_pred = model.predict( [seed, seed_aux] )
        preds.append( new_pred )
        new_pred_idx = new_pred.argmax()

        onehot_newpred = np.zeros( (1, 1, n_bins) )
        if feed_uncertainty:
            onehot_newpred[0, 0, :] = new_pred
        else:
            onehot_newpred[0, 0, new_pred_idx] = 1
        seed = np.concatenate( ( seed[:, 1:, :],  onehot_newpred), axis=1 )
        seed_aux = np.concatenate( (seed_aux[:, 1:], bins[new_pred_idx, None, None]), axis=1 )
    
    return preds

def plot_attractor( x, ax, lag, col='b', lbl='no_label' ):
    if x.ndim != 1:
        print 'Input must have a single dimension'
    if lag < 1:
        print 'Lag must be positive'
    
    ax.plot( x[:-2*lag], x[lag:-lag], x[2*lag:], col + '-', label=lbl )
    ax.legend()
     
def plot_attractors( x_list, lag=10, lbls=['data', 'pred', 'random', 'ones', 'zeros', 'gp'], seed_length=100, with_seed=False, title='Attractor' ):
    cols = ['or', 'ob', 'om', 'og', 'oc']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, x in enumerate(x_list):
        if with_seed:
            plot_attractor( x, ax, lag, col=cols[i], lbl=lbls[i] )
        else:
            plot_attractor( x[seed_length:], ax, lag, col=cols[i], lbl=lbls[i] )

    plt.title( title )
    fig.savefig( './test_figs/attractor_{}.pdf'.format( title ) )
    plt.close()


def build_attractor( x, lag=10 ):
    x_att = np.vstack( (x[:-2*lag], x[lag:-lag], x[2*lag:]) )
    return x_att

#assumes vectors don't have the seed!!!
def attractor_distance( x, y ):
    x_att = build_attractor( x ) 
    y_att = build_attractor( y ) 
    timesteps = x.shape[0]
    
    d =  cdist( x_att.T, y_att.T )
    return d.min( axis=0 ).sum()
