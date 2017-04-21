from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, TimeDistributedDense, Input, merge, Flatten
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import WeightRegularizer, l2
from keras.callbacks import EarlyStopping
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
###################################TEST BIDIR MODEL FUNCTIONS###########################################
def create_bidir_model_ORE08( hidden_units, l, timesteps, n_bins, concat_input=False ):
    x = Input( shape=(timesteps, n_bins) )
    lstm1 = LSTM( input_shape=(timesteps,n_bins), 
                    output_dim=hidden_units, 
                    activation='tanh', 
                    inner_activation='sigmoid', 
                    return_sequences=False
                    )(x)

    lstm2 = LSTM( input_shape=(timesteps,n_bins), 
                    output_dim=hidden_units, 
                    activation='tanh', 
                    inner_activation='sigmoid', 
                    return_sequences=False,
                    go_backwards=True
                    )(x)

    flat_x = Flatten()(x)
    
    if concat_input:
        h = merge( [flat_x, lstm1, lstm2], mode='concat' )
    else:
        h = merge( [lstm1, lstm2], mode='concat' )
    
    h = Dropout(0.5)(h)
    predictions = Dense( n_bins, activation='softmax', W_regularizer=l2(l) )(h)
    
    
    model = Model( input=x, output=predictions )
    model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    
    #print "Model created successfully."
    #print model.summary()
    
    return model

###################################TEST BIDIR MODEL FUNCTIONS###########################################
def create_bidir_model( hidden_units, l, timesteps, n_bins, concat_input=False ):
    x = Input( shape=(timesteps, n_bins) )
    lstm1 = LSTM( input_shape=(timesteps,n_bins), 
                    output_dim=hidden_units, 
                    activation='tanh', 
                    inner_activation='sigmoid', 
                    return_sequences=False
                    )(x)

    lstm2 = LSTM( input_shape=(timesteps,n_bins), 
                    output_dim=hidden_units, 
                    activation='tanh', 
                    inner_activation='sigmoid', 
                    return_sequences=False,
                    go_backwards=True
                    )(x)

    aux_x = Input( shape=(timesteps,) )
    
    if concat_input:
        h = merge( [aux_x, lstm1, lstm2], mode='concat' )
    else:
        h = merge( [lstm1, lstm2], mode='concat' )
    
    h = Dropout(0.5)(h)
    predictions = Dense( n_bins, activation='softmax', W_regularizer=l2(l) )(h)
    
    
    model = Model( input=[x, aux_x], output=predictions )
    model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    
    #print "Model created successfully."
    #print model.summary()
    
    return model

###################################TEST ONE LAYER MODEL FUNCTIONS###########################################
def create_oneL_model( hidden_units, l, timesteps, n_bins ):
    model = Sequential()
    model.add( LSTM( input_shape=(timesteps,n_bins), 
                    output_dim=hidden_units, 
                    activation='tanh', 
                    inner_activation='sigmoid', 
                    return_sequences=False
                    ) )
    model.add( Dropout(0.5) )
    model.add( Dense(n_bins, activation='softmax', W_regularizer=l2(l)) )

    adam = Adam( lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-10 )

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["accuracy"])
    
    return model

def create_shallow_model( hidden_units, l, timesteps, n_bins ):
    model = Sequential()
    model.add(Flatten(input_shape=(timesteps,n_bins)))
    model.add( Dense(hidden_units, W_regularizer=l2(l)) )
    model.add( Dense(n_bins, W_regularizer=l2(l)) )
    model.compile( optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'] )
    
    #print "Shallow model created successfully."
    #print model.summary()
    
    return model


def test_model_aux( model, weights_file, seed, y_test, aux_x, bins, n_pred=300, debug=False ):
    if weights_file != '':
        model.load_weights( weights_file )
        
    preds = []
    seed_binned = [ x.argmax() for x in seed[0] ]
    n_bins = seed.shape[-1]

    for i in range(n_pred):
		new_pred = model.predict( [seed, aux_x] ).argmax()
		preds.append( new_pred )
		onehot_newpred = np.zeros( (1, 1, n_bins) )
		onehot_newpred[0, 0, new_pred] = 1
		seed = np.concatenate( ( seed[:, 1:, :],  onehot_newpred), axis=1 )
		aux_x = np.concatenate( (aux_x[:, 1:], bins[new_pred, None, None]), axis=1 )

    y_test_binned = [ y.argmax() for y in y_test ]

    if debug:
		plt.plot( seed_binned + y_test_binned )
		plt.plot( seed_binned + preds)
		plt.legend( ['data', 'pred'] )
		plt.title('300-sample prediction from validation seed')
		
    y = np.array(y_test_binned)
    y_hat = np.array(preds[0:len( y_test_binned )])
    new_err = np.linalg.norm( y - y_hat )
    nmse = np.linalg.norm( np.array(preds[0:len( y_test_binned )]) - np.array(y_test_binned) ) / (n_pred * y.var())
    
    return nmse

def test_model( model, weights_file, seed, y_test, n_pred=300, debug=False ):
    if weights_file != '':
        model.load_weights( weights_file )
        
    preds = []
    seed_binned = [ x.argmax() for x in seed[0] ]
    n_bins = seed.shape[-1]

    for i in range(n_pred):
		new_pred = model.predict( seed ).argmax()
		preds.append( new_pred )
		onehot_newpred = np.zeros( (1, 1, n_bins) )
		onehot_newpred[0, 0, new_pred] = 1
		seed = np.concatenate( ( seed[:, 1:, :],  onehot_newpred), axis=1 )

    y_test_binned = [ y.argmax() for y in y_test ]

    if debug:
		plt.plot( seed_binned + y_test_binned )
		plt.plot( seed_binned + preds)
		plt.legend( ['data', 'pred'] )
		plt.title('300-sample prediction from validation seed')

    y = np.array(y_test_binned)
    y_hat = np.array(preds[0:len( y_test_binned )])
    new_err = np.linalg.norm( y - y_hat )
    nmse = np.linalg.norm( np.array(preds[0:len( y_test_binned )]) - np.array(y_test_binned) ) / (n_pred * y.var())
    
    return nmse