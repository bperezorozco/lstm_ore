from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, LSTM, Embedding, TimeDistributedDense, Input, merge, Flatten
from keras.optimizers import Adam, SGD
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import WeightRegularizer, l2
from keras.callbacks import EarlyStopping
from keras import backend as K
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

import theano
import theano.tensor as T
from theano import shared
from theano import function
import numpy as np
import scipy.io as spio
import pickle

import matplotlib
#matplotlib.use('Agg')
from tools import get_frames, long_prediction, get_samples, get_full_prediction, get_full_prediction_aux, plot_attractors, plot_attractor, attractor_distance
from ore_models import create_bidir_model_ORE08, create_bidir_model, create_oneL_model, create_shallow_model, test_model, test_model_aux
import sys
import matplotlib.pyplot as plt
plt.style.use('seaborn-ticks')
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['figure.figsize'] = (6, 5)
#  'font.serif':'times'
labelsize = 20
params= { 
	'text.usetex':True, "font.family": "serif", "font.serif": ["Computer Modern"],
	#'font.family':'serif', 'font.serif': ["Times", "Times New Roman"], 
	'legend.fontsize':labelsize, 'axes.labelsize':labelsize, 'axes.titlesize':labelsize, 
	'xtick.labelsize' :labelsize, 'ytick.labelsize' : labelsize
}
matplotlib.rcParams.update(params)
linecolors = ["#348ABD", "#a60628"]
errs = []

experiment = ""
data_id = ""
DEBUG = False
m_train = 9886
timesteps = 100
hop = 1
n_bins = 150
val_ex = 0
test_ex = 1000
n_pred = 300
max_pred = 2000
desc = {
	'ORE07' : 'LSTM',
	'ORE08_1' : 'Bidirectional LSTM',
	'ORE08_2' : 'Input-connected Bidirectional LSTM',
	'ORE09' : 'Input-connected Bidirectional LSTM',
	'ORE10' : 'MLP'
}
codes = ['data', 'LSTM', 'BiDirLSTM', 'Input-conn BiDirLSTM', 'MLP']

attractors = {}

def mg_gen( a, b, tau, n=11000, lag=85, skip=1000 ):
    N = n + lag + 1
    x = 0.1 * np.ones( N )
    
    for t in np.arange( tau, N - 1 ):
        x[t+1] = ( 1.0 - b ) * x[t] + a * ( x[t-tau] / ( 1.0 + np.power( x[t-tau], 10.0 ) ) )

    return x[skip:]

def load_data( data_id, sigma=0.05, standardise=True ):
	if data_id == "MG":
		data = mg_gen( 0.2, 0.1, 17, n=22000 )     
		m_train = 9886
	elif data_id == "wind":	
		fname = "ds/windPredProb.mat"
		dic = spio.loadmat( fname )
		data = np.vstack( (dic['xtr'], dic['xte']) ).squeeze()
		m_train = 20000
		sigma = 0.4
	elif data_id == "tide":
		fname = "ds/tideData.mat"
		dic = spio.loadmat( fname )
		data = dic['YReals'].squeeze()
		m_train = int( 0.8 * data.shape[0] )
		sigma = 0.01
	elif data_id == "air":
		fname = "ds/biomed.mat"
		dic = spio.loadmat( fname )
		data = dic['air'].squeeze()
		m_train = int( 0.8 * data.shape[0] )
		sigma = 1.
	elif data_id == "heart":
		fname = "ds/biomed.mat"
		dic = spio.loadmat( fname )
		data = dic['ecg'].squeeze()
		m_train = int( 0.8 * data.shape[0] )
		sigma = 1.
	elif data_id == "tide_new":
		fname = "ds/tide_new.npy"
		data = np.load( fname )
		m_train = 10000
		sigma = 0.01
	else:
		raise Exception('Invalid dataset id. Choose from: [MG, wind, tide, air, heart, tide_new]. Terminating...')

	m = data.shape[0]
	if sigma > 0.:
		data_x = data + sigma * np.random.randn(m)
	else:
		data_x = data

	if DEBUG:
		plt.plot( data[:1000] )
		plt.title("First 1000 samples of data")
		plt.savefig(data_id + "samples.pdf")
		plt.clf()

	return (data, data_x, m)

def bin_data( data, data_x, m, n_bins=150, return_one_hot=True ):
	bins = np.linspace( data_x.min(), data_x.max(), n_bins )
	binned_x = np.digitize( data_x, bins ) - 1;
	binned_y = np.digitize( data, bins ) - 1;
	classes = np.zeros( (m, n_bins) )
	classes_pure = np.zeros( (m, n_bins) )

	for i, j, k in zip( range( 0, m ), binned_x.tolist(), binned_y.tolist() ):
	    classes[i, j] = 1
	    classes_pure[i, k] = 1
	   
	if DEBUG:
		print "Printing data..."
		plt.plot(binned_x[:1000])
		#plt.plot(binned_y[:1000])
		plt.plot( classes_pure.argmax(axis=-1)[:1000] )
		plt.title('Data with and without Gaussian noise')
		plt.legend(['noisy', 'noiseless'])
		plt.savefig(data_id + "_binned_samples.pdf")
		plt.clf()

	if return_one_hot:
		return (classes, classes_pure, bins)
	else:
		return (binned_x, binned_y, bins)

def prepare_frames( binned_data, return_full=True, squeezed=False ):
	frames = get_frames( binned_data, timesteps, hop )
	if return_full:
		if squeezed:
			return frames.squeeze()
		else:
			return np.swapaxes( frames, 1, 2 ).astype('float32')
	else:
		if squeezed:
			print frames.shape
			return frames[:, :, -1]
		else:
			return np.swapaxes( frames, 1, 2 ).astype('float32')[:, -1, :]

def split_tr_te( classes, classes_pure, m_train=m_train, squeezed=False ):
	y_train = prepare_frames( classes.T[:, 1:], return_full=False, squeezed=squeezed )[:m_train]
	y_test = prepare_frames( classes_pure.T[:, 1:], return_full=False, squeezed=squeezed )[m_train:]
	X_test = prepare_frames( classes.T[:, :-1], squeezed=squeezed )[m_train:]
	X_train = prepare_frames( classes.T[:, :-1], squeezed=squeezed )[:m_train]

	return (X_train, y_train, X_test, y_test)

def create_aux_input( data_x ):
	X_train_aux = prepare_frames( data_x[None, :-1] ).squeeze()
	return ( X_train_aux[:m_train], X_train_aux[m_train:] )

def create_model( experiment, hidden_units, l ):
	if experiment == "ORE07":
		model = create_oneL_model( hidden_units, l, timesteps, n_bins )
	elif experiment == "ORE08_1":
		model = create_bidir_model_ORE08( hidden_units, l, timesteps, n_bins, concat_input=False )
	elif experiment == "ORE08_2":
		model = create_bidir_model_ORE08( hidden_units, l, timesteps, n_bins, concat_input=True )
	elif experiment == "ORE09":
		model = create_bidir_model( hidden_units, l, timesteps, n_bins, concat_input=True )
	elif experiment == "ORE10":
		model = create_shallow_model( hidden_units, l, timesteps, n_bins )
	else:
		raise Exception('Invalid experiment id. Choose from [ORE07, ORE08_1, ORE08_2, ORE09, ORE10] ')

	return model

def train_model( model, X_train, y_train, hu, lam ):
	early = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=1, verbose=1, mode='auto')
	hist = model.fit(X_train, y_train, nb_epoch=20, validation_split=0.2, batch_size=50, callbacks=[early])
	return model

def run_train( X_train, y_train, X_test, y_test ):
	err = 10000000
	hu1 = [ 75, 200, 300, 500 ]
	lam = np.logspace(-7, -3, num=5)
	for hidden_units in hu1:
		for l in lam:
			model = train_model( create_model( experiment, hidden_units, l ), X_train, y_train, hu1, lam )
			new_err = test_model( model, '', X_test[val_ex:val_ex+1], y_test[val_ex:val_ex+n_pred], debug=True )
			errs.append(new_err)
			filename = "/%s_%s_h1%s_l%s.h5" % ( experiment, data_id, hidden_units, l )
			model.save_weights( "results" + filename, overwrite=True )
			plt.savefig( "figs" + filename + ".pdf" )
			plt.clf()
			print "--------------------------------------------"
			print "hu1: " + str( hidden_units )
			print "lambda: " + str(l)
			print "NMSE: " + str( new_err )
			print "--------------------------------------------"

	print experiment
	print errs
	print "BEST ERROR: " + str( err )

def run_train_aux( X_train, y_train, X_test, y_test, X_train_aux, X_test_aux, bins ):
	err = 10000000
	hu1 = [ 75, 200, 300, 500 ]
	lam = np.logspace(-7, -3, num=5)

	for hidden_units in hu1:
		for l in lam:
			model = train_model( create_model( experiment, hidden_units, l ), [X_train, X_train_aux], y_train, hu1, lam )
			new_err = test_model_aux( model, '', X_test[val_ex:val_ex+1], y_test[val_ex:val_ex+n_pred], X_test_aux[val_ex:val_ex+1], bins, debug=True )
			errs.append(new_err)
			filename = "/%s_%s_h1%s_l%s.h5" % ( experiment, data_id, hidden_units, l )
			model.save_weights( "results" + filename, overwrite=True )
			plt.savefig( "figs" + filename + ".pdf" )
			plt.clf()
			print "--------------------------------------------"
			print "hu1: " + str( hidden_units )
			print "lambda: " + str(l)
			print "NMSE: " + str( new_err )
			print "--------------------------------------------"

	print experiment
	print errs
	print "BEST ERROR: " + str( err )

def load_and_test_model( fname_test, hu, l, seed, y_test, seed_aux=None, bins=None, title=None ):
	global experiment
	model = create_model( experiment, hu, l )
	try:
		model.load_weights( fname_test )
	except IOError as detail:
		print detail.args
		raise IOError( "Test file " + fname_test + " not found." )

	if seed_aux != None and bins != None:
		nmse = test_model_aux( model, '', seed, y_test, seed_aux, bins, n_pred=n_pred, debug=True )
	else:
		nmse = test_model( model, '', seed, y_test, n_pred=n_pred, debug=True )
	print "TEST RESULT: " + str( nmse )
	if title:
		plt.title( title )
	else:
		plt.title( "Forecasting " + str( n_pred ) + " samples." )
	plt.savefig( "test_figs/" + str(n_pred) + fname_test.split("/")[-1]  + ".pdf" )
	plt.clf()

	return nmse

def run_test( test_args, mode="test" ):
	global experiment, data_id, n_pred, attractors

	experiment = test_args[0]
	data_id = test_args[1]
	hu = int( test_args[2] )
	l = float( test_args[3] )
	fname_test = test_args[4]
	n_pred = int( test_args[5] )

	try:
		seed = np.load( './test/' + data_id + '_test_seed.npy' )
		y_true = np.load( './test/' + data_id + '_y.npy' )[:n_pred]
		seed_aux = np.load( './test/' + data_id + '_test_seed_aux.npy' )
		bins = np.load( './test/' + data_id + '_bins.npy' )
		print 'Loading test data....'
	except IOError as detail:
		print detail.args
		print 'Generating test data....'
		(data, data_x, m) = load_data( data_id )
		(classes, classes_pure, bins) = bin_data( data, data_x, m, n_bins )
		(X_train, y_train, X_test, y_test) = split_tr_te( classes, classes_pure )
		X_train = None
		y_train = None

		test_ex = np.random.randint( 1000 )
		print 'Test sample #' + str( test_ex ) + ' has been chosen at random.'

		(X_train_aux, X_test_aux) = create_aux_input( data_x )
		seed = X_test[test_ex:test_ex+1] #seed.shape = (1, timesteps, n_bins)
		y_true = y_test[test_ex:test_ex+max_pred]
		seed_aux = X_test_aux[test_ex:test_ex+1]

		np.save( './test/' + data_id + '_test_seed', seed )
		np.save( './test/' + data_id + '_y', y_true )
		np.save( './test/' + data_id + '_test_seed_aux', seed_aux )
		np.save( './test/' + data_id + '_bins', bins )
		y_true = y_true[:n_pred]

	print 'Testing: ' + fname_test 

	if mode == "test":
		title = 'Forecasting {} \'{}\' samples using {} with {} neurons'.format( n_pred, data_id, desc[experiment], hu )

		if experiment in ["ORE09",]:
			print "Testing model with auxiliary inputs..."
			return load_and_test_model( fname_test, hu, l, seed, y_true, seed_aux=seed_aux, bins=bins, title=title )
		else:
			print "Testing model without auxiliary inputs..."
			return load_and_test_model( fname_test, hu, l, seed, y_true, title=title )

	elif mode == "extra":
		model = create_model( experiment, hu, l )
		try:
			model.load_weights( fname_test )
		except IOError as detail:
			print detail.args
			raise IOError( "Test file " + fname_test + " not found." )

		y_true = np.concatenate( (seed[0], y_true) )

		for feed_uncertainty in [True, False]:
			prep = ''
			if not feed_uncertainty:
				prep = 'no'

			title = 'Forecasting {} \'{}\' samples using {} with {} neurons and {} uncertainty propagation'.format( n_pred, data_id, desc[experiment], hu, prep )

			if experiment in ["ORE09",]:
				y_hat = np.array( get_full_prediction_aux( model, '', seed, seed_aux, bins, n_pred=n_pred, feed_uncertainty=feed_uncertainty ) ).squeeze()
			else:
				y_hat = np.array( get_full_prediction( model, '', seed, n_pred=n_pred, feed_uncertainty=feed_uncertainty ) ).squeeze()
			y_hat = np.concatenate( (seed.squeeze(), y_hat), axis=0 )

			np.save( './preds/{}_{}uncertainty'.format( fname_test.split('/')[-1][:-3], prep ), y_hat )

			y_hat_norm = y_hat / y_hat.max( axis=1 )[:, None]
			
			plt.subplot(2, 1, 1)
			plt.title( title )
			plt.imshow( y_hat_norm.T, cmap='GnBu', origin='lower' )
			plt.subplot(2, 1, 2)
			plt.title('Ground truth')
			plt.imshow( y_true.T, cmap='GnBu', origin='lower' )
			plt.savefig( "test_figs/uncertainty_" + str( feed_uncertainty ) + "_" + str(n_pred) + fname_test.split("/")[-1]  + ".pdf" )
			plt.clf()

			if experiment in ["ORE09",]:
				continue

			key = '{}_{}_{}uncertainty'.format( n_pred, data_id, prep )
			if key in attractors:
				attractors[key].append( y_hat.argmax(axis=1) )
			else:
				attractors[key] = []
				attractors[key].append( y_true.argmax(axis=1) )
				attractors[key].append( y_hat.argmax(axis=1) )




def main():
	if len( sys.argv ) < 5:
		raise Exception('You forgot to send command line args: [ExpId, DataId, bool DEBUG, Mode=train or test or extra or att_mse]')

	global experiment, data_id, DEBUG, n_pred, attractors
	experiment = sys.argv[1]
	data_id = sys.argv[2]
	if sys.argv[3] == 'False':
		DEBUG = False

	mode = sys.argv[4]
	att_lag = 10

	if mode == "train":
		n_pred = 300
		(data, data_x, m) = load_data( data_id )
		(classes, classes_pure, bins) = bin_data( data, data_x, m, n_bins )
		(X_train, y_train, X_test, y_test) = split_tr_te( classes, classes_pure )

		if experiment in ["ORE09",]:
			print "Training model with auxiliary inputs..."
			(X_train_aux, X_test_aux) = create_aux_input( data_x )
			run_train_aux( X_train, y_train, X_test, y_test, X_train_aux, X_test_aux, bins )
		else:
			print "Training model without auxiliary inputs..."
			run_train( X_train, y_train, X_test, y_test )

	elif mode == "test":
		print "Disregarding command line arguments... " + experiment + ", " + data_id
		with open( 'test_files.txt', 'r' ) as test_file, open( 'test_results.txt', 'wb' ) as test_results:
			for line in test_file:
				test_args = line.split(',')
				nmse = run_test( test_args )
				res = '{} with {}, {} samples for {}: {}\n'.format(test_args[0], test_args[2], int(test_args[-1]), test_args[1], nmse)
				print res
				test_results.write( res )
			test_file.close()
			test_results.close()
	elif mode == "extra":
		print "Disregarding command line arguments... " + experiment + ", " + data_id
		with open( 'test_files.txt', 'r' ) as test_file:
			for line in test_file:
				test_args = line.split(',')
				run_test( test_args, mode="extra" )
				#res = '{} with {}, {} samples for {}: {}\n'.format(test_args[0], test_args[2], int(test_args[-1]), test_args[1], nmse)
				#print res
			test_file.close()

		output = open('attractors.pkl', 'wb')
		pickle.dump(attractors, output)
		output.close()

		for key, val in attractors.iteritems():
			plot_attractors( val, lbls=codes, title=key )
	elif mode == "att_mse":
		for i_att in range(1, 4):
			f = open('attractors/attractors_run{}.pkl'.format(i_att))
			att = pickle.load(f)
			f.close()

			###merge attractor dictionary with the tide_new attractor dictionary which was generated separately
			f = open('attractors/attractors_tide_new_run{}.pkl'.format(i_att))
			att_tide = pickle.load(f)
			f.close()

			att.update( att_tide ) 

			fres = open( 'attractors/uncertainty_res{}.txt'.format(i_att), 'wb' )
			ftable = open( 'attractors/table_uncertainty_res{}.txt'.format(i_att), 'wb' )
			ftable2 = open( 'attractors/table_no_uncertainty_res{}.txt'.format(i_att), 'wb' )
			ftable3 = open( 'attractors/table_dtw_no_uncertainty_res{}.txt'.format(i_att), 'wb' )
			ftable4 = open( 'attractors/table_dtw_uncertainty_res{}.txt'.format(i_att), 'wb' )
			ftable5 = open( 'attractors/table_att_no_uncertainty_res{}.txt'.format(i_att), 'wb' )
			ftable6 = open( 'attractors/table_att_uncertainty_res{}.txt'.format(i_att), 'wb' )

			ftabs = [ftable, ftable2, ftable3, ftable4, ftable5, ftable6]

			for i_pred in [100, 500, 1000]:
				for ftab in ftabs:
					ftab.write('\\begin{table}\\begin{center}\\begin{tabular}{|' + 'c|'*6 + '}\n')
					ftab.write('\\hline\\textbf{data}&\\textbf{LSTM}&\\textbf{BidirLSTM}&\\textbf{Input-BidirLSTM}&\\textbf{MLP}\\\\\n')
				
				for i_data in ['MG', 'tide_new', 'heart']:
					for ftab in ftabs:
						ftab.write('\\hline {}'.format(i_data))

					key = '{}_{}_uncertainty'.format(i_pred, i_data)
					key2 = '{}_{}_nouncertainty'.format(i_pred, i_data)
					ts = att[key]
					ts2 = att[key2]
					y_true = ts[0]
					y_true_seed = y_true
					y_true = y_true[timesteps:]

					for i, (y_hat, y_hat_nounc) in enumerate(zip(ts[1:], ts2[1:])):
						y_hat_seed = y_hat 
						y_hat_nounc_seed = y_hat_nounc
						y_hat = y_hat[timesteps:]
						y_hat_nounc = y_hat_nounc[timesteps:]
						Z = i_pred * y_true.var()
						Z2 = ( i_pred - 2*att_lag ) * y_true.var()

						nmse = np.linalg.norm( y_hat - y_true ) / Z
						nmse2 = np.linalg.norm( y_hat_nounc - y_true ) / Z

						dtw, path = fastdtw( y_true, y_hat, dist=euclidean )
						dtw /= Z
						dtw2, path2 = fastdtw( y_true, y_hat_nounc, dist=euclidean )
						dtw2 /= Z
						y_true_dtw = []
						y_hat_dtw = []
						y_true_nounc_dtw = []
						y_hat_nounc_dtw = []

						for (k, j), (k2, j2) in zip(path, path2):
							y_true_dtw.append( y_true[k] )
							y_hat_dtw.append( y_hat[j] )
							y_true_nounc_dtw.append( y_true[k2] )
							y_hat_nounc_dtw.append( y_hat_nounc[j2] )

						plt.plot( y_true_dtw, color=linecolors[0] )
						plt.plot( y_hat_dtw, color=linecolors[1] )
						legend = plt.legend(['data', 'pred'])
						legend.get_frame().set_facecolor((1, 1, 1, 0.8))
						plt.axvline( 100, color='k' )
						plt.tight_layout()
						plt.savefig('./attractors/dtw{}_{}_{}_{}.pdf'.format(i_att, i_data, i_pred, codes[i+1]))
						plt.clf()
						plt.plot( y_true_nounc_dtw, color=linecolors[0] )
						plt.plot( y_hat_nounc_dtw, color=linecolors[1] )
						plt.legend(['data', 'pred'])
						plt.axvline( 100, color='k' )
						plt.tight_layout()
						plt.savefig('./attractors/dtw_nounc{}_{}_{}_{}.pdf'.format(i_att, i_data, i_pred, codes[i+1]))
						plt.clf()

						ad = attractor_distance( y_true, y_hat ) / Z2
						ad2 = attractor_distance( y_true, y_hat_nounc ) /Z2

						ftable.write('&{}'.format(nmse))
						ftable2.write('&{}'.format(nmse2))
						ftable3.write('&{}'.format(dtw))
						ftable4.write('&{}'.format(dtw2))
						ftable5.write('&{}'.format(ad))
						ftable6.write('&{}'.format(ad2))

						tmp = '{} NMSE for {} preds using {} and forecasting with uncertainty: {}\n'.format(i_data, i_pred, codes[i+1],nmse)
						tmp2 = '{} NMSE for {} preds using {} and forecasting with no uncertainty: {}\n'.format(i_data, i_pred, codes[i+1],nmse2)
						tmp3 = '{} DTW distance for {} preds using {} and forecasting with uncertainty: {}\n'.format(i_data, i_pred, codes[i+1],dtw)
						tmp4 = '{} DTW distance for {} preds using {} and forecasting with no uncertainty: {}\n'.format(i_data, i_pred, codes[i+1],dtw2)
						tmp5 = '{} attractor distance for {} preds using {} and forecasting with uncertainty: {}\n'.format(i_data, i_pred, codes[i+1],ad)
						tmp6 = '{} attractor distance for {} preds using {} and forecasting with no uncertainty: {}\n'.format(i_data, i_pred, codes[i+1],ad2)

						print tmp
						print tmp2
						print tmp3
						print tmp4
						print tmp5
						print tmp6

						fres.write(tmp)
						fres.write(tmp2)
						fres.write(tmp3)
						fres.write(tmp4)
						fres.write(tmp5)
						fres.write(tmp6)

						title = '{}-sample forecast for {} dataset using {} with uncertainty propagation'.format( i_pred, i_data, codes[i+1] )

						#plt.title(title)
						plt.plot( y_true_seed, color=linecolors[0] )
						plt.plot( y_hat_seed, color=linecolors[1] )
						plt.axvline( 100, color='k' )
						plt.xlim( [0, i_pred + timesteps] )
						plt.legend(['data', 'pred'])
						plt.tight_layout()
						plt.savefig('./attractors/run{}_{}_{}_{}.pdf'.format(i_att, i_data, i_pred, codes[i+1]))
						plt.clf()

						title2 = '{}-sample forecast for {} dataset using {} without uncertainty propagation'.format( i_pred, i_data, codes[i+1] )

						#plt.title(title2)
						plt.plot( y_true_seed, color=linecolors[0] )
						plt.plot( y_hat_nounc_seed, color=linecolors[1] )
						plt.axvline( 100, color='k' )
						plt.xlim( [0, i_pred + timesteps] )
						plt.legend(['data', 'pred'])
						plt.tight_layout()
						plt.savefig('./attractors/run{}_{}_{}_{}_nounc.pdf'.format(i_att, i_data, i_pred, codes[i+1]))
						plt.clf()
					for ftab in ftabs:
						ftab.write('\\\\\n')

				for ftab in ftabs:
					ftab.write('\\hline\\end{tabular}\\end{center}\\end{table}\n\n\n')

			fres.close()
			ftable.close()
			ftable2.close()
			ftable3.close()
			ftable4.close()

	else:
		raise Exception("Mode not recognised. Choose from [train, test, extra].")



if __name__ == "__main__":
	main()