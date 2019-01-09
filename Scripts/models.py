import sys
import os
import numpy as np
import h5py
import scipy.io
np.random.seed(7) # for reproducibility

import tensorflow as tf
#tf.python.control_flow_ops = tf


from keras.utils import plot_model
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional
import keras.backend as K

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
#from keras.utils.layer_utils import print_layer_shapes

from sklearn.metrics import f1_score

path_to_scripts="./Scripts/"
sys.path.append(path_to_scripts)
import roc_func as roc
import PR_func as PR
import generate_color as gc

import math



################################################################################
# Accessry functions
################################################################################
def create_class_weight(labels_dict,total,mu=0.15):
    keys = labels_dict.keys()
    class_weight = dict()
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight


################################################################################
# loading data
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################
def load_data_merged(path_to_data):

	data = h5py.File(path_to_data,'r')
	#X_train = np.transpose(np.array(data['train_in']),axes=(2,0,1))
	X_train_seq = np.transpose(np.array(data['train_in_seq']),axes=(0,2,1))
	X_train_region = np.transpose(np.array(data['train_in_region']),axes=(0,2,1))
	#y_train = np.array(data['train_out']).T
	y_train = np.array(data['train_out'])

	X_valid_seq = np.transpose(np.array(data['valid_in_seq']),axes=(0,2,1))
	X_valid_region = np.transpose(np.array(data['valid_in_region']),axes=(0,2,1))
	y_valid = np.array(data['valid_out'])

	X_test_seq = np.transpose(np.array(data['test_in_seq']),axes=(0,2,1))
	X_test_region = np.transpose(np.array(data['test_in_region']),axes=(0,2,1))
	y_test = np.array(data['test_out'])

	data.close()

	return X_train_seq, X_train_region, y_train, X_valid_seq, X_valid_region, y_valid, X_test_seq, X_test_region ,y_test

def load_data(path_to_data):

	data = h5py.File(path_to_data,'r')
	#X_train = np.transpose(np.array(data['train_in']),axes=(2,0,1))
	X_train = np.transpose(np.array(data['train_in']),axes=(0,2,1))
	#y_train = np.array(data['train_out']).T
	y_train = np.array(data['train_out'])

	X_valid = np.transpose(np.array(data['valid_in']),axes=(0,2,1))
	y_valid = np.array(data['valid_out'])

	X_test = np.transpose(np.array(data['test_in']),axes=(0,2,1))
	y_test = np.array(data['test_out'])

	data.close()

	return X_train,y_train,X_valid,y_valid,X_test,y_test

################################################################################
# Creating model
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################

### Model only using sequence data
def create_model1(num_task,input_len_l,input_len_r):
	K.clear_session()
	tf.set_random_seed(5005)
	left_dim=4
	right_dim=4
	num_units=50
	input_l=input_len_l
	input_r=input_len_r

	nb_f_l=[90,100]
	f_len_l=[7,7]
	p_len_l=[4,10]
	s_l=[2,5]
	nb_f_r=[90,100]
	f_len_r=[7,7]
	p_len_r=[10,10]
	s_r=[5,5]

	left_input = Input(shape=(input_l,left_dim),name="left_input")
	right_input = Input(shape=(input_r,right_dim),name="right_input")

	left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
	left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
	left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

	conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(left_drop1)
	merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
	merged_drop = Dropout(0.25)(merged_pool)
	merged_flat = Flatten()(merged_drop)

	hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=[left_input,right_input], outputs=output)
	print(model.summary())
	return model

### Model using both sequence data and region data
def create_model2(num_task,input_len_l,input_len_r):
	K.clear_session()
	tf.set_random_seed(5005)
	left_dim=4
	right_dim=4
	num_units=50
	input_l=input_len_l
	input_r=input_len_r

	nb_f_l=[90,100]
	f_len_l=[7,7]
	p_len_l=[4,10]
	s_l=[2,5]
	nb_f_r=[90,100]
	f_len_r=[7,7]
	p_len_r=[10,10]
	s_r=[5,5]

	left_input = Input(shape=(input_l,left_dim),name="left_input")
	right_input = Input(shape=(input_r,right_dim),name="right_input")

	left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
	left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
	left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

	right_conv1 = Conv1D(filters=nb_f_r[0],kernel_size=f_len_r[0], padding='valid',activation="relu",name="right_conv1")(right_input)
	right_pool1 = MaxPooling1D(pool_size=p_len_r[0], strides=s_r[0],name="right_pool1")(right_conv1)
	right_drop1 = Dropout(0.25,name="right_drop1")(right_pool1)

	merge = concatenate([left_drop1,right_drop1],name="merge",axis=-2)
	conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(merge)
	#merged_pool = MaxPooling1D(pool_size=4, strides=2)(conv_merged)
	merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
	merged_drop = Dropout(0.25)(merged_pool)
	merged_flat = Flatten()(merged_drop)
	hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=[left_input,right_input], outputs=output)
	print(model.summary())
	return model

### model using GRU
def create_model4(num_task,input_len_l,input_len_r):
	K.clear_session()
	tf.set_random_seed(5005)
	left_dim=4
	right_dim=4
	num_units=60
	input_l=input_len_l
	input_r=input_len_r

	nb_f_l=[90,100]
	f_len_l=[7,7]
	p_len_l=[4,10]
	s_l=[2,5]
	nb_f_r=[90,100]
	f_len_r=[7,7]
	p_len_r=[10,10]
	s_r=[5,5]

	left_input = Input(shape=(input_l,left_dim),name="left_input")
	right_input = Input(shape=(input_r,right_dim),name="right_input")


	left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
	left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
	left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

	right_conv1 = Conv1D(filters=nb_f_r[0],kernel_size=f_len_r[0], padding='valid',activation="relu",name="right_conv1")(right_input)
	right_pool1 = MaxPooling1D(pool_size=p_len_r[0], strides=s_r[0],name="right_pool1")(right_conv1)
	right_drop1 = Dropout(0.25,name="right_drop1")(right_pool1)

	merge = concatenate([left_drop1,right_drop1],name="merge",axis=-2)

	gru = Bidirectional(GRU(num_units,return_sequences=True),name="gru")(merge)
	#gru = Bidirectional(GRU(num_units),return_sequences=True,name="gru")(merged)
	flat = Flatten(name="flat")(gru)
	hidden1 = Dense(250, activation='relu',name="hidden1")(flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=[left_input,right_input], outputs=output)
	print(model.summary())
	return model

################################################################################
# Training the model
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################

def train_diff_model(data_path, model_funname,res_path, model_name,num_task,input_len_l,input_len_r, num_epoch, batchsize, model_path="./weights.hdf5",plot=False):
	filter_lengths = [4,5]
	print ('creating model')
	if isinstance(model_funname, str):
		dispatcher={'create_model1':create_model1, 'create_model2':create_model2,'create_model3':create_model3}
		try:
			model_funname=dispatcher[model_funname]
		except KeyError:
			raise ValueError('invalid input')
	model = model_funname(num_task,input_len_l,input_len_r)
	print ('compiling model')
	adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',precision,recall])
	checkpointer = ModelCheckpoint(filepath= model_path, verbose=1, save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
	#tb=TensorBoard(log_dir='./Output/logs', histogram_freq=0, write_graph=True, write_images=False)

	print ('loading data')
	X_train_seq, X_train_region, y_train, X_valid_seq, X_valid_region, y_valid, X_test_seq, X_test_region ,y_test=load_data_merged(data_path)

	print("left shape:",X_train_seq.shape)
	print("right shape:",X_train_region.shape)

	total=y_train.shape[0]
	labels_dict=dict(zip(range(num_task),[sum(y_train[:,i]) for i in range(num_task)]))
	class_weight=create_class_weight(labels_dict,total,mu=0.5)

	print ('fitting the model')
	history = model.fit([X_train_seq,X_train_region], y_train, epochs=num_epoch, batch_size=batchsize,validation_data=([X_valid_seq,X_valid_region],y_valid), class_weight=class_weight, verbose=2, callbacks=[checkpointer,earlystopper,tb])

	print ('saving the model')
	model.save(os.path.join(res_path, model_name + ".h5"))

	print ('testing the model')
	score = model.evaluate([X_test_seq,X_test_region],y_test)

	print(model.metrics_names)

	for i in range(len(model.metrics_names)):
		print(str(model.metrics_names[i]) + ": " + str(score[i]))

	print("{}: {:.2f}".format (model.metrics_names[0], score[0]))
	print("{}: {:.2f}".format (model.metrics_names[1], score[1]))
	print("{}: {:.2f}".format (model.metrics_names[2], score[2]))


################################################################################
# Testing the model
#
# Input: path to file (consist of train, valid and test data)
#
################################################################################

def test_model(output_path,data_path,model_path,model_funname,res_path,model_name,input_len_l,input_len_r,num_task,nb_f_r,nb_f_l,f_len_r,f_len_l,p_len_r,p_len_l,s_l,s_r):
	print('test the model and plot the curve')
	model = load_model(os.path.join(res_path,model_name+".h5"), custom_objects={'precision':precision,'recall':recall})
	data = h5py.File(data_path,'r')
	X_test_seq = np.transpose(np.array(data['test_in_seq']),axes=(0,2,1))
	X_test_region = np.transpose(np.array(data['test_in_region']),axes=(0,2,1))
	y_test = np.array(data['test_out'])
	data.close()

	print ('predicting on test data')
	if isinstance(model.input, list):
		y_pred = model.predict([X_test_seq,X_test_region], verbose=1)
		model.evaluate([X_test_seq,X_test_region], y_test)
	else:
		y_pred = model.predict(X_test_seq, verbose=1)
		model.evaluate(X_test_seq, y_test)

	print ("saving the prediction to " + output_path)
	f = h5py.File(output_path, "w")
	f.create_dataset("y_pred", data=y_pred)
	f.close()


################################################################################
# custume metric####
################################################################################

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

################################################################################
# main function
################################################################################
def main():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-nt', dest='num_task', default=None, type=int, help='number of tasks')
	parser.add_argument('-ll', dest='input_len_l', default=None, type=int, help='left input length')
	parser.add_argument('-rl', dest='input_len_r', default=None, type=int, help='right input length')
	parser.add_argument('-ne', dest='num_epoch', default=None, type=int, help='number of epochs')
	parser.add_argument('-bs', dest='batchsize', default=None, type=int, help='Batch size')
	parser.add_argument('-dp', dest='data_path', default=None, type=str, help='path to the data')
	parser.add_argument('-op', dest='output_path', default=None, type=str, help='path to the output')
	parser.add_argument('-mp', dest='model_path', default=None, type=str, help='path to the model')
	parser.add_argument('-pp', dest='prediction_path', default=None, type=str, help='path to the prediction')
	parser.add_argument('-fun', dest='model_funname', default=None, type=str, help='name of the model')
	parser.add_argument('-name', dest='model_name', default=None, type=str, help='name of the model')
	parser.add_argument('-nfl', nargs="+", default=None, type=int, help='number of filters of left branch')
	parser.add_argument('-nfr', nargs="+", default=None, type=int, help='number of filters of right branch')
	parser.add_argument('-fll',nargs="+", default=None, type=int, help='filter lengths of left branch')
	parser.add_argument('-flr',nargs="+", default=None, type=int, help='filter lengths of right branch')
	parser.add_argument('-pll', nargs="+",default=None, type=int, help='pool_lengths of left branch')
	parser.add_argument('-plr', nargs="+",default=None, type=int, help='pool_lengths of right branch')
	parser.add_argument('-sl', nargs="+",default=None, type=int, help='strides of left branch')
	parser.add_argument('-sr', nargs="+",default=None, type=int, help='strides of right branch')
	parser.add_argument('-p',dest='plot', default=False, type=bool, help='plot the learning curve')
	parser.add_argument('-t',dest='test', default=False, type=bool, help='test the model')
	args = parser.parse_args()

	dispatcher={'create_model1':create_model1, 'create_model2':create_model2, 'create_model3':create_model3}
	try:
		funname=dispatcher[args.model_funname]
	except KeyError:
		raise ValueError('invalid input')

	nf_list_l = args.nfl
	fl_list_l = args.fll
	pl_list_l = args.pll
	s_list_l= args.sl

	nf_list_r = args.nfr
	fl_list_r = args.flr
	pl_list_r = args.plr
	s_list_r = args.sr

	train_diff_model(data_path=args.data_path, model_funname=funname, res_path=args.output_path, model_name=args.model_name,num_task=args.num_task, input_len_l=args.input_len_l,input_len_r=args.input_len_r,
	num_epoch=args.num_epoch, batchsize=args.batchsize, model_path=args.model_path,plot=args.plot)
	test_flag=args.test
	print("test flag:",test_flag)
	if test_flag:
		print("testing the model and plot the curves")
		test_model(output_path=args.prediction_path,data_path=args.data_path,model_path=args.model_path,model_funname=funname,res_path=args.output_path,model_name=args.model_name,
		input_len_l=args.input_len_l,input_len_r=args.input_len_r, num_task=args.num_task,nb_f_l=nf_list_l,f_len_l=fl_list_l,p_len_l=pl_list_l,s_l=s_list_l,
		nb_f_r=nf_list_r,f_len_r=fl_list_r,p_len_r=pl_list_r,s_r=s_list_r)

if __name__ == '__main__':
	main()
