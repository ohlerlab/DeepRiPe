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
def cnn_outdim(input_dim,f_len,p_len,s):
	dim = ((input_dim-f_len+1)-p_len)/s + 1
	return( math.ceil(dim) )

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
def create_diff_model1(num_task,input_len_l,input_len_r):
	K.clear_session()
	tf.set_random_seed(5005)
	left_dim=4
	right_dim=4
	num_units=50
	input_l=input_len_l
	input_r=input_len_r
	filter_lengths_left = [4,5,6,7]
	#filter_lengths_left = [4,5,6,7,8,9,10,12,15][4,5,6,7]
	#filter_lengths_right = [10,15,20,25] [2,3,4,5]
	filter_lengths_right = [2,5,7,10]
	nb_filter = 30

	left_input = Input(shape=(input_l,left_dim),name="left_input")
	right_input = Input(shape=(input_r,right_dim),name="right_input")

	left_convs_pooled = []
	for i, filter_length in enumerate(filter_lengths_left):
		left_conv1 = Conv1D(filters=nb_filter,kernel_size=filter_length, padding='valid',activation="relu")(left_input)
		left_pool1 = MaxPooling1D(pool_size=4, strides=2)(left_conv1)
		left_drop = Dropout(0.25)(left_pool1)
		left_convs_pooled.append(left_drop)
		#left_convs_pooled.append(left_pool1)
	left_merged = concatenate(left_convs_pooled,name="left_merged",axis=-2)
    #dropped_convs = Dropout(dropout)(left_merged)
	#print(left_merged.shape)

	right_convs_pooled = []
	for i, filter_length in enumerate(filter_lengths_right):
		right_conv1 = Conv1D(filters=nb_filter,kernel_size=filter_length, padding='valid',activation="relu")(right_input)
		right_pool1 = MaxPooling1D(pool_size=10, strides=5)(right_conv1)
		right_drop = Dropout(0.25)(right_pool1)
		right_convs_pooled.append(right_drop)
		#right_convs_pooled.append(right_pool1)

	right_merged = concatenate(right_convs_pooled,name="right_merged",axis=-2)
    #dropped_convs = Dropout(dropout)(conv_merged)


	merge = concatenate([left_merged,right_merged],name="merge",axis=-2)
	conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(merge)
	#merged_pool = MaxPooling1D(pool_size=4, strides=2)(conv_merged)
	merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
	merged_drop = Dropout(0.25)(merged_pool)
	merged_flat = Flatten()(merged_drop)
	#merged_flat = Flatten()(merged_pool)
	#merge = concatenate([left_flat, right_flat])
	# interpretation model
	hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=[left_input,right_input], outputs=output)
	print(model.summary())
	return model

def create_diff_model2(num_task,input_len_l,input_len_r):
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

def create_diff_model3(num_task,input_len_l,input_len_r):
	K.clear_session()
	tf.set_random_seed(5005)
	input_dim=4
	num_units=20
	filter_lengths_left = [4,5,6,7]
	#filter_lengths_left = [4,6,8,10,12]
	filter_lengths_right = [2,3,4,5]
	nb_filter = 30

	left_input = Input(shape=(input_len_l,input_dim),name="left_input")
	right_input = Input(shape=(input_len_r,input_dim),name="right_input")


	l_convs_pooled = []
	for i, filter_length in enumerate(filter_lengths_left,1):
		l_conv1 = Conv1D(filters=nb_filter,kernel_size=filter_length, padding='valid',activation="relu",name=str(i)+'st_lconv')(left_input)
		l_pool1 = MaxPooling1D(pool_size=4, strides=4,name=str(i)+'st_lpool')(l_conv1)
		l_drop1 = Dropout(0.25)(l_pool1)
		l_convs_pooled.append(l_drop1)
	l_merged = concatenate(l_convs_pooled,name="l_merged",axis=-2)

	r_convs_pooled = []
	for i, filter_length in enumerate(filter_lengths_right):
		r_conv1 = Conv1D(filters=nb_filter,kernel_size=filter_length, padding='valid',activation="relu",name=str(i)+'st_rconv')(right_input)
		r_pool1 = MaxPooling1D(pool_size=4, strides=4,name=str(i)+'st_rpool')(r_conv1)
		r_drop1 = Dropout(0.25)(r_pool1)
		r_convs_pooled.append(r_drop1)
	r_merged = concatenate(r_convs_pooled,name="r_merged",axis=-2)

	merged = concatenate([l_merged,r_merged],name="merged",axis=-2)

    #dropped_convs = Dropout(dropout)(left_merged)
	print(merged.shape)
	gru = Bidirectional(GRU(num_units,return_sequences=True),name="gru")(merged)
	#gru = Bidirectional(GRU(num_units),return_sequences=True,name="gru")(merged)
	flat = Flatten(name="flat")(gru)
	hidden1 = Dense(250, activation='relu',name="hidden1")(flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=[left_input,right_input], outputs=output)
	print(model.summary())
	return model

#only seq
def create_diff_model4(num_task,input_len_l,input_len_r):
	K.clear_session()
	tf.set_random_seed(5005)
	left_dim=4
	right_dim=4
	num_units=50
	input_l=input_len_l
	input_r=input_len_r
	#filter_lengths_left = [4,5,6,7]
	#filter_lengths_left = [4,5,6,7,8,9,10,12,15]
	#filter_lengths_right = [10,15,20,25]
	#filter_lengths_right = [2,3,4,5]
	#nb_filter = 30
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

	#left_convs_pooled = []
	#for i, filter_length in enumerate(filter_lengths_left):
		#left_conv1 = Conv1D(filters=nb_filter,kernel_size=filter_length, padding='valid',activation="relu")(left_input)
		#left_pool1 = MaxPooling1D(pool_size=4, strides=2)(left_conv1)
		#left_drop = Dropout(0.25)(left_pool1)
		#left_convs_pooled.append(left_drop)
	#merge = concatenate(left_convs_pooled,name="left_merged",axis=-2)
	#dropped_convs = Dropout(dropout)(left_merged)
	#print(left_merged.shape)

	left_conv1 = Conv1D(filters=nb_f_l[0],kernel_size=f_len_l[0], padding='valid',activation="relu",name="left_conv1")(left_input)
	left_pool1 = MaxPooling1D(pool_size=p_len_l[0], strides=s_l[0],name="left_pool1")(left_conv1)
	left_drop1 = Dropout(0.25,name="left_drop1")(left_pool1)

	conv_merged = Conv1D(filters=100,kernel_size= 5, padding='valid',activation="relu",name="conv_merged")(left_drop1)
	#merged_pool = MaxPooling1D(pool_size=4, strides=2)(conv_merged)
	merged_pool = MaxPooling1D(pool_size=10, strides=5)(conv_merged)
	merged_drop = Dropout(0.25)(merged_pool)
	merged_flat = Flatten()(merged_drop)
	#merged_flat = Flatten()(merged_pool)
	#merge = concatenate([left_flat, right_flat])
	# interpretation model
	hidden1 = Dense(250, activation='relu',name="hidden1")(merged_flat)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=[left_input,right_input], outputs=output)
	print(model.summary())
	return model


def create_diff_model5(num_task,input_len_l,input_len_r):
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
	left_conv2 = Conv1D(filters=nb_f_l[1],kernel_size=f_len_l[1], padding='valid',activation="relu",subsample_length=1,name="left_conv2")(left_drop1)
	left_pool2 = MaxPooling1D(pool_size=p_len_l[1], strides=s_l[1],name="left_pool2")(left_conv2)
	left_drop2 = Dropout(0.25,name="left_drop2")(left_pool2)
	left_flat = Flatten(name="left_flat")(left_drop2)

	right_conv1 = Conv1D(filters=nb_f_r[0],kernel_size=f_len_r[0], padding='valid',activation="relu",name="right_conv1")(right_input)
	right_pool1 = MaxPooling1D(pool_size=p_len_r[0], strides=s_r[0],name="right_pool1")(right_conv1)
	right_drop1 = Dropout(0.25,name="right_drop1")(right_pool1)
	right_conv2 = Conv1D(filters=nb_f_r[1],kernel_size=f_len_r[1], padding='valid',activation="relu",subsample_length=1,name="right_conv2")(right_drop1)
	right_pool2 = MaxPooling1D(pool_size=p_len_r[1], strides=s_r[1],name="right_pool2")(right_conv2)
	right_drop2 = Dropout(0.25,name="right_drop2")(right_pool2)
	right_flat = Flatten(name="right_flat")(right_drop2)

	merge = concatenate([left_flat, right_flat])
	# interpretation model
	hidden1 = Dense(250, activation='relu',name="hidden1")(merge)
	output = Dense(num_task, activation='sigmoid',name="output")(hidden1)
	model = Model(inputs=[left_input,right_input], outputs=output)
	print(model.summary())
	return model

def create_diff_model6(num_task,input_len_l,input_len_r):
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
		dispatcher={'create_diff_model1':create_diff_model1, 'create_diff_model2':create_diff_model2,'create_diff_model3':create_diff_model3,'create_diff_model4':create_diff_model4,'create_diff_model5':create_diff_model5,'create_diff_model6':create_diff_model6}
		try:
			model_funname=dispatcher[model_funname]
		except KeyError:
			raise ValueError('invalid input')
	model = model_funname(num_task,input_len_l,input_len_r)
	print ('compiling model')
	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.8, nesterov=True)
	rms=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=1e-6)
	adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

	#model.compile(loss='binary_crossentropy', optimizer=sgd, class_mode="binary", metrics=['accuracy', fbeta_score, precision,recall,matthews_correlation])
	#model.compile(loss='binary_crossentropy', optimizer=rms, class_mode="binary", metrics=['accuracy', fbeta_score, precision,recall,matthews_correlation])
	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy',precision,recall]) # this was used for paper
	#model.compile(loss=multitask_loss,optimizer='adam', metrics=['accuracy',precision,recall])
	#model.compile(loss='binary_crossentropy', optimizer='adadelta', class_mode="binary", metrics=['accuracy',fbeta_score, precision,recall,matthews_correlation,acc_score],show_accuracy=True)

	#model.compile(loss='binary_crossentropy', optimizer='rmsprop', class_mode="binary", metrics=['accuracy', fbeta_score])
	checkpointer = ModelCheckpoint(filepath= model_path, verbose=1, save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
	tb=TensorBoard(log_dir='./Output/logs', histogram_freq=0, write_graph=True, write_images=False)
	#earlystopper = EarlyStopping(monitor='fbeta_score', patience=50, verbose=1)
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
	#print("{}: {:.2f}%".format (model.metrics_names[1], score[1]*100))
	print("{}: {:.2f}".format (model.metrics_names[1], score[1]))
	print("{}: {:.2f}".format (model.metrics_names[2], score[2]))
	#print("{}: {:.2f}".format (model.metrics_names[4], score[4]))
	#print(history.history)
	#if plot:
		#print(plot)
		#plot_curve(history)

	#preds = model.predict([X_test_seq,X_test_region])
	#preds[preds>=0.5] = 1
	#preds[preds<0.5] = 0
	#print ("f1_score:",f1_score(y_test, preds, average='micro'))
	#print ("fbeta_score:",fbeta_score(y_test, preds))


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

	print ("ploting the curve")
	colors = []
	n_classes=y_test.shape[1]
	for i in range(0,n_classes):
		colors.append(gc.generate_new_color(colors,pastel_factor = 0.9))

	#RBPnames=["DND1","ELAVL2","ELAVL3","ELAVL4","RBM20","IGF2BP3","HNRNPC","AGO2","CPSF7","CPSF6","CPSF1","FIP1L1",
	#"CSTF2","CSTF2T","CAPRIN1","ZC3H7B","FMR1iso1","FMR1iso7","AGO1","ORF1","RBM10","MOV10","ELAVL1"]
	RBPnames=["DND1","ELAVL2","ELAVL3","ELAVL4","RBM20","IGF2BP3","AGO2","CPSF7","CPSF6","CPSF1","FIP1L1",
	"CSTF2","CSTF2T","CAPRIN1","ZC3H7B","FMR1iso1","FMR1iso7","AGO1","ORF1","RBM10","MOV10","ELAVL1"]
	roc.plot_roccurve(y_test,y_pred,colors=colors,curvenames=RBPnames)
	PR.plot_PRcurve(y_test,y_pred,colors=colors,curvenames=RBPnames)


################################################################################
# custume metric####
################################################################################

def fbeta_score(y_true, y_pred, beta=1):
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')
	# Count positive samples.
	c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
	c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
	# If there are no true samples, fix the F score at 0.
	if c3 == 0:
		return 0
    # How many selected items are relevant?
	precision = c1 / K.cast(c2, K.floatx())
    # How many relevant items are selected?
	recall = c1 / K.cast(c3, K.floatx())
    # Weight precision and recall together as a single scalar.
	beta2 = beta ** 2
	f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
	return f_score

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	#TPs=K.sum(K.round(K.clip(y_true * y_pred , 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	#TPs=K.sum(K.round(K.clip(y_ture * y_pred , 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def matthews_correlation(y_true, y_pred):

	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	numerator = (tp * tn - fp * fn)
	denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

	return numerator / (denominator + K.epsilon())

def acc_score(y_true, y_pred):
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos

	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos

	tp = K.sum(y_pos * y_pred_pos)
	tn = K.sum(y_neg * y_pred_neg)

	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg)

	numerator = (tp + tn)
	denominator = (tp + fp + tn + fn)

	return numerator / (denominator + K.epsilon())


	#################################################################


#################################################################
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

	dispatcher={'create_diff_model1':create_diff_model1, 'create_diff_model2':create_diff_model2, 'create_diff_model3':create_diff_model3,'create_diff_model4':create_diff_model4,'create_diff_model5':create_diff_model5,'create_diff_model6':create_diff_model6}
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
