## Input format: code NameofTrainingSeq GpuNo. Path2Data Path2sav Type

from __future__ import print_function
import sys
import os as os
os.environ['THEANO_FLAGS'] = "device=gpu" + sys.argv[2]
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, BatchNormalization, Dropout,Dense, Reshape,Flatten, Deconvolution2D
from keras.models import Model
import numpy as np
from keras.models import model_from_json
from keras import backend as K
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import glob as glob
import numpy as np
import random
from scipy.stats import threshold

######### Set path where data (dense flow and raw frames) are stored
path=sys.argv[3]
#path='/Neutron9/bharat.b/data_pre/'


######### Set path where models must be stored
path2sav=sys.argv[4]
#path2sav='/Neutron9/bharat.b/folder5/'

name=sys.argv[1]
suffix='ver4_2'
fr_rate=30
if not(os.path.isdir(path2sav)):
	os.mkdir(path2sav)

if not(os.path.isdir(path2sav+name)):
	os.mkdir(path2sav+name)

if not(os.path.isdir(path2sav+name+'/models_'+suffix)):
	os.mkdir(path2sav+name+'/models_'+suffix)

'''
if not(os.path.isdir(path2sav+name+'/mean_std_'+suffix)):
	os.mkdir(path2sav+name+'/mean_std_'+suffix)
'''

########## Set parameters here
inp_sz=(36,64)
drop=0
epoch=200
opti='adam'
no_layer=[8,18]
no_auto=20
no_sp_per_vid=20
batch_sz=2000

####Set type here, x for flow_x, y for flow_y and <blank> for frame
if(len(sys.argv)==6):
	type=sys.argv[5] #or 'y'
else:
	type=''


#Architecture

# Define layer 1
input_img1 = Input(shape=(1, inp_sz[0], inp_sz[1]))
x1 = Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same')(input_img1)
x1 = Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same')(x1)
x1 = BatchNormalization(mode=0, axis=1) (x1)
y1 = MaxPooling2D((2,2), border_mode='valid')(x1)
x1 = Dropout(drop)(y1)
x1 = Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same')(x1)
x1 = Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same')(x1)
x1 = BatchNormalization(mode=0, axis=1) (x1)
x1 = UpSampling2D((2, 2))(x1)
decoded1 = Convolution2D(1,3,3, border_mode='same', activation='sigmoid')(x1)

#Make layer1
autoencoder1 = Model(input_img1, decoded1)
autoencoder1.compile(optimizer=opti, loss='binary_crossentropy')
encoder1 = Model(input=input_img1, output=y1)
json_string = autoencoder1.to_json()
open(path2sav+'autoencoder1_temp.json', 'w').write(json_string)


# Define layer2
input_img2 = Input(shape=(encoder1.output_shape[1], encoder1.output_shape[2], encoder1.output_shape[3]))
x2 = Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same')(input_img2)
x2 = Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same')(x2)
x2 = BatchNormalization(mode=0, axis=1) (x2)
y2 = MaxPooling2D((2,2), border_mode='valid')(x2)
x2 = Dropout(drop)(y2)
x2 = Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same')(x2)
x2 = Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same')(x2)
x2 = BatchNormalization(mode=0, axis=1)(x2)
x2 = UpSampling2D((2, 2))(x2)
decode2 = Convolution2D(no_layer[1],3,3, border_mode='same', activation='tanh')(x2)

#Make layer2
autoencoder2 = Model(input_img2, decode2)
autoencoder2.compile(optimizer=opti, loss='mean_squared_error')
encoder2 = Model(input=input_img2, output=y2)
json_string = autoencoder2.to_json()
open(path2sav+'autoencoder2_temp.json', 'w').write(json_string)

# Define layer3
input_img3 = Input(shape=(encoder2.output_shape[1], encoder2.output_shape[2], encoder2.output_shape[3]))
x3 = Flatten()(input_img3)
x3 = Dense(500, activation='tanh')(x3)
y3 = Dense(100, activation='tanh')(x3)
x3 = Dense(500, activation='tanh')(y3)
x3 = Dense(encoder2.output_shape[1]*encoder2.output_shape[2]*encoder2.output_shape[3], activation='tanh')(x3)
decode3 = Reshape((encoder2.output_shape[1], encoder2.output_shape[2], encoder2.output_shape[3])) (x3)

#Make layer3
autoencoder3 = Model(input_img3, decode3)
autoencoder3.compile(optimizer=opti, loss='mean_squared_error')
encoder3 = Model(input=input_img3, output=y3)
json_string = autoencoder3.to_json()
open(path2sav+'autoencoder3_temp.json', 'w').write(json_string)

folder=sorted(glob.glob(path+name+'/*'))


# Load file names and randomly shuffle
file=[]
for i in folder:
	i
	if(file==[]):
		temp=glob.glob(i+'/'+'*'+type+'*.npy')
		random.shuffle(temp)
		file=temp[0:no_sp_per_vid]
	else:
		temp=glob.glob(i+'/*'+type+'*.npy')
		random.shuffle(temp)
		file.extend(temp[0:no_sp_per_vid])


random.shuffle(file)
#l=len(file)-1
l=len(file)
mean=[]
std=[]
print('starting '+name+' '+type)
for k in range(no_auto):
	if os.path.exists(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_'+type+'_'+str(k).zfill(2)+'.h5'):
		print('skipping '+type+' '+str(k))
		continue
	
	# Load each decoder and encoder layer as autoencoder1,2,3
	autoencoder1 = model_from_json(open(path2sav+'autoencoder1_temp.json').read())
	autoencoder1.compile(optimizer=opti, loss='binary_crossentropy')
	
	autoencoder2 = model_from_json(open(path2sav+'autoencoder2_temp.json').read())
	autoencoder2.compile(optimizer=opti, loss='mean_squared_error')
	
	autoencoder3 = model_from_json(open(path2sav+'autoencoder3_temp.json').read())
	autoencoder3.compile(optimizer=opti, loss='mean_squared_error')
	fin=[]
	
	# Load data from each file
	for i in range(k*l/no_auto,(k+1)*l/no_auto):
		data=np.load(file[i])
		for ip,io in enumerate(data):
			data[ip]=io-io.mean()
			data[ip]=data[ip]/data[ip].std()
		
		data=np.reshape(data, (fr_rate, 1, inp_sz[0],inp_sz[1]))
		if(fin==[]):
			fin=data
		else:
			fin=np.append(fin,data,0)
	'''
	if(mean==[]):
		mean=np.mean(np.mean(fin,1),1)
		std=np.std(np.std(abs(fin),1),1)
	else:
		mean=np.append(mean,np.mean(np.mean(fin,1),1),0)
		std=np.append(std,np.std(np.std(abs(fin),1),1),0)
	
	fin=np.subtract(fin,mean[k])
	fin=np.divide(fin,std[k])
	'''
	# Train each successive layer
	history1=autoencoder1.fit(fin,fin, nb_epoch=epoch*2,batch_size=batch_sz,verbose=0)
	q1 = K.function([autoencoder1.layers[0].input,K.learning_phase()],[autoencoder1.layers[4].output])
	out1=q1([fin,0])[0]
	history2=autoencoder2.fit(out1,out1,nb_epoch=epoch,batch_size=batch_sz,verbose=0)
	q1 = K.function([autoencoder2.layers[0].input,K.learning_phase()],[autoencoder2.layers[4].output])
	out2=q1([out1,0])[0]
	history3=autoencoder3.fit(out2,out2,nb_epoch=epoch*2,batch_size=batch_sz,verbose=0)
	
	#Fine tune
	input_img = Input(shape=(1, inp_sz[0], inp_sz[1]))
	x=Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same',weights=autoencoder1.layers[1].get_weights())(input_img)
	x=Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same',weights=autoencoder1.layers[2].get_weights())(x)
	x=BatchNormalization(mode=0, axis=1, weights=autoencoder1.layers[3].get_weights())(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=MaxPooling2D((2,2), border_mode='valid')(x)
	
	x=Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same', weights=autoencoder2.layers[1].get_weights())(x)
	x=Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same', weights=autoencoder2.layers[2].get_weights())(x)
	x=BatchNormalization(mode=0, axis=1, weights=autoencoder2.layers[3].get_weights())(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=MaxPooling2D((2,2), border_mode='valid')(x)
		
	x = Flatten()(x)
	x = Dense(500, activation='tanh', weights=autoencoder3.layers[2].get_weights())(x)
	y = Dense(100, activation='tanh', weights=autoencoder3.layers[3].get_weights())(x)
	x = Dense(500, activation='tanh', weights=autoencoder3.layers[4].get_weights())(y)
	x = Dense(encoder2.output_shape[1]*encoder2.output_shape[2]*encoder2.output_shape[3], weights=autoencoder3.layers[5].get_weights(), activation='tanh')(x)
	x = Reshape((encoder2.output_shape[1],encoder2.output_shape[2],encoder2.output_shape[3])) (x)
	
	x=Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same',weights=autoencoder2.layers[6].get_weights())(x)
	x=Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same',weights=autoencoder2.layers[7].get_weights())(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=BatchNormalization(mode=0, axis=1, weights=autoencoder2.layers[8].get_weights())(x)
	x=UpSampling2D((2, 2))(x)
	x=Convolution2D(no_layer[1],3,3, activation='tanh', border_mode='same',weights=autoencoder2.layers[10].get_weights())(x)
	
	x=Convolution2D(no_layer[1], 5,5, activation='tanh', border_mode='same',weights=autoencoder1.layers[6].get_weights())(x)
	x=Convolution2D(no_layer[0], 3,3, activation='tanh', border_mode='same',weights=autoencoder1.layers[7].get_weights())(x)
	#x=BatchNormalization(mode=2, axis=1)(x)
	x=BatchNormalization(mode=0, axis=1, weights=autoencoder1.layers[8].get_weights())(x)
	x=UpSampling2D((2, 2))(x)
	decode=Convolution2D(1,3,3, activation='sigmoid', border_mode='same',weights=autoencoder1.layers[10].get_weights())(x)
	
	model = Model(input_img, decode)
	model.compile(optimizer=opti, loss='binary_crossentropy')
	history = model.fit(fin, fin,nb_epoch=epoch,batch_size=batch_sz,verbose=0)
	model.save_weights(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_'+type+'_'+str(k).zfill(2)+'.h5')
	print(k)

json_string = model.to_json()
open(path2sav+'pretrained_conv_auto_'+suffix+'.json', 'w').write(json_string)

#np.save(path2sav+name+'/mean_std_'+suffix+'/mean_'+type+'.npy',mean)
#np.save(path2sav+name+'/mean_std_'+suffix+'/std_'+type+'.npy',std)
print(type)
