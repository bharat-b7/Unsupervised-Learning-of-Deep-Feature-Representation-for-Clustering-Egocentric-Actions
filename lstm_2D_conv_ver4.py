## Input format: code NameofTrainingSeq GpuNo. Path2Data Path2sav Type

from __future__ import print_function
import os as os  
import sys
os.environ['THEANO_FLAGS'] = "device=gpu" + sys.argv[2]
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, BatchNormalization, Dropout, GRU
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import glob as glob
import numpy as np

######### Set path where data (dense flow and raw frames) are stored
path=sys.argv[3]
#path='/Neutron9/bharat.b/data_pre/'


######### Set path where models must be stored
path2sav=sys.argv[4]
#path2sav='/Neutron9/bharat.b/folder5/'

name=sys.argv[1]
####Set type here, x for flow_x, y for flow_y and <blank> for frame
if(len(sys.argv)==6):
	type=sys.argv[5]
else:
	type=''

fr_rate=30
suffix='lstm_ver4_2'
suffix1='ver4_2'
no_sp_per_vid=100

'''
if os.path.exists(path2sav+name+'/feats_'+suffix+'/'+name+'_feat_'+type+'.npy'):
	print('already there')
	sys.exit(0)
'''
#from sklearn.decomposition import KernelPCA




if not(os.path.isdir(path2sav+name)):
	os.mkdir(path2sav+name)

if not(os.path.isdir(path2sav+name+'/models_'+suffix)):
	os.mkdir(path2sav+name+'/models_'+suffix)

if not(os.path.isdir(path2sav+name+'/mean_std_'+suffix)):
	os.mkdir(path2sav+name+'/mean_std_'+suffix)

'''
if not(os.path.isdir(path2sav+name+'/feats_'+suffix)):
	os.mkdir(path2sav+name+'/feats_'+suffix)
'''

feat_sz=100
drop=0
epoch=100
opti='RMSprop'


folder=sorted(glob.glob(path2sav+name+'/*feats*'+suffix1))

fin_feat1=[]
for i in folder:
	i
	if(fin_feat1==[]):
		temp=glob.glob(i+'/'+'*'+type+'*.npy')
		temp=np.load(temp[0])
		np.random.shuffle(temp)
		fin_feat1=temp[0:no_sp_per_vid,...]
	else:
		temp=glob.glob(i+'/*'+type+'*.npy')
		temp=np.load(temp[0])
		np.random.shuffle(temp)
		fin_feat1=np.append(fin_feat1,temp[0:no_sp_per_vid,...],0)



no_auto=np.shape(fin_feat1)[2]/feat_sz

input_img = Input(shape=(fr_rate, feat_sz*no_auto))
l=GRU(100, return_sequences=True, activation='tanh',inner_activation='hard_sigmoid') (input_img)
l=GRU(feat_sz*no_auto, return_sequences=True, activation='tanh',inner_activation='hard_sigmoid') (l)
model = Model(input_img, l)
model.compile(optimizer='RMSprop', loss='mean_squared_error')

# Train model at fps (regular temporal resolution)
history=model.fit(fin_feat1, fin_feat1[:,::-1,:],nb_epoch=200,batch_size=250,verbose=1)
model.save(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_0_'+type+'.h5')

# Train model at fps/2
feat2=np.copy(fin_feat1)
if(len(feat2)%2==1):
	feat2=feat2[:-1]

input_img = Input(shape=(fr_rate, feat_sz*no_auto))
l=GRU(100, return_sequences=True, activation='tanh',inner_activation='hard_sigmoid') (input_img)
l=GRU(feat_sz*no_auto, return_sequences=True, activation='tanh',inner_activation='hard_sigmoid') (l)
model = Model(input_img, l)
model.compile(optimizer='RMSprop', loss='mean_squared_error')

feat2=np.reshape(feat2,(np.shape(feat2)[0]/2,np.shape(feat2)[1]*2,np.shape(feat2)[2]))
feat2=feat2[:,::2,:]
history=model.fit(feat2, feat2[:,::-1,:],nb_epoch=200,batch_size=250,verbose=1)
model.save(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_1_'+type+'.h5')

# Train model at fps/4
feat3=np.copy(fin_feat1)
if(len(feat3)%2==1):
	feat3=feat3[:-1]

feat3=feat3[:,::2,:]
feat4=np.zeros(np.shape(fin_feat1))
if(len(feat4)%2==1):
	feat4=feat4[:-1]

feat4[:,::2,:]=feat3
feat4[:,1::2,:]=feat3

input_img = Input(shape=(fr_rate, feat_sz*no_auto))
l=GRU(100, return_sequences=True, activation='tanh',inner_activation='hard_sigmoid') (input_img)
l=GRU(feat_sz*no_auto, return_sequences=True, activation='tanh',inner_activation='hard_sigmoid') (l)
model = Model(input_img, l)
model.compile(optimizer='RMSprop', loss='mean_squared_error')

history=model.fit(feat4, feat4[:,::-1,:],nb_epoch=200,batch_size=250,verbose=1)
model.save(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_2_'+type+'.h5')
