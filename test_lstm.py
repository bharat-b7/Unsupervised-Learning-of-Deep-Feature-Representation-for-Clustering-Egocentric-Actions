#Input: code NameofTrainingSeq NameOfTestSeq Gpu Path2sav Type

from __future__ import print_function
import os as os  
import sys
os.environ['THEANO_FLAGS'] = "device=gpu" + sys.argv[3]
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, BatchNormalization, Dropout, GRU
from keras.models import Model
import numpy as np
from keras.models import model_from_json
from keras import backend as K
from keras.models import load_model
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import glob as glob
import numpy as np

path2sav=sys.argv[4]

name=sys.argv[1]
name_test=sys.argv[2]
#####Set type
if(len(sys.argv)==6):
	type=sys.argv[5]
else:
	type=''

	
fr_rate=30
suffix='lstm_ver4_2'
suffix1='ver4_2'

if not(os.path.isdir(path2sav+name+'/'+name_test+'_'+suffix)):
	os.mkdir(path2sav+name+'/'+name_test+'_'+suffix)


feat_sz=100
drop=0
epoch=100
opti='RMSprop'

# Compute features at fps (regular temporal resolution)
feat=np.load(path2sav+name+'/'+name_test+'_feats_'+suffix1+'/feat_'+type+'.npy')
no_auto=np.shape(feat)[2]/feat_sz

model=load_model(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_0_'+type+'.h5')
model.compile(optimizer='RMSprop', loss='mean_squared_error')

q1 = K.function([model.layers[0].input],[model.layers[1].output])
rec=q1([feat])
rec_fin=np.reshape(rec,(np.shape(rec)[1],np.prod(model.layers[1].output_shape[1:])))

model=load_model(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_1_'+type+'.h5')
model.compile(optimizer='RMSprop', loss='mean_squared_error')


# Compute features at fps/2
feat2=np.copy(feat)
if(len(feat2)%2==1):
	feat2=feat2[:-1]


feat2=np.reshape(feat2,(np.shape(feat2)[0]/2,np.shape(feat2)[1]*2,np.shape(feat2)[2]))
feat2=feat2[:,::2,:]
q1 = K.function([model.layers[0].input],[model.layers[1].output])
rec=q1([feat2])
rec=np.reshape(rec,(np.shape(rec)[1],np.prod(model.layers[1].output_shape[1:])))
rect=np.zeros((np.shape(rec)[0]*2,np.shape(rec)[1]))
rect[::2,:]=rec
rect[1::2,:]=rec
rec_fin=rec_fin[:len(rect),:]
rec_fin=np.append(rec_fin,rect,axis=1)

# Compute features at fps/4
model=load_model(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_2_'+type+'.h5')
model.compile(optimizer='RMSprop', loss='mean_squared_error')
feat3=np.copy(feat)
if(len(feat3)%2==1):
	feat3=feat3[:-1]

feat3=feat3[:,::2,:]
feat4=np.zeros(np.shape(feat))
if(len(feat4)%2==1):
	feat4=feat4[:-1]

feat4[:,::2,:]=feat3
feat4[:,1::2,:]=feat3

q1 = K.function([model.layers[0].input],[model.layers[1].output])
rec=q1([feat4])
rec=np.reshape(rec,(np.shape(rec)[1],np.prod(model.layers[1].output_shape[1:])))
rec_fin=rec_fin[:len(rec),:]
rec_fin=np.append(rec_fin,rec,axis=1)

print('completed '+name_test+' '+type+ ' done')
np.save(path2sav+name+'/'+name_test+'_'+suffix+'/feat_'+type,rec_fin)