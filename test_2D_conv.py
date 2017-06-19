## Input format: code NameofTrainingSeq NameOfTestSequence GpuNo. Path2Data Path2sav Type

from __future__ import print_function
import os as os  
import sys
os.environ['THEANO_FLAGS'] = "device=gpu" + sys.argv[3]
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, AveragePooling2D, BatchNormalization, Dropout, GRU
from keras.models import Model
import numpy as np
from keras.models import model_from_json
from keras import backend as K
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import glob as glob
import numpy as np


######### Set path where data (dense flow and raw frames) are stored
path=sys.argv[4]
#path='/Neutron9/bharat.b/data_pre/'


######### Set path where models must be stored
path2sav=sys.argv[5]
#path2sav='/Neutron9/bharat.b/folder5/'


### Set layer from which features are to be extracted
ext_later=11#11

name=sys.argv[1]
name_test=sys.argv[2]

suffix='ver4_2'
fr_rate=30
drop=0
if not(os.path.isdir(path2sav+name+'/'+name_test+'_feats_'+suffix)):
	os.mkdir(path2sav+name+'/'+name_test+'_feats_'+suffix)

inp_sz=(36,64)

####Set type here, x for flow_x, y for flow_y and <blank> for frame
if(len(sys.argv)==7):
	type=sys.argv[6]
else:
	type=''


#### Set tst seq

file=sorted(glob.glob(path+name+'/'+name_test+'/'+'*'+type+'*.npy'))

d=glob.glob(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_'+type+'_*')
no_auto=len(d)
#l=len(file)-1
l=len(file)

# Load pretrained model
model = model_from_json(open(path2sav+'pretrained_conv_auto_'+suffix+'.json').read())
feat_sz=np.prod(model.layers[ext_later].output_shape[1:])
q1 = K.function([model.layers[0].input,K.learning_phase()],[model.layers[ext_later].output])

#mean=np.load(path2sav+name+'/mean_std_'+suffix+'/mean_'+type+'.npy')
#std=np.load(path2sav+name+'/mean_std_'+suffix+'/std_'+type+'.npy')
feat=np.zeros((l,fr_rate,feat_sz*no_auto))

for k in range(no_auto):
	model.load_weights(path2sav+name+'/models_'+suffix+'/pretrained_conv_auto_'+type+'_'+str(k).zfill(2)+'.h5')
	for j in range(l):
		data=np.load(file[j])
		for ip,io in enumerate(data):
			data[ip]=io-io.mean()
			data[ip]=data[ip]/data[ip].std()
		
		data=np.reshape(data, (fr_rate, 1, inp_sz[0],inp_sz[1]))
		#data=np.subtract(data,mean[k])
		#data=np.divide(data,std[k])
		feat[j,:,k*feat_sz:(k+1)*feat_sz]=np.array(q1([data,0]))
	
	print(k)

np.save(path2sav+name+'/'+name_test+'_feats_'+suffix+'/feat_'+type,feat)


print('feat_sz_'+type,feat_sz)
