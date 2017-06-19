# Input format: code NameOfVideoSequence NumberOfFramesToBeKeptInSplice path2sav

import cv2
import os
import numpy as np
import sys
import ntpath

name2=sys.argv[1][:-4]

path2vid,name2=ntpath.split(name2)
path2vid+='/'
fmt=sys.argv[1][-4:]
fr_rate=int(sys.argv[2])
#print(fr_rate)
#
name=name2+'_'+str(fr_rate)
#print(name,name2,path2vid,fmt)

######### Set path where flow must be stored
#path2sav='/data4/bharat.b/folder4/data_pre/'
#path2sav='/Neutron9/bharat.b/data_pre/'
path2sav=sys.argv[3]

if not(os.path.isdir(path2sav)):
	os.mkdir(path2sav)
if not(os.path.isdir(path2sav+name)):
	os.mkdir(path2sav+name)
inp_sz=(64,36)
cap = cv2.VideoCapture(path2vid+name2+fmt)
#print(path2vid+name2+fmt)

ret, frame1 = cap.read()
#print(np.shape(frame1))
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#
#prvs=cv2.resize(prvs,(860,540))
fin_x = np.zeros((fr_rate,inp_sz[1],inp_sz[0]))
fin_y = np.zeros((fr_rate,inp_sz[1],inp_sz[0]))
j=0
k=0
while(1):
	if(j%fr_rate==0 and j!=0):
		np.save(path2sav+name+'/flow_x_'+str(k).zfill(3),fin_x)
		np.save(path2sav+name+'/flow_y_'+str(k).zfill(3),fin_y)
		k+=1
		j=0
		#print(name,str(k))
	ret, frame2 = cap.read()
	if(np.shape(frame2)==()):
		break
	next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
	#
	#next=cv2.resize(next,(860,540))
	flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)
	#
	flow = cv2.resize(flow,(inp_sz[0],inp_sz[1]))
	fin_x[j,:,:]=flow[...,0]
	fin_y[j,:,:]=flow[...,1]
	prvs = next
	j+=1

print(name+'_done')
cap.release()
cv2.destroyAllWindows()
