# Unsupervised-Learning-of-Deep-Feature-Representation-for-Clustering-Egocentric-Actions

This is the code for our paper "Unsupervised Learning of Deep Feature Representation for Clustering Egocentric Actions, IJCAI 2017".

The folder contains following files.

1) train_2D_mult_auto_ver4.py  :  Train multilayer 2D convolutional autoencoders to extract frame level features.
2) test_2D_conv.py             :  Extract spatial features from the video
3) lstm_2D_conv_ver4.py        :  Train lstm autoencoder to learn splice level representations
4) test_lstm.py                :  Extract splice level representations
5) opti.py                     :  Extract dense optical flow and save it for each splice
6) raw_frame.py                :  Extract raw frames and save them for each splice

7) ind2vec.m confusion.m       :  Helper functions
8) cluster.m                   :  Clusters the features extracted from the LSTM
9) make_gtea.m                 :  Sample code to generate GT in required format (from GTEA dataset in this case)
10) match_greedy2.m             :  Code for greedy matching of generated labels and GT
11) readNPY.m, readNPYheader.m  :  Helper functions to read .npy files in matlab
12) metadata_S2.mat             :  Contains information regarding video length etc.

13) gteagroundtruth             :  Contains sample files to form GT for GTEA dataset
14) complete.sh 				   :  Runs entire feature extraction pipeline (from optical flow extraction to LSTM features)

* In order to run these codes you will need to extract dense optical flow and raw frames from the video (use code opti.py and raw_frames.py).
* All videos were converted to 15 fps and splices were formed at 2 sec for short term actions and 30 sec for long term actions.
