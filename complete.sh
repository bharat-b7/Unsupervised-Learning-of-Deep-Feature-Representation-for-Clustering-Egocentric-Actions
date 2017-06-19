gpu=0

# Extract optical flow
search_dir="./S2"
for entry in "$search_dir"/*
do
  echo "$entry"
  /usr/bin/python opti.py $entry 30 "./data_pre/S2"
done

# Extract raw_frames

for entry in "$search_dir"/*
do
  echo "$entry"
  /usr/bin/python raw_frame.py $entry 30 "./data_pre/S2"
done

# Train spatial autoencoders for flow_x, flow_y and frame respectivly
python train_2D_mult_auto_ver4.py "S2" $gpu "./data_pre/" "./folder5/" "x"
python train_2D_mult_auto_ver4.py "S2" $gpu "./data_pre/" "./folder5/" "y"
python train_2D_mult_auto_ver4.py "S2_frame" $gpu "./data_pre/" "./folder5/"

# Extract features from flow_x stream
ser_dir="./data_pre/S2"
name="S2"
type="x"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python test_2D_conv.py $name $f $gpu "./data_pre/" "./folder5/" $type
done

# Extract features from flow_y stream
name="S2"
type="y"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python test_2D_conv.py $name $f $gpu "./data_pre/" "./folder5/" $type
done

# Extract features from frame stream
ser_dir="./data_pre/S2_frame"
name="S2_frame"
type=""
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python test_2D_conv.py $name $f $gpu "./data_pre/" "./folder5/" $type
done

# Train LSTM autoencoder for flow_x, flow_y and frame respectivly
python lstm_2D_conv_ver4.py "S2" $gpu './data_pre/' './folder5/' 'x'
python lstm_2D_conv_ver4.py "S2" $gpu './data_pre/' './folder5/' 'y'
python lstm_2D_conv_ver4.py "S2_frame" $gpu './data_pre/' './folder5/'

# Extract LSTM features for flow_x
ser_dir="./data_pre/S2"
name="S2"
type="x"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python test_lstm.py $name $f $gpu './folder5/' $type
done

# Extract LSTM features for flow_y
ser_dir="./data_pre/S2"
name="S2"
type="y"
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python test_lstm.py $name $f $gpu './folder5/' $type
done

# Extract LSTM features for frame
ser_dir="./data_pre/S2_frame"
name="S2_frame"
type=""
for e in "$ser_dir"/*
do
	f=`basename $e`
	echo $name $f
	python test_lstm.py $name $f $gpu './folder5/' $type
done
