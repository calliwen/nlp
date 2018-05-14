#/bin/bash
echo $1
echo $2
python preprocess.py $1
python nn_model.py $2