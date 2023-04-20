DATA_ROOT=/home/fanqi/data/COCO
DATA_PATH=./
cd datasets/coco

ln -s $DATA_ROOT/train2017 .
ln -s $DATA_ROOT/val2017 .
ln -s $DATA_ROOT/annotations .

python3 1_split_filter.py $DATA_PATH
python3 2_balance.py $DATA_PATH
python3 3_gen_support_pool.py $DATA_PATH
python3 4_gen_support_pool_10_shot.py $DATA_PATH

