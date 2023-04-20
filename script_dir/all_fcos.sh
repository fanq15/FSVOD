CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 \
	--config-file configs/fcos/R_50_1x.yaml 2>&1 | tee log/fcos_train_log.txt

