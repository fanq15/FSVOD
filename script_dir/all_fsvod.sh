rm support_dir/offline_fsvod_val_support_feature.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
	--config-file configs/fsvod/R_50_C4_1x.yaml 2>&1 | tee log/fsvod_train_log.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
	--config-file configs/fsvod/R_50_C4_1x.yaml \
	--eval-only MODEL.WEIGHTS ./output/fsvod/R_50_C4_1x/model_final.pth 2>&1 | tee log/new_fsvod_val_log_20_way.txt

#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 fsod_train_net.py --num-gpus 4 \
#	--config-file configs/fsvod/R_50_C4_1x.yaml \
#	--eval-only MODEL.WEIGHTS ./output/fsvod/R_50_C4_1x/model_final.pth 2>&1 | tee log/new_fsvod_val_log_20_way.txt
