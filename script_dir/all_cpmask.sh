#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 \
#	--config-file configs/cpmask/R_50_1x.yaml #2>&1 | tee log/cpmask_train_log.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_net.py --num-gpus 4 \
	--config-file configs/cpmask/R_50_1x.yaml \
	--eval-only MODEL.WEIGHTS ./output/cpmask/R_50_1x/model_final.pth 2>&1 | tee log/cpmask_test_log.txt
