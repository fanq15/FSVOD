rm -r mot_dir
CUDA_VISIBLE_DEVICES=0 python tools/test_tpn.py --config-file ./configs/tpn/R_50_1x.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth
sh eval.sh

sh test_fsvod.sh
#python evaluate_coco.py 2>&1 | tee log/final_tao_fsvod_test_log.txt

#rm -r mot_dir
#rm -r gt_mot_dir 
