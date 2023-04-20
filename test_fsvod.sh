rm -r support_dir
CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_555.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth

CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_555.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth 2>&1 | tee log/555_fsvod_log.txt

python evaluate_coco.py 2>&1 | tee log/555_test_log.txt
<<commnet
rm -r support_dir
CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_666.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth

CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_666.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth 2>&1 | tee log/666_fsvod_log.txt

python evaluate_coco.py 2>&1 | tee log/666_test_log.txt


rm -r support_dir
CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_777.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth

CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_777.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth 2>&1 | tee log/777_fsvod_log.txt

python evaluate_coco.py 2>&1 | tee log/777_test_log.txt

rm -r support_dir
CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_888.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth

CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_888.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth 2>&1 | tee log/888_fsvod_log.txt

python evaluate_coco.py 2>&1 | tee log/888_test_log.txt

rm -r support_dir
CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_999.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth

CUDA_VISIBLE_DEVICES=0 python tools/test_fsvod.py --config-file ./configs/tpn/R_50_1x_999.yaml \
    --opt MODEL.WEIGHTS ./output/tpn/R_50_1x/model_final.pth 2>&1 | tee log/999_fsvod_log.txt

python evaluate_coco.py 2>&1 | tee log/999_test_log.txt
comment
