python3 -u -m motmetrics.apps.evaluateTracking ./ ./ ./mot_dir/path.txt 2>&1 | tee log/mot_test_log.txt
python3 tao-master/scripts/evaluation/evaluate.py datasets/fsvod/annotations/fsvod_val.json output/tpn_results.json --output-dir output/ 2>&1 | tee log/tao_mot_test_log.txt
