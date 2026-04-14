python main_score_cls.py \
  --mode test \
  --score_type BE \
  --model ResNet34 \
  --ordinal_method coral \
  --data_path /mnt/data2/datasx/FullHand/NIPS26/RAM-H1200/SvdH_Scoring/SvdH_BE_Scoring \
  --checkpoint /mnt/data1/songxiao/RAM-H1200/ckpts/Benchmark_BEScoring_be_resnet34_coral_20260410162915 \
  --val_batch_size 32 \
  --num_workers 4 \
  --save_csv