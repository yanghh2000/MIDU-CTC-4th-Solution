cd ..

# 1 准备数据
# 1.1 下载数据
# 下载微信公众号语料库(https://github.com/nonamestreet/weixin_public_corpus)到目录 data/weixin/
# 下载新闻语料news2016zh（https://github.com/brightmart/nlp_chinese_corpus）到目录 data/news2016zh/
# 下载预训练模型到目录 pretrained_model/
# chinese-roberta-wwm-ext(https://huggingface.co/hfl/chinese-roberta-wwm-ext)
# chinese-macbert-base(https://huggingface.co/hfl/chinese-macbert-base)
# chinese-pert-base(https://huggingface.co/hfl/chinese-pert-base)

# 1.2 数据增强和数据清洗
# 1.2.1 大文件切分为小文件
# split weixin
python -m utils.data_augment_utils.data_augment_dicts.dataAUG_util \
      --data_file 'data/weixin/articles.json' \
      --output_dir 'data/weixin/split_train' \
      --split_file_num 7

# split news
python -m utils.data_augment_utils.data_augment_dicts.dataAUG_util \
      --data_file 'data/news2016zh/news2016zh_train.json' \
      --output_dir 'data/news2016zh/split_train' \
      --split_file_num 20

# 1.2.2 数据增强和数据清洗（若报decode错误，需要将pynlpir的__init__文件241行改为result = _decode(result, errors='ignore')）
# augment weixin
python -m utils.data_augment_utils.data_augment \
      --input_dir 'data/weixin/split_train' \
      --output_dir 'data/weixin/aug_train_gec' \
      --file_num 7 \
      --worker_num 16 \
      --aug_type 'gec'

# augment news
python -m utils.data_augment_utils.data_augment \
      --input_dir 'data/news2016zh/split_train' \
      --output_dir 'data/news2016zh/aug_train_gec' \
      --file_num 20 \
      --worker_num 16 \
      --aug_type 'gec'

# 2 训练（（训练三个模型集成）以下是训练单模型roberta脚本，训练macbert和pert只需将模型目录名称改为相应目录名称macbert或pert，并更换随机种子）
# 2.1 stage 1（weixin，news）
# weixin（训练2个epoch，每个epoch7个文件，每个文件约600M）
echo "=========================== training: stage 1 weixin ==========================="
for i in $(seq 0 13)
do
  in_model_dir=model/roberta/train_stage_1_weixin/ctc_train_${i}
  out_model_dir=model/roberta/train_stage_1_weixin/ctc_train_$(($i+1))
  train_fp=data/weixin/aug_train_gec/$(($i%7+1)).txt.clean
  lr=2e-5
  if [ ! -d ${out_model_dir} ]; then
    mkdir -p ${out_model_dir}
  fi
  if [ ${i} -eq 0 ]; then
    in_model_dir=pretrained_model/chinese-roberta-wwm-ext
  fi
  if [ ${i} -gt 6 ]; then
    lr=5e-6
  fi
  echo "in_model_dir: ${in_model_dir}"
  echo "out_model_dir: ${out_model_dir}"
  echo "train_fp: ${train_fp}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.train \
  --in_model_dir ${in_model_dir} \
  --out_model_dir ${out_model_dir} \
  --epochs "1" \
  --batch_size "72" \
  --max_seq_len "128" \
  --learning_rate ${lr} \
  --train_fp ${train_fp} \
  --test_fp "data/final_data/final_val.json" \
  --random_seed_num "1234" \
  --check_val_every_n_epoch "1" \
  --early_stop_times "20" \
  --warmup_steps "-1" \
  --dev_data_ratio "0.01" \
  --training_mode "ddp" \
  --amp true \
  --freeze_embedding false
done

# news（训练2个epoch，每个epoch20个文件，每个文件约500M）
echo "=========================== training: stage 1 news ==========================="
for i in $(seq 0 39)
do
  in_model_dir=model/roberta/train_stage_1_news/ctc_train_${i}
  out_model_dir=model/roberta/train_stage_1_news/ctc_train_$(($i+1))
  train_fp=data/news2016zh/aug_train_gec/$(($i%20+1)).txt.clean
  lr=2e-5
  if [ ! -d ${out_model_dir} ]; then
    mkdir -p ${out_model_dir}
  fi
  if [ ${i} -eq 0 ]; then
    in_model_dir=model/roberta/train_stage_1_weixin/ctc_train_14
  fi
  if [ ${i} -gt 19 ]; then
    lr=5e-6
  fi
  echo "in_model_dir: ${in_model_dir}"
  echo "out_model_dir: ${out_model_dir}"
  echo "train_fp: ${train_fp}"
  echo "lr: ${lr}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.train \
  --in_model_dir ${in_model_dir} \
  --out_model_dir ${out_model_dir} \
  --epochs "1" \
  --batch_size "72" \
  --max_seq_len "128" \
  --learning_rate ${lr} \
  --train_fp ${train_fp} \
  --test_fp "data/final_data/final_val.json" \
  --random_seed_num "1234" \
  --check_val_every_n_epoch "1" \
  --early_stop_times "20" \
  --warmup_steps "-1" \
  --dev_data_ratio "0.01" \
  --training_mode "ddp" \
  --amp true \
  --freeze_embedding false
done

# 2.2 stage 2（官方伪数据 preliminary_train.json， 训练3个epoch）
echo "=========================== training: stage 2 ==========================="
for i in $(seq 0 2)
do
  in_model_dir=model/roberta/train_stage_2/ctc_train_${i}
  out_model_dir=model/roberta/train_stage_2/ctc_train_$(($i+1))
  train_fp=data/preliminary_a_data/preliminary_train.json
  lr=2e-5
  if [ ! -d ${out_model_dir} ]; then
    mkdir -p ${out_model_dir}
  fi
  if [ ${i} -eq 0 ]; then
    in_model_dir=model/roberta/train_stage_1_news/ctc_train_40
  fi
  if [ ${i} -gt 0 ]; then
    lr=5e-6
  fi
  echo "in_model_dir: ${in_model_dir}"
  echo "out_model_dir: ${out_model_dir}"
  echo "train_fp: ${train_fp}"
  echo "lr: ${lr}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.train \
  --in_model_dir ${in_model_dir} \
  --out_model_dir ${out_model_dir} \
  --epochs "1" \
  --batch_size "72" \
  --max_seq_len "128" \
  --learning_rate ${lr} \
  --train_fp ${train_fp} \
  --test_fp "data/final_data/final_val.json" \
  --random_seed_num "1234" \
  --check_val_every_n_epoch "1" \
  --early_stop_times "20" \
  --warmup_steps "-1" \
  --dev_data_ratio "0.01" \
  --training_mode "ddp" \
  --amp true \
  --freeze_embedding false
done

# 2.3 stage 3（官方真实数据 preliminary_extend_train.json，preliminary_val.json， final_train.json，训练15个epoch）
# 合并preliminary_extend_train.json，preliminary_val.json， final_train.json 到 data/final_data/train_stage_3.json
echo "=========================== merge data ==========================="
python -m utils.merge_json_file

# 训练15个epoch
echo "=========================== training: stage 3 ==========================="
for i in $(seq 0 14)
do
  in_model_dir=model/roberta/train_stage_3/ctc_train_${i}
  out_model_dir=model/roberta/train_stage_3/ctc_train_$(($i+1))
  train_fp=data/final_data/train_stage_3.json
  batch_size=36
  lr=1e-5
  if [ ! -d ${out_model_dir} ]; then
    mkdir -p ${out_model_dir}
  fi
  if [ ${i} -eq 0 ]; then
    in_model_dir=model/roberta/train_stage_2/ctc_train_3
  fi
  if [ ${i} -gt 4 ]; then
    lr=5e-6
  fi
  if [ ${i} -gt 9 ]; then
    lr=5e-7
  fi
  echo "in_model_dir: ${in_model_dir}"
  echo "out_model_dir: ${out_model_dir}"
  echo "train_fp: ${train_fp}"
  echo "lr: ${lr}"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m src.train \
  --in_model_dir ${in_model_dir} \
  --out_model_dir ${out_model_dir} \
  --epochs "1" \
  --batch_size ${batch_size} \
  --max_seq_len "128" \
  --learning_rate ${lr} \
  --train_fp ${train_fp} \
  --test_fp "data/final_data/final_val.json" \
  --random_seed_num "1234" \
  --check_val_every_n_epoch "1" \
  --early_stop_times "20" \
  --warmup_steps "-1" \
  --dev_data_ratio "0.01" \
  --training_mode "ddp" \
  --amp true \
  --freeze_embedding false
done

# 3 评估验证集
#echo "=========================== evaluating ==========================="
python -m src.evaluate \
      --json_data_file 'data/final_data/final_val.json' \
      --in_model_dir 'model/roberta/train_stage_3/ctc_train_15,model/macbert/train_stage_3/ctc_train_15,model/pert/train_stage_3/ctc_train_15' \
      --cuda_id 0 \
      --batch_size 64 \
      --keep_bias 0

# 4 预测结果文件
#echo "=========================== predict and upload ==========================="
python -m src.prepare_for_upload \
      --json_data_file 'data/final_data/final_test_source.json' \
      --output_json_file 'result/final_test_inference.json' \
      --in_model_dir 'model/roberta/train_stage_3/ctc_train_15,model/macbert/train_stage_3/ctc_train_15,model/pert/train_stage_3/ctc_train_15' \
      --cuda_id 0 \
      --batch_size 128 \
      --keep_bias 0