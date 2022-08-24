cd ..

# 加载已有checkpoint
python -m src.evaluate \
      --json_data_file 'data/final_data/final_val.json' \
      --in_model_dir 'model/roberta_best,model/macbert_best,model/pert_best' \
      --cuda_id 0 \
      --batch_size 64 \
      --keep_bias 0