cd ..

# 加载已有checkpoint
python -m src.prepare_for_upload \
      --json_data_file 'data/final_data/final_test_source.json' \
      --output_json_file 'result/final_test_inference.json' \
      --in_model_dir 'model/roberta_best,model/macbert_best,model/pert_best' \
      --cuda_id 0 \
      --batch_size 128 \
      --keep_bias 0