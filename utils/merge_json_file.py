import json

json_data1 = json.load(open('data/preliminary_a_data/preliminary_extend_train.json', 'r', encoding='utf-8'))
json_data2 = json.load(open('data/preliminary_a_data/preliminary_val.json', 'r', encoding='utf-8'))
json_data3 = json.load(open('data/final_data/final_train.json', 'r', encoding='utf-8'))

all_data  = json_data1 + json_data2 + json_data3

json.dump(all_data, open('data/final_data/train_stage_3.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=3)
