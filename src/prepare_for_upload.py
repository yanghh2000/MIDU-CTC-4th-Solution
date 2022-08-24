import json

from src.corrector import Corrector
from gec_error_template.run import correct_sents

from opencc import OpenCC

cc = OpenCC("t2s")
def prepare_for_uploadfile(in_model_dir,
                           in_json_file, 
                           out_json_file='result/final_test_inference.json',
                           cuda_id=0,
                           batch_size=32,
                           keep_bias=0,
                           model_weights=None):
        
    json_data_list = json.load(open(in_json_file, 'r', encoding='utf-8'))

    src_texts = [ json_data['source'] for json_data in json_data_list]

    corrector = Corrector(in_model_dirs=in_model_dir.split(','),
                          cuda_id=cuda_id,
                          batch_size=batch_size,
                          n_iter=1,
                          keep_bias=keep_bias,
                          model_weights=model_weights)
    pred_texts = corrector(texts=src_texts)

    pred_texts = [cc.convert(pred) for pred in pred_texts]

    pred_texts, sentence_num, error_num = correct_sents(pred_texts)
    print("Fix {:d} errors from {:d} sentences.".format(sentence_num, error_num))

    output_json_data = [ {'id':json_data['id'], 'inference': pred_text} for json_data, pred_text in zip(json_data_list, pred_texts)]
    
    out_json_file = open(out_json_file, 'w', encoding='utf-8')
    json.dump(output_json_data, out_json_file, ensure_ascii=False, indent=4)


# def prepare_for_uploadfile_macbert(in_model_dir,
#                            in_json_file,
#                            out_json_file='result/final_test_inference.json',
#                            cuda_id=0,
#                            batch_size=32,
#                            keep_bias=0,
#                            model_weights=None):
#     json_data_list = json.load(open(in_json_file, 'r', encoding='utf-8'))
#
#     src_texts = [json_data['source'] for json_data in json_data_list]
#     macbert_preds = [json_data['predict'] for json_data in json_data_list]
#
#     corrector = Corrector(in_model_dirs=in_model_dir.split(','),
#                           cuda_id=cuda_id,
#                           batch_size=batch_size,
#                           n_iter=1,
#                           keep_bias=keep_bias,
#                           model_weights=model_weights)
#     pred_texts = corrector(texts=macbert_preds)
#
#     pred_texts = [cc.convert(pred) for pred in pred_texts]
#
#     pred_texts, sentence_num, error_num = correct_sents(pred_texts)
#     print("Fix {:d} errors from {:d} sentences.".format(sentence_num, error_num))
#
#     output_json_data = [{'id': json_data['id'], 'inference': pred_text} for json_data, pred_text in
#                         zip(json_data_list, pred_texts)]
#
#     out_json_file = open(out_json_file, 'w', encoding='utf-8')
#     json.dump(output_json_data, out_json_file, ensure_ascii=False, indent=4)

def main(args):
    prepare_for_uploadfile(in_model_dir=args.in_model_dir,
                           in_json_file=args.json_data_file,
                           out_json_file=args.output_json_file,
                           cuda_id=args.cuda_id,
                           batch_size=args.batch_size,
                           model_weights=args.model_weights,
                           keep_bias=args.keep_bias)

if __name__ == '__main__':
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--json_data_file',
                        default='data/final_data/final_test_source.json')
    parser.add_argument('--output_json_file',
                        default='result/final_test_inference.json')
    parser.add_argument('--in_model_dir',               # 多个模型集成预测时，用逗号隔开
                        default='model/roberta_best,model/macbert_best,model/pert_best')
    parser.add_argument('--cuda_id',
                        type=int,
                        default=7)
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--model_weights',
                        type=list,
                        default=[0.3, 0.5, 0.4])
    parser.add_argument('--keep_bias',
                        type=float,
                        default=0)

    args = parser.parse_args()

    main(args)