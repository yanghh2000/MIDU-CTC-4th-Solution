import json
from src.corrector import Corrector
from src.metric import final_f1_score
import tqdm
from gec_error_template.run import correct_sents


from opencc import OpenCC

cc = OpenCC("t2s")

def evaluate(in_model_dirs,
             json_data_file,
             log_fp='logs/f1_score.log',
             cuda_id=None,
             batch_size=32,
             n_iter=1,
             keep_bias=0,
             model_weights=None):
    """输入模型目录，数据， 计算模型在该数据下的指标

    """
    
    json_data = json.load(open(json_data_file, 'r', encoding='utf-8'))
    src_texts, trg_texts = [], []
    is_positive, id = [], []
    for line in json_data:
        src_texts.append(line['source'])
        trg_texts.append(line['target'])
        is_positive.append(line['type'])
        id.append(line['id'])
    
    corrector = Corrector(in_model_dirs=in_model_dirs.split(','),
                          cuda_id=cuda_id,
                          batch_size=batch_size,
                          n_iter=n_iter,
                          keep_bias=keep_bias,
                          model_weights=model_weights)
    pred_texts = corrector(texts=src_texts)

    pred_texts = [cc.convert(pred) for pred in pred_texts]

    f1_score1 = final_f1_score(src_texts=src_texts,
                              pred_texts=pred_texts,
                              trg_texts=trg_texts,
                              log_fp=log_fp)

    # 语义纠错模板
    output_texts, sentence_num, error_num = correct_sents(pred_texts, trg_texts)

    f1_score2 = final_f1_score(src_texts=src_texts,
                              pred_texts=output_texts,
                              trg_texts=trg_texts,
                              log_fp=log_fp)

    print("Fix {:d} errors from {:d} sentences.".format(sentence_num, error_num))
    print('before: ', f1_score1)
    print('after: ', f1_score2)

    ####
    #  输出模型预测错误的句子

    err_texts = []
    correct_texts = []
    count = 0
    err_positve_count = 0
    err_negative_count = 0

    for i in tqdm.tqdm(range(len(pred_texts))):
        assert len(trg_texts) == len(pred_texts)
        if trg_texts[i] != pred_texts[i]:   # 预测不正确的句子
            err_texts.append(str(id[i]) + '\n' + 'src: ' + src_texts[i] + '\n' +
                             'tgt: ' + trg_texts[i] + '\n' + 'pre: ' + pred_texts[i] + '\n' + is_positive[i] + '\n' + '\n')
            count += 1
            if is_positive[i] == 'positive':
                err_positve_count += 1
            else:
                err_negative_count += 1
        else:           # 输出预测正确的句子
            correct_texts.append(str(id[i]) + '\n' + 'src: ' + src_texts[i] + '\n' +
                             'tgt: ' + trg_texts[i] + '\n' + 'pre: ' + pred_texts[i] + '\n' + is_positive[i] + '\n' + '\n')

    model_name = ','.join([model.split('/')[-1] for model in in_model_dirs.split(',')])

    with open(f'logs/pred_error_texts_{model_name}', 'w', encoding='utf-8') as f2:
        f2.write(''.join(err_texts))
        f2.write(f"correct count: {len(trg_texts)-count} of {len(trg_texts)}" + "\n")
        f2.write(f"error count: {count} of {len(trg_texts)}, positive: {err_positve_count}, negative: {err_negative_count}")

    with open(f'logs/pred_correct_texts_{model_name}', 'w', encoding='utf-8') as f2:
        f2.write(''.join(correct_texts))

    print(f"correct count: {len(trg_texts)-count} of {len(trg_texts)}")
    print(f"error count: {count} of {len(trg_texts)}, positive: {err_positve_count}, negative: {err_negative_count}")

    return f1_score2, f"error count: {count} of {len(trg_texts)}, positive: {err_positve_count}, negative: {err_negative_count}"

### 用于macbert拼写纠错后再进行语法纠错（unused）
def evaluate_macbert(in_model_dirs,
             json_data_file,
             log_fp='logs/f1_score.log',
             cuda_id=None,
             batch_size=32,
             n_iter=1,
             keep_bias=0,
             model_weights=None):
    """输入模型目录，数据， 计算模型在该数据下的指标

    """

    json_data = json.load(open(json_data_file, 'r', encoding='utf-8'))
    src_texts, trg_texts = [], []
    is_positive, id = [], []
    macbert_preds = []
    for line in json_data:
        src_texts.append(line['source'])
        trg_texts.append(line['target'])
        is_positive.append(line['type'])
        id.append(line['id'])
        macbert_preds.append(line['predict'])

    corrector = Corrector(in_model_dirs=in_model_dirs.split(','),
                          cuda_id=cuda_id,
                          batch_size=batch_size,
                          n_iter=n_iter,
                          keep_bias=keep_bias,
                          model_weights=model_weights)

    pred_texts = corrector(texts=macbert_preds)     # 对macbert进行拼写纠错后的句子进行语法纠错

    pred_texts = [cc.convert(pred) for pred in pred_texts]

    f1_score1 = final_f1_score(src_texts=src_texts,
                               pred_texts=pred_texts,
                               trg_texts=trg_texts,
                               log_fp=log_fp)

    output_texts, sentence_num, error_num = correct_sents(pred_texts, trg_texts)

    f1_score2 = final_f1_score(src_texts=src_texts,
                               pred_texts=output_texts,
                               trg_texts=trg_texts,
                               log_fp=log_fp)

    print("Fix {:d} errors from {:d} sentences.".format(sentence_num, error_num))
    print('before: ', f1_score1)
    print('after: ', f1_score2)

    ####
    #  输出模型预测错误的句子

    err_texts = []
    correct_texts = []
    count = 0
    err_positve_count = 0
    err_negative_count = 0

    for i in tqdm.tqdm(range(len(pred_texts))):
        assert len(trg_texts) == len(pred_texts)
        if trg_texts[i] != pred_texts[i]:  # 预测不正确的句子
            err_texts.append(str(id[i]) + '\n' + 'src: ' + src_texts[i] + '\n' +
                             'tgt: ' + trg_texts[i] + '\n' + 'pre: ' + pred_texts[i] + '\n' + is_positive[
                                 i] + '\n' + '\n')
            count += 1
            if is_positive[i] == 'positive':
                err_positve_count += 1
            else:
                err_negative_count += 1
        else:  # 输出预测正确的句子
            correct_texts.append(str(id[i]) + '\n' + 'src: ' + src_texts[i] + '\n' +
                                 'tgt: ' + trg_texts[i] + '\n' + 'pre: ' + pred_texts[i] + '\n' + is_positive[
                                     i] + '\n' + '\n')

    model_name = ','.join([model.split('/')[-1] for model in in_model_dirs.split(',')])

    with open(f'logs/pred_error_texts_{model_name}', 'w', encoding='utf-8') as f2:
        f2.write(''.join(err_texts))
        f2.write(f"correct count: {len(trg_texts) - count} of {len(trg_texts)}" + "\n")
        f2.write(
            f"error count: {count} of {len(trg_texts)}, positive: {err_positve_count}, negative: {err_negative_count}")

    with open(f'logs/pred_correct_texts_{model_name}', 'w', encoding='utf-8') as f2:
        f2.write(''.join(correct_texts))

    print(f"correct count: {len(trg_texts) - count} of {len(trg_texts)}")
    print(f"error count: {count} of {len(trg_texts)}, positive: {err_positve_count}, negative: {err_negative_count}")

    return f1_score2, f"error count: {count} of {len(trg_texts)}, positive: {err_positve_count}, negative: {err_negative_count}"

def main(args):
    f1 = evaluate(in_model_dirs=args.in_model_dir,
                  json_data_file=args.json_data_file,
                  cuda_id=args.cuda_id,
                  batch_size=args.batch_size,
                  n_iter=1,
                  keep_bias=args.keep_bias,
                  model_weights=args.model_weights)

    print(f1)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_data_file',
                        default='data/final_data/final_val.json')
    parser.add_argument('--in_model_dir',               # 多个模型集成预测时，用逗号隔开
                        default='model/roberta_best,model/macbert_best,model/pert_best')
    parser.add_argument('--model_weights',          # [pert] 0.6683 [macbert] 0.6449 [roberta] 0.6853   [ensemble] 0.7055
                        type=list,
                        default=[0.5, 0.3, 0.4])        # 手动调参最好的模型权重
    parser.add_argument('--cuda_id',
                        type=int,
                        default=0)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--keep_bias',
                        type=float,
                        default=0)

    args = parser.parse_args()

    main(args)


