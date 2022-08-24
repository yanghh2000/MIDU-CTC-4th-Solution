
from typing import List

from src.baseline.predictor import PredictorCtc


class Corrector:
    def __init__(self, in_model_dirs: List[str], cuda_id=0, batch_size=32, n_iter=1, keep_bias=0, model_weights=None):
        """_summary_

        Args:
            in_model_dir (List[str]): 训练好的模型目录，可以是多个模型集成
        """
        self._predictor = PredictorCtc(
        in_model_dirs=in_model_dirs,
        ctc_label_vocab_dir='src/baseline/ctc_vocab',
        use_cuda=True,
        cuda_id=cuda_id,
        batch_size=batch_size,
        keep_bias=keep_bias,
        model_weights=model_weights)
        self.n_iter = n_iter
        
    
    def __call__(self, texts:List[str]) -> List[str]:
        final_texts = texts[:]
        pred_ids = [i for i in range(len(texts))]
        prev_preds_dict = {i: [final_texts[i]] for i in range(len(final_texts))}

        for iter in range(self.n_iter):
            orig_texts = [final_texts[i] for i in pred_ids]

            pred_outputs = self._predictor.predict(orig_texts)
            pred_texts = [PredictorCtc.output2text(output) for output in pred_outputs]

            final_texts, pred_ids, cnt = self.update_final_batch(final_texts, pred_ids, pred_texts, prev_preds_dict)

            if not pred_ids:  # 如果本轮纠正，没有出现新的结果，那么就提前停止迭代纠正（early-stopping）
                break
                
        return final_texts

    def update_final_batch(self, final_batch, pred_ids, pred_batch, prev_preds_dict):
        """
        更新最终的纠错结果（需要有个迭代纠正的过程）
        :param final_batch: 最终的纠错结果列表
        :param pred_ids: 需要预测的句子id
        :param pred_batch: 当前迭代轮次的纠错结果列表
        :param prev_preds_dict: 之前迭代轮次的纠错结果列表
        :return:
        final_batch: 更新后的最终纠错结果列表
        new_pred_ids: 预测了多少个新句子
        total_updated: 当前轮次更新了多少句子
        """
        new_pred_ids = []
        total_updated = 0
        for i, orig_id in enumerate(pred_ids):
            orig = final_batch[orig_id]  # 上一轮纠正结果
            pred = pred_batch[i]  # 本轮纠正结果
            prev_preds = prev_preds_dict[orig_id]  # 之前出现过的纠正结果
            if orig != pred and pred not in prev_preds:  # 纠错，并且纠错结果之前从未出现过
                final_batch[orig_id] = pred
                new_pred_ids.append(orig_id)
                prev_preds_dict[orig_id].append(pred)
                total_updated += 1
            elif orig != pred and pred in prev_preds:  # 纠错，但纠错结果之前出现过
                # update final batch, but stop iterations
                final_batch[orig_id] = pred
                total_updated += 1
            else:
                continue
        return final_batch, new_pred_ids, total_updated

    