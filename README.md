# 文本智能校对大赛——YanSun团队方案（第四名）

[toc]

## 方案说明
文本智能校对大赛 [比赛官方链接](https://aistudio.baidu.com/aistudio/competition/detail/404/0/introduction)
### 解决方案
对于本次比赛的解决方案，我们主要参考了CTC 2021冠军S&A队的解决方案，[S&A 队 CTC2021 比赛技术评测报告](https://github.com/HillZhang1999/CTC-Report/blob/main/Report.pdf)。主要是以下方面：
1. 参考该评测报告，选取了[微信公众号语料](#数据来源)和[中文新闻语料](#数据来源)作为预训练数据。
2. 数据增强策略参考该评测报告，代码自己实现。
3. 训练数据的编辑标签抽取参考了 https://github.com/HillZhang1999/MuCGEC/blob/main/models/seq2edit-based-CGEC/utils/preprocess_data.py 中 align_sequences函数的实现。
4. 参考了其[语义模板纠错](https://github.com/HillZhang1999/gec_error_template)。
### 模型代码
模型代码结构，沿用了比赛官方提供的 [baseline](https://github.com/bitallin/MiduCTC-competition)，主要增加/修改了以下功能：
1. 由于官方baseline中编辑标签抽取部分在遇到连续append或replace后append或append后replace的情况时，存在标签无法抽取的问题。于是参考[MUCGEC](https://github.com/HillZhang1999/MuCGEC/blob/main/models/seq2edit-based-CGEC/utils/preprocess_data.py)，修改了dataset.py中训练数据的编辑标签抽取策略部分的代码，如下：
```
 def parse_item(self, src, trg):
    ...
    # replace_idx_list, delete_idx_list, missing_idx_list = self.match_ctc_idx(src, trg)    # 替换该部分

    labels = align_sequences(src[1:], trg[1:])    # 提取编辑标签
    ...
```
2. 增加了模型集成和多轮迭代纠错功能，使用了三个模型集成预测。 
### 训练流程
1. 用[中文reberta预训练模型](https://huggingface.co/hfl/chinese-roberta-wwm-ext)初始化模型编码器，首先在增强的微信和新闻伪数据上进行预训练，训练2个epoch（在7块2080Ti上一个epoch大概需要13个小时），选取得分最高的checkpoint进入下一阶段训练
2. 然后在官方提供的伪数据上（对官方提供的伪数据中的原句进行增强可得到两倍伪数据）训练2-3个epoch，选择得分最高的checkpoint进入下一阶段训练
3. 最后在真实数据上进行多轮微调，训练10-20个epoch（小学习率），选取得分最高的checkpoint作为best model，进行预测
4. 重新选取[chinese macbert 预训练模型](https://huggingface.co/hfl/chinese-macbert-base)和[chinese pert预训练模型](https://huggingface.co/hfl/chinese-pert-base)初始化模型编码器，更换随机种子，再训练2个新模型，用于集成预测。

## 数据来源&准备
1. 微信公众号语料(3G) [https://github.com/nonamestreet/weixin_public_corpus](https://github.com/nonamestreet/weixin_public_corpus)，下载并解压到 data/weixin/ 目录下
2. 中文新闻语料(9G) [https://github.com/brightmart/nlp_chinese_corpus](https://github.com/brightmart/nlp_chinese_corpus#2%E6%96%B0%E9%97%BB%E8%AF%AD%E6%96%99json%E7%89%88news2016zh)，下载并解压到 data/news2016zh/ 目录下
3. 官方提供的训练数据，存放到 data/preliminary_a_data/ 和 data/final_data/ 目录下。

## 预训练模型准备&下载
本方案使用了chinese-roberta-wwm-ext，chinese-macbert-base，chinese-pert-base 三个预训练模型，请查看 pretrained_model/ 目录下是否已经有这些预训练模型权重文件，若没有，需要从huggingface下载对应模型权重文件：
chinese-roberta-wwm-ext: https://huggingface.co/hfl/chinese-roberta-wwm-ext
chinese-macbert-base: https://huggingface.co/hfl/chinese-macbert-base
chinese-pert-base: https://huggingface.co/hfl/chinese-pert-base

## 环境安装
```
conda create -n  miduCTC python==3.7
conda activate miduCTC
pip install -r requirements.txt
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```
数据增强需要使用pynlpir分词库，需要先更新pynlpir库授权文件
```
pynlpir update      # 更新pynlpir库授权文件
```
或者手动下载授权文件： https://github.com/NLPIR-team/NLPIR/tree/master/License/license%20for%20a%20month/NLPIR-ICTCLAS%E5%88%86%E8%AF%8D%E7%B3%BB%E7%BB%9F%E6%8E%88%E6%9D%83 下载NLPIR.user,覆盖到 envs\环境\Lib\site-packages\pynlpir\Data中

## 简单使用
### 代码结构
```
├── command
│   ├── evaluate.sh     # 验证集评估脚本
│   ├── pipeline.sh     # 训练脚本
│   └── predict_and_upload.sh   # 生成结果文件脚本
├── data                # 数据目录
│   ├── final_data
│   ├── preliminary_a_data
│   ├── preliminary_b_data
│   ├── news2016zh
│   └── weixin
├── gec_error_template  # 语义纠错模板
├── logs                # 存放训练日志文件
├── model               # 模型保存目录
│   ├── macbert_best
│   ├── pert_best
│   └── roberta_best
├── pretrained_model    # 预训练模型目录
│   ├── chinese-macbert-base
│   ├── chinese-pert-base
│   └── chinese-roberta-wwm-ext
├── result              # 结果文件保存目录
├── utils               # 数据增强、清洗工具等
└── src
    ├── __init__.py
    ├── baseline        # baseline系统
    ├── corrector.py    # 文本校对入口
    ├── evaluate.py     # 指标评估
    ├── metric.py       # 指标计算文件 
    ├── prepare_for_upload.py  # 生成要提交的结果文件
    └── train.py        # 训练入口
```

### 使用说明
1. 训练之前，保证数据集已经下载到目录中[here](#数据来源准备)
2. 下载预训练模型[here](#预训练模型准备下载)
3. model/ 目录中包含本次比赛已训练checkpoint，分别在 model/roberta_best，model/macbert_best，model/pert_best 目录中，可以直接加载进行推理评估或生成预测结果文件

### 使用已有checkpoint进行evaluate或upload
```
cd command
sh evaluate.sh      # 验证集评估
sh predict_and_upload.sh    # 生成预测结果文件，存放在result目录中
```

### 从头开始训练（包括数据准备与数据增强）
```
sh pipeline.sh
```