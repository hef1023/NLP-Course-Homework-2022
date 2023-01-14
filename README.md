# **实践报告**

> 学号：2271981
>
> 姓名：赫洱锋

本项目基于pytorch框架，使用Helsinki-NLP预训练模型实现中英机器翻译任务，在500条数据上测试得到的平均BLEU分数为26.96。

数据集选择News Commentary-V13，选取了其中5000条数据，4500条用作训练，处理拆分后加载为dataset，使用huggingface的API加载Helsinki-NLP进行微调。

# 模型安装使用

运行环境：
```python
pip install -r ./requirements.txt
```

或者

```python
conda install --yes --file requirements.txt
```

代码中import的库

```python
from datasets import load_dataset,load_metric
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,Seq2SeqTrainingArguments,DataCollatorForSeq2Seq,Seq2SeqTrainer
import numpy as np
```

代码运行

```python
python dataProcess.py
python translate.py
```

# 数据集下载

数据集下载链接https://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz

# 项目结构

- NLP-Course-Homework-2022
  - dataProcess.py——数据处理文件
  - BLEU.py——评估指标BLEU
  - translate.py——模型加载，训练与预测
  - data
    - train.txt——处理完的训练集
    - val.txt——处理完的验证集
    - zh-en.en——源数据集中的英文文本数据
    - zh-en.zh——源数据集中的中文文本数据
  - translations
    - checkpoint-n——保存的模型数据
 
 #处理后的数据样例
 
 ```python
{'en': 'PARIS – As the economic crisis deepens and widens, the world has been searching for historical analogies to help us understand what has been happening.', 'zh': '巴黎-随着经济危机不断加深和蔓延，整个世界一直在寻找历史上的类似事件希望有助于我们了解目前正在发生的情况。'}
{'en': 'They set inflation targets at around 2% – leaving little room for maneuver when the water got choppy.', 'zh': '它们将通胀目标设定在2%左右——这意味着当波涛汹涌时他们根本没有多少施展空间。'}
```



