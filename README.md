# 垃圾邮件判定器

本项目分为两个部分 `Spam-Detector-model` `Spam-Detector-front`，大致结构如下：

```arduino
Spam-DETECTOR
├── Spam-Detector-model
│   ├── .ipynb_checkpoints
│   ├── save_model
│   ├── CNN+决策树.ipynb
│   ├── LSTM.ipynb
│   ├── 朴素贝叶斯.ipynb
│   └── email_text.csv
├── Spam-Detector-front
│   ├── save_model
│   ├── static
│   ├── templates
│   └── app.py
```

## 数据集

[TREC 2007 Public Corpus Dataset](https://www.kaggle.com/datasets/imdeepmind/preprocessed-trec-2007-public-corpus-dataset)

 `Spam-Detector-model` 文件夹下的`email_text.csv`文件。

## 环境配置

本项目使用 Python 3.9 进行部署，其余环境不可保证。

 `Spam-Detector-model` 下的模型实现均为Jupyter， `Spam-Detector-front`为Flask搭建。

先在 `Spam-Detector-model` 运行如下命令，进行库的安装：

```bash
pip install -r requirements.txt
```

*建议 `Spam-Detector-front`下选择同样的解释器。如有特殊需求，需要独立解释器，需安装该文件下的`requirements.txt`

## Spam-Detector-model

用于垃圾邮件判定器模型训练与评估的相关文件夹，实现了CNN, Bi-LSTM, DecisionTree, NaiveBayes四种深度学习模型。

首先运行如下命令启动jupyter：

```bash
jupyter notebook
```

其余可直接通过jupyter进行操作。

*如显示 nltk 无法导入，尝试科学上网。

![image-20241211062507879](https://gitee.com/r0otsu/images/raw/master/image-20241211062507879.png)

## Spam-Detector-front

搭建垃圾邮件判定器前后端的相关文件夹，使用Flask进行部署，将输入的邮件内容进行预测后，判定其是否为垃圾邮件，并输出结果。

使用如下命令打开前端页面，脚本中设置端口为8080：

```bash
python app.py
```

*如显示 nltk 无法导入，尝试科学上网。

输入待预测文本，点击`PREDICT`后，即可得到四个模型的预测结果：

![image-20241211070544162.png](https://gitee.com/r0otsu/images/raw/master/image-20241211070544162.png)
