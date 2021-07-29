
# KnowQA

knowQA Source Code 

[预训练模型知识量度量比赛](https://www.datafountain.cn/competitions/509)Baseline模型, 使用BERT进行端到端的fine-tuning, 平台评测F1值0.35。

inference时提供两种decode方式(argparse的mode参数)

mode=1: top5只解码生成单term的mask情况。

mode=2(默认): 单term mask生成三个, 双term生成1个, 三term生成一个。

# Preinstallation

Before launch the script install these packages in your **Python3** environment:
- pytorch >= 1.4
- transformers 4.2.0

建议使用Conda安装 :) 


```
 conda create -n knowqa -c pytorch python=3.6 pytorch
 conda activate knowqa
 pip install transformers==4.2.0 tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

下载Huggingface BERT模型的```pytorch_model.bin```, 并放至```./models/bert/```
```
git clone https://github.com/zhengyima/knowqa.git knowqa
cd knowqa
wget -O ./models/bert/pytorch_model.bin https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/bert-base-cased-pytorch_model.bin
```

# Launch the script

环境配好，模型下好之后便可以运行代码了. 代码会自动在训练集上进行训练，并在testset上进行预测。最终在生成文件至```./output/score.txt```,可直接用于提交。

```
 python runBert.py
```

注: ```./data/```下面已经提供了预处理的数据。为什么没有提供处理数据的脚本？因为经过迭代，生成当时这个数据格式的脚本已经不见了:(



# Experimental Resuls

| Models | F1 on testset | 
| :---------------- | :---------------|
| Official baseline | 0.172 |
| Ours (mode=1) | 0.2574 |
| Ours (mode=2) | 0.3536 |


## Links
- https://huggingface.co/

