# NLP2018 UCAS

## Description
> 实现一个汉语自动分词系统（Chinese word segmentation）

> 本题目要求实现一个汉语自动分词系统，并在微博等非规范文本测试集上进
行测试分析。如果在本题目中不考虑命名实体识别问题，歧义消解和集外词
处理是汉语自动分词中的关键问题

## Data && Experiment
1.  [icwb2-data](http://sighan.cs.uchicago.edu/bakeoff2005/)

2.  [NLPCC2016-WordSeg-Weibo](https://github.com/FudanNLP/NLPCC-WordSeg-Weibo)

PKU MSR: 标准数据集
Weibo: 非标准数据集

## Methods
- [x] 最大正向匹配 hjj
- [x] perceptron hjj
- [x] CRF++ wj
- [x] HMM & hl
- [x] Bi-LSTM+CRF hl

## System Design

- [x] SPA-Form Page
- [x] Results Page
- [ ] Feedback (反馈机制，前端完成)

![](./imgs/system.gif)



## Install

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```
注：crf需要额外的安装库，见[说明](./models/crf/readme.md)




###  相关资料
https://github.com/hankcs/multi-criteria-cws

http://jkx.fudan.edu.cn/~qzhang/paper/aaai2017-cws.pdf

http://www.shizhuolin.com/2018/05/29/2920.html


