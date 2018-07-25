import React, {Component} from 'react';
import ReactMarkdown from 'react-markdown'
import './md.css';


const input = `
## 任务:中文分词（Chinese Word Segment)

>    实现一个汉语自动分词系统（Chinese word segmentation）

>    本题目要求实现一个汉语自动分词系统，并在微博等非规范文本测试集上进行测试分析。

>    如果在本题目中不考虑命名实体识别问题，歧义消解和集外词 处理是汉语自动分词中的关键问题  

## 数据集

1.  [icwb2-data](http://sighan.cs.uchicago.edu/bakeoff2005/)

2.  [NLPCC2016-WordSeg-Weibo](https://github.com/FudanNLP/NLPCC-WordSeg-Weibo)

其中，PKU MSR为标准数据集，Weibo为非标准数据集。

为了验证模型的领域适应问题，使用PKU训练的数据集合训练，使用Weibo作为测试数据

## 方法

使用了多种分词方法，包括：

- [x] Maximum Match
- [x] Perceptron
- [x] CRF++
- [x] Bi-LSTM
- [x] HMM

这些模型中部分使用机器学习的方法，部分使用词典规则。
我们使用[SIGHAN的脚本](http://sighan.cs.uchicago.edu/bakeoff2005/)进行测试验证

## 系统设计

我们设计了一个系统，能够选择不同的模型和数据集，对给定的句子进行划分。截图如下：

![image](http://ws2.sinaimg.cn/large/006C73MUgy1ftmd62ieogj30o00codh6.jpg)

### 反馈机制

由于不同领域的数据存在极大的差距，标准数据集很难迁移到如微博等日常化用语的数据集上，因此构建日常化用语的数据集
是非常重要的步骤。

但数据集的构建费时费力，一个人通常难以完成。而[众包](https://baike.baidu.com/item/%E4%BC%97%E5%8C%85/548975)则提供了很好的思路，将切词任务自由自愿的形式外包给非特定的大众网络，然后对收集到的数据进行清洗，便能够收集到有价值的数据。相关的著名数据集是[ImageNet](http://www.image-net.org/)

我们设计的反馈机制如下：

![image](http://wx3.sinaimg.cn/large/006C73MUgy1ftmds9erk6j313i0od7aa.jpg)

`


export default class Description extends Component {

    render(){
        return <ReactMarkdown source={input} />
    }
}