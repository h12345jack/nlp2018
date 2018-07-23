import React, {Component} from 'react';
import ReactMarkdown from 'react-markdown'

const input = `
# 结果

## 数据集
1.  [SIGHAN](http://sighan.cs.uchicago.edu/bakeoff2005/)
2.  [NLPCC 2016 微博分词](https://github.com/FudanNLP/NLPCC-WordSeg-Weibo)

## 结果
`


export default class Results extends Component {

    render(){
        return <ReactMarkdown source={input} />
    }
}