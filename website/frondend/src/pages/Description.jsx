import React, {Component} from 'react';
import ReactMarkdown from 'react-markdown'

const input = `
# 任务--中文分词（cws)

## 
`


export default class Description extends Component {

    render(){
        return <ReactMarkdown source={input} />
    }
}