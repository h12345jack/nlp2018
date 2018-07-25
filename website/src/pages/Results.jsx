import React, {Component} from 'react';
import ReactMarkdown from 'react-markdown'
import Select from '@material-ui/core/Select';
import MenuItem from '@material-ui/core/MenuItem';

import CustomizedTable from '../components/CustomizedTable';

const input = `
## 结果

### 任务
1.  pku标准数据集：train(19056), test(1945)
2.  msr标准数据集: train(86924), test(3985)
3.  微博数据集: train(20135), test(2052)
4.  使用pku标准数据集进行训练，使用微博数据集进行测试

### 结果
`


let id = 0;
function createData(name, p, r, f1) {
  id += 1;
  return { id, name, p, r, f1 };
}

const nameList = ['HMM', 'MaxMatch', 'Perceptron', 'CRF', 'BiLSTM+CRF']
const datas = {
    'pku': [
        [0.803,	0.789,	0.796],
        [0.843,	0.907,	0.874],
        [0.933,	0.916,	0.925],
        [0.937,	0.922,	0.929],
    ],
    'msr': [
        [0.788, 0.804, 0.796],
        [0.917, 0.957, 0.937],
        [0.968, 0.953, 0.955],
        [0.964, 0.965, 0.965],
    ],
    'weibo': [
        [0.805, 0.814, 0.810],
        [0.836, 0.932, 0.882],
        [0.896, 0.947, 0.921],
        [0.929, 0.937, 0.933],
    ],
    'pku2weibo': [
        [0.742,	0.730,	0.736],
        [0.696,	0.860,	0.769],
        [0.834,	0.912,	0.872],
        [0.812,	0.842,	0.827],
    ]
}
const results = Object.keys(datas).map((key)=>{
    const tmp = datas[key];
    return tmp.map((item, index)=>{
        const pku = {}
        pku['name'] = nameList[index];
        pku['p'] = item[0];
        pku['r'] = item[1];
        pku['f1'] = item[2];
        return pku;
    })
})

export default class Results extends Component {
    state = {
        dataset: 'pku'
    }

    handleChange = event => {
        this.setState({ dataset : event.target.value });
    };

    render(){
        const dataName = Object.keys(datas);
        const dataIndex = dataName.findIndex( k=> this.state.dataset === k);
        return <div>
                <ReactMarkdown source={input} />
                <Select
                    value={this.state.dataset}
                    onChange={this.handleChange}
                    inputProps={{
                    id: 'age-simple',
                    }}
                >
                    <MenuItem value="pku">
                    PKU
                    </MenuItem>
                    <MenuItem value="msr">MSR</MenuItem>
                    <MenuItem value="weibo">Weibo</MenuItem>
                    <MenuItem value="pku2weibo">PKU2Weibo</MenuItem>
                </Select>
                <CustomizedTable data={results[dataIndex]}/>
               </div>
    }
}