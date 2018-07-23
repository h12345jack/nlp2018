import React, {Component} from 'react';
import PropTypes from 'prop-types';
import {Icon} from 'antd';
import { withStyles } from '@material-ui/core/styles';
import MenuItem from '@material-ui/core/MenuItem';
import TextField from '@material-ui/core/TextField';

export default class System extends Component {
    state = {
        text: 'hell'
    }

    handleChange = (e)=>{
        const value = e.target ? e.target.value : e; 
        this.setState({
            text: value
        })
    }

    render(){
        return <TextField
                    id="multiline-flexible"
                    label="输入切词文本信息"
                    multiline
                    rows = '4'
                    rowsMax="10"
                    value={this.state.text}
                    onChange={this.handleChange}
                    // className={classes.textField}
                    InputLabelProps={{
                        shrink: true,
                    }}
                    placeholder="输入你想要切词的文本"
                    helperText={<span><Icon type="info-circle-o" />复制粘贴将会节省大量时间哦</span>}
                    fullWidth
                    margin="normal"
                />

    }
}