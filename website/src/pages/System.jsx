import React, {Component} from 'react';
import PropTypes from 'prop-types';
import {Icon} from 'antd';
import { withStyles } from '@material-ui/core/styles';
import MenuItem from '@material-ui/core/MenuItem';
import TextField from '@material-ui/core/TextField';
import LinearProgress from '@material-ui/core/LinearProgress';
import CircularProgress from '@material-ui/core/CircularProgress';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';


import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormHelperText from '@material-ui/core/FormHelperText';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import FormLabel from '@material-ui/core/FormLabel';

import CWS from '../components/cws';


const styles = {
    title: {
      marginBottom: 16,
      marginTop: 16,
    },
    formControl:{
        marginTop: 20,
        marginRight: 80,
    }
}

class System extends Component {
    state = {
        text: '成为专业程序员的6个技巧',
        loading: false,
        model: "hmm",
        dataset: "pku"
    }

    handleTextChange = (e)=>{
        const value = e.target ? e.target.value : e; 
        this.setState({
            text: value
        })
    }

    handleModelChange = (e)=>{
        const value = e.target ? e.target.value: e;
        this.setState({
            model: value
        })
    }

    handleDataSetChange = (e)=>{
        const value = e.target ? e.target.value : e;
        this.setState({
            dataset: value
        })
    }

    render(){
        const { classes } = this.props;
        return <div>
                <Typography className={classes.title} variant="headline" component="h2">
                    在线系统
                </Typography>
                    {this.state.loading && <CircularProgress style={{top:10, left:'50%', width:40, position:'fixed', zIndex:5000}} color="secondary"/>}
                    <TextField
                        id="multiline-flexible"
                        label="输入切词文本信息"
                        multiline
                        rows = '4'
                        rowsMax="10"
                        value={this.state.text}
                        onChange={this.handleTextChange}
                        InputLabelProps={{
                            shrink: true,
                        }}
                        placeholder="输入你想要切词的文本"
                        helperText={<span><Icon type="info-circle-o" />复制粘贴将会节省大量时间哦</span>}
                        fullWidth
                        margin="normal"
                    />
                    {/* <Grid container spacing={24}>
                <Grid item xs={12}>
                <Paper className={classes.paper}>xs=12</Paper>
                </Grid>
        <Grid item xs={12} sm={6}>
          <Paper className={classes.paper}>xs=12 sm=6</Paper>
        </Grid> */}
                       <FormControl component="fieldset" required className={classes.formControl}>
                        <FormLabel component="label" style={{fontSize: '0.75em'}}>模型选择</FormLabel>
                        <RadioGroup
                            aria-label="模型选择"
                            name="models1"
                            className={classes.group}
                            value={this.state.model}
                            onChange={this.handleModelChange}
                            row
                        >
                            <FormControlLabel value="hmm" control={<Radio />} label="HMM" />
                            <FormControlLabel value="perceptron" control={<Radio />} label="Perceptron" />
                            <FormControlLabel value="lstm" control={<Radio />} label="LSTM" />
                            <FormControlLabel value="crf" control={<Radio />} label="CRF" />
                        </RadioGroup>
                        </FormControl>
                        <FormControl component="fieldset" required className={classes.formControl}>
                            <FormLabel component="label" style={{fontSize: '0.75em'}}>数据集选择</FormLabel>
                            <RadioGroup
                                aria-label="数据集选择"
                                name="dataset1"
                                className={classes.group}
                                value={this.state.dataset}
                                onChange={this.handleDataSetChange}
                                row
                            >
                            <FormControlLabel value="pku" control={<Radio />} label="PKU" />
                            <FormControlLabel value="msr" control={<Radio />} label="MSR" />
                            <FormControlLabel value="weibo" control={<Radio />} label="Weibo" />
                        </RadioGroup>
                        </FormControl>


                    <div style={{textAlign: 'right'}}>
                    <Button size="large" color="primary" variant="outlined">
                       提交
                    </Button>
                    </div>
                    <CWS />
                </div>

    }
}

System.propTypes = {
    classes: PropTypes.object.isRequired,
  };
  
  
export default withStyles(styles)(System);