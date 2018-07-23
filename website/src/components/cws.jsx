import React from 'react';

import {Icon} from 'antd';
import PropTypes from 'prop-types';
import { withStyles } from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import CardActions from '@material-ui/core/CardActions';
import CardContent from '@material-ui/core/CardContent';
import Button from '@material-ui/core/Button';
import Typography from '@material-ui/core/Typography';
import Divider from '@material-ui/core/Divider';
import TextField from '@material-ui/core/TextField';
import DoneIcon from '@material-ui/icons/Done';
import IconButton from '@material-ui/core/IconButton';

import './cws.css';

const styles = {
  result:{
    marginTop: 20
  },
  card: {
    marginTop: 20,
    minWidth: 275,
  },
  bullet: {
    display: 'inline-block',
    margin: '0 2px',
    transform: 'scale(0.8)',
  },
  title: {
    marginBottom: 16,
    marginTop: 16,
  },
  pos: {
    marginBottom: 12,
  },
  button:{
    padding: 0,
  },
  button2:{
    padding: 0,
    fontSize: '0.8em',
  },
  seg:{
    paddingLeft: 10,
    paddingRight: 10,
  }
};

class CWS extends React.Component{
  state = {
    text: "",
    edit_confirm: false,
    to_do_edit: false,
  }

  componentWillMount(){
    this.setState({
      text: "【 神秘 中国 财团 本周 或 买下 AC米兰 ， 代价 超 10亿 欧元 】"
    })
  }

  handleTextChange = (e)=>{
    const value = e.target ? e.target.value : e; 
    this.setState({
        text: value
    })
}

  render(){
    const { classes } = this.props;
    const bull = <span className={classes.bullet}>•</span>;

    const sen_cutted = "【 神秘 中国 财团 本周 或 买下 AC米兰 ， 代价 超 10亿 欧元 】";

    const words = sen_cutted.split(' ');
    return <div className={classes.result}>
      <Typography className={classes.title} variant="headline" component="h2">
          切词结果
      </Typography>
      <Card className={classes.card}>
        <CardContent>
         
          <Typography variant="subheading" component="h4">
          原句:
          </Typography>
          <Typography className={classes.pos} color="textSecondary">
          {words.map((item, index)=>{
              return <span>{item}</span>
            })
          }
          </Typography>
          <Typography variant="subheading" component="h4">
          切分后:
          </Typography>
          <Typography className={classes.pos} color="textSecondary">
          {words.map((item, index)=>{
              if(index == 0){
                return <span>{item}</span>
              }
              return <span className={classes.seg}>{item}</span>
            })
          }
          {!this.state.to_do_edit && <Icon type="edit" onClick={()=>this.setState({to_do_edit:true})}/>}
          </Typography>
          {(()=>{
            if(this.state.to_do_edit){
              return <div>
                      <Typography variant="subheading" component="h4">
                      编辑:
                      </Typography>
                      {this.state.edit_confirm 
                        ? [this.state.text.split(" ").map((item, index)=>{
                                  if(index == 0){
                                    return <span>{item}</span>
                                  }
                                  return <span className={classes.seg}>{item}</span>
                              }),
                          <Icon type="edit" onClick={()=>{this.setState({edit_confirm:false})}}/>,
                          <Icon type="cloud-upload-o" style={{marginLeft: 10}}/>  
                          ]
                        : [<TextField
                              id="multiline-flexible"
                              multiline
                              rows = '2'
                              rowsMax="10"
                              value={this.state.text}
                              onChange={this.handleTextChange}
                              InputLabelProps={{
                                  shrink: true,
                              }}
                              placeholder="输入你想要切词的文本"
                              helperText={<span><Icon type="info-circle-o" />空格分隔</span>}
                              fullWidth
                              margin="normal"
                          />,
                          <div style={{textAlign: 'right', marginTop: -30}}>
                            <IconButton color="primary" className={classes.button2} component="span" onClick={()=>this.setState({edit_confirm: true})}>
                              <DoneIcon style={{ fontSize: '1.2em' }} />
                            </IconButton>
                          </div>]
                      }
                    </div>
              }
            })()}

        </CardContent>
      </Card>
    </div>
  }  
}
CWS.propTypes = {
  classes: PropTypes.object.isRequired,
};


export default withStyles(styles)(CWS);

