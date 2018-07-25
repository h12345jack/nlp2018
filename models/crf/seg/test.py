#coding:utf-8              #由于.py文件是utf-8的，所以必须有这一句
from crf_api import *
import sys

#input_str = "（二○○○年十二月三十一日）（附图片1张）"
input_str = "【缩量回调，3000点还要徘徊多久？】"
print(type(input_str))
crf_pku = CRF('pku')
crf_msr = CRF('msr')
crf_weibo = CRF('weibo')
print(crf_pku.cut(input_str))
print(crf_msr.cut(input_str))
print(crf_weibo.cut(input_str))

