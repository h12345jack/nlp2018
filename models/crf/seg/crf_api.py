#-*-coding:utf-8-*-
 
#CRF Segmenter based character tagging:
# 4-tags for character tagging: B(Begin), E(End), M(Middle), S(Single)
 
import codecs
import sys
 
import CRFPP

class CRF(object): 
    def __init__(self, model_type):
        if model_type in ['pku', 'msr', 'weibo']:
            self.model = "./model/"+model_type+".model"
            self.tagger = CRFPP.Tagger("-m " + self.model)
        else:
            raise Exception("CRF's model type must be pku, msr or weibo.")

    def cut(self, input_str):
        tagger = self.tagger
        tagger.clear()
        for word in input_str.strip():
            word = word.strip()
            if word:
                tagger.add((word + "\to\tB"))

        tagger.parse()
        size = tagger.size()
        xsize = tagger.xsize()
        output_str = ""
        for i in range(0, size):
            for j in range(0, xsize):
                char = tagger.x(i, j)
                tag = tagger.y2(i)
                if tag == 'B':
                    output_str += (' ' + char)
                elif tag == 'M':
                    output_str += (char)
                elif tag == 'E':
                    output_str += (char + ' ')
                else:
                    output_str += (' ' + char + ' ')
        return output_str
