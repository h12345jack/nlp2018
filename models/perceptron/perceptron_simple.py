# coding=utf-8
__author__ = 'h12345jack'

import sys
from collections import defaultdict
import pickle
import random

from tqdm import tqdm

class AveragedPerceptron(object):
    """docstring for Perceptron."""
    def __init__(self):
        super(AveragedPerceptron, self).__init__()
        self.weights = {}
        self.classes = set(['b','m','e','s'])
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    

    def predict(self, features):
        # print features
        '''Dot-product the features and current weights and return the best class.'''
        scores = defaultdict(float)
        for feat in features:
            if feat not in self.weights:
                continue
            weights = self.weights[feat]
            for cls, weight in weights.items():
                scores[cls] += weight
        # Do a secondary alphabetic sort, for stability
        return max(self.classes, key=lambda cls: scores[cls])

    def update(self, truth, guess, features):
        '''Update the feature weights.'''
        def upd_feat(c, f, w, v):
            param = (f, c)
            self._totals[param] += (self.i - self._tstamps[param]) * w
            self._tstamps[param] = self.i
            self.weights[f][c] = w + v

        self.i += 1
        if truth == guess:
            return None
        for f in features:
            weights = self.weights.setdefault(f, {})
            upd_feat(truth, f, weights.get(truth, 0.0), 1.0)
            upd_feat(guess, f, weights.get(guess, 0.0), -1.0)
        return None

    def average_weights(self):
        '''Average weights from all iterations.'''
        for feat, weights in self.weights.items():
            new_feat_weights = {}
            for clas, weight in weights.items():
                param = (feat, clas)
                total = self._totals[param]
                total += (self.i - self._tstamps[param]) * weight
                averaged = round(total / float(self.i), 3)
                if averaged:
                    new_feat_weights[clas] = averaged
            self.weights[feat] = new_feat_weights
        return None

    def save(self, path):
        '''Save the pickled model weights.'''
        return pickle.dump(dict(self.weights), open(path, 'wb'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path, 'rb'))
        return None

class Indexer(dict):
    def __init__(self, filename, mode='r'):
        '''初始化一个文件，基本上这个对于每一行，'''
        self.dict = dict()
        if mode == 'r' or mode == 'a':
            for line in open(filename, encoding='utf8'):
                line=line.strip()
                self.dict[line]=len(self.dict)
        self.app_flag = False
        if mode == 'w' or mode=='a':
            self.app_flag = True
            self.file = open(filename, 'w', encoding='utf8')
        
    def __call__(self,key):

        if self.app_flag:
            if key not in self.dict:
                self.dict[key]=len(self)
                print(key, file = self.file)
                self.file.flush()
            return self.dict[key]
        else:
            return self.dict.get(key,-1)


class PerceptronCWS(object):
    """docstring for PerceptronCWS."""
    def __init__(self, train_src, train_dst):
        super(PerceptronCWS, self).__init__()
        self.train_src = train_src
        self.train_dst = train_dst


    def _to_tags(self, sen):
        tags=[]
        for w in sen:
            if len(w) == 1:
                tags.append('s')
            else:
                tags.append('b')
                for i in range(len(w)-2):
                    tags.append('m')
                tags.append('e')
        return tags


    def gen_keys(self, seq, i):
        '''产生feature'''
        mid=seq[i]
        left=seq[i-1] if i>0 else '#'
        left2=seq[i-2] if i-1>0 else '#'
        right=seq[i+1] if i+1<len(seq) else '#'
        right2=seq[i+2] if i+2<len(seq) else '#'
        return [mid+"_1",left+"_2",right+"_3",
                left+mid+'_1',mid+right+'_2',
                left2+left+'_3',right+right2+'_4']

    def main_fea(self, src, index):
        '''训练文件index文件'''
        print('begin index...')
        print('from', src, 'to', index)
        inder = Indexer(index,'w') #build indexer
        global_graph=[]

        with open(src, encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split()
                seq = ''.join(line)
                graph = []
                fs = [[inder(k) for k in self.gen_keys(seq, x)] for x in range(len(seq))]
        
                for c,v in zip(self._to_tags(line),fs):
                    graph.append([0,[],c,v])
                if not graph:continue
                graph[0][0]+=1;
                graph[-1][0]+=2;
                for i in range(1,len(graph)):
                    graph[i][1]=[i-1]
                global_graph.extend(graph)
            return global_graph, len(inder.dict)

    def test_fea(self, index, src):
        inder = Indexer(index,'r')
        global_graph=[]
        with open(src,'r',encoding='utf8') as f:
            for line in tqdm(f.readlines()):
                seq=line.strip()
                graph=[]
                fs=[filter(lambda x:x>=0,[inder(k) for k in self.gen_keys(seq,x)]) for x in range(len(seq))]
                for c,v in zip(self._to_tags(line),fs):
                    graph.append([0,[],c,v])
                if not graph:continue
                graph[0][0]+=1
                graph[-1][0]+=2
                for i in range(1,len(graph)):
                    graph[i][1]=[i-1]
                global_graph.extend(graph)
            return global_graph

    def train(self, nr_iter, examples,length):
        '''Return an averaged perceptron model trained on ``examples`` for
        ``nr_iter`` iterations.
        '''
        tt=['b','e','s','m']
        model = AveragedPerceptron()
        for j in range(int(length)):
            model.weights[str(j)]={tt[i]:0 for i in range(4)}
        print('begin training...')
        for i in tqdm(range(nr_iter)):
            random.shuffle(examples)
            for features, class_ in examples:
                guess = model.predict(features)
                if guess != class_:
                    model.update(class_, guess, features)
            print(i,'iter done')
        model.average_weights()
        return model

    def train_it(self):
        index_fpath = self.train_dst[:self.train_dst.rfind('.')] + '.index'
        model_fpath = self.train_dst[:self.train_dst.rfind('.')] + '.model'
        graph, length = self.main_fea(self.train_src, index_fpath)
        examples = []
        for i in graph:
            examples.append((i[3], i[2]))
        model = self.train(5, examples, length)
        model.save(model_fpath)

def test_fea(index, src):
    inder = Indexer(index,'r')
    global_graph=[]
    with open(src,'r',encoding='utf8') as f:
        for line in tqdm(f.readlines()):
            seq=line.strip()
            graph=[]
            fs=[filter(lambda x:x>=0,[inder(k) for k in gen_keys(seq,x)]) for x in range(len(seq))]
            for c,v in zip(_to_tags(line),fs):
                graph.append([0,[],c,v])
            if not graph:continue
            graph[0][0]+=1
            graph[-1][0]+=2
            for i in range(1,len(graph)):
                graph[i][1]=[i-1]
            global_graph.extend(graph)
        return global_graph


def _to_tags(sen):
    tags=[]
    for w in sen:
        if len(w) == 1:
            tags.append('s')
        else:
            tags.append('b')
            for i in range(len(w)-2):
                tags.append('m')
            tags.append('e')
    return tags


def gen_keys(seq, i):
    '''产生feature'''
    mid=seq[i]
    left=seq[i-1] if i>0 else '#'
    left2=seq[i-2] if i-1>0 else '#'
    right=seq[i+1] if i+1<len(seq) else '#'
    right2=seq[i+2] if i+2<len(seq) else '#'
    return [mid+"_1",left+"_2",right+"_3",
            left+mid+'_1',mid+right+'_2',
            left2+left+'_3',right+right2+'_4']


def cut_the_text(index,src,model_dat,dst):
    graph2=test_fea(index,src)
    model=AveragedPerceptron()
    model.load(model_dat)
    f=open(src, 'r', encoding='utf8')
    rs=f.readlines()
    data=[]
    for i in graph2:
        result=model.predict(i[3])
        data.append(result)
    num=0
    f2=open(dst,'w', encoding='utf8')
    for line in rs:
        line=line.strip()
        length=len(line)
        rs_line=[]
        for i in range(length):
            if data[num]=='e' or data[num]=='s':
                rs_line.append(line[i]+' ')
            else:
                rs_line.append(line[i])
            num+=1
        rs_line=''.join(rs_line)
        print(rs_line, file=f2)
    

def main():
    # model1 = PerceptronCWS('../datas/pku_training.utf8', './model/pku.model')
    # model1.train_it()
    cut_the_text('./model/pku.index', '../datas/pku_test.utf8', './model/pku.model', 'rs.txt')


if __name__ == '__main__':
    main()