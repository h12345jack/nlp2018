
#coding=utf8
"""
Averaged perceptron classifier. Implementation geared for simplicity rather than
efficiency.
"""

import sys
import codecs
from collections import defaultdict
import pickle
import random





def _to_tags(sen):
    tags=[]
    for w in sen:
        if len(w)==1:
            tags.append('s')
        else:
            tags.append('b')
            for i in range(len(w)-2):tags.append('m')
            tags.append('e')
    return tags


def gen_keys(seq,i):
    mid=seq[i]
    left=seq[i-1] if i>0 else '#'
    left2=seq[i-2] if i-1>0 else '#'
    right=seq[i+1] if i+1<len(seq) else '#'
    right2=seq[i+2] if i+2<len(seq) else '#'
    return [mid+"_1",left+"_2",right+"_3",
            left+mid+'_1',mid+right+'_2',
            left2+left+'_3',right+right2+'_4']



class AveragedPerceptron(object):

    def __init__(self):
        # Each feature gets its own weight vector, so weights is a dict-of-dicts
        self.weights = {}
        self.classes = set(['b','m','e','s'])
        # The accumulated values, for the averaging. These will be keyed by
        # feature/clas tuples
        self._totals = defaultdict(int)
        # The last time the feature was changed, for the averaging. Also
        # keyed by feature/clas tuples
        # (tstamps is short for timestamps)
        self._tstamps = defaultdict(int)
        # Number of instances seen
        self.i = 0

    def predict(self, features,print_flag=2):
        # print features
        '''Dot-product the features and current weights and return the best class.'''
        scores = defaultdict(float)
        for feat in features:
            if feat not in self.weights:
                continue
            weights = self.weights[feat]
            for clas, weight in weights.items():
                scores[clas] += weight
        # Do a secondary alphabetic sort, for stability
        return max(self.classes, key=lambda clas: scores[clas])

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
        return pickle.dump(dict(self.weights), open(path, 'w'))

    def load(self, path):
        '''Load the pickled model weights.'''
        self.weights = pickle.load(open(path))
        return None

class Indexer(dict):
    def __init__(self,filename,mode='r'):
        '''初始化一个文件，基本上这个对于每一行，'''
        if mode=='r' or mode=='a':
            for line in open(filename):
                line=line.strip()
                self[line.strip().decode('utf8')]=len(self)
        self.append=False
        if mode=='w' or mode=='a':
            self.append=True
            self.file=file(filename,'w')
        
            
    def __call__(self,key):
        if self.append:
            if key not in self:
                self[key]=len(self)
                print>>self.file,key.encode('utf8')
                self.file.flush()
            return self[key]
        else:
            return self.get(key,-1)


def main_fea(src,index):
    '''训练文件index文件'''
    print 'begin'
    print src
    inder=Indexer(index,'w')#inder为一个模型
    f=codecs.open(src,'r','utf8')
    global_graph=[]
    for line in f.readlines():
        line=line.split()
        seq=''.join(line)
        graph=[]

        fs=[[inder(k) for k in gen_keys(seq,x)] for x in range(len(seq))]
        for c,v in zip(_to_tags(line),fs):
            graph.append([0,[],c,v])
        if not graph:continue
        graph[0][0]+=1;
        graph[-1][0]+=2;
        for i in range(1,len(graph)):
            graph[i][1]=[i-1]
        global_graph.extend(graph) 
    print 'the end'
    return global_graph,len(inder)

def test_fea(index,src):
    inder=Indexer(index,'r')
    f=codecs.open(src,'r','utf8')
    global_graph=[]
    for line in f.readlines():
        seq=line.strip()
        graph=[]
        fs=[filter(lambda x:x>=0,[inder(k) for k in gen_keys(seq,x)]) for x in range(len(seq))]
        for c,v in zip(_to_tags(line),fs):
            graph.append([0,[],c,v])
        if not graph:continue
        graph[0][0]+=1;
        graph[-1][0]+=2;
        for i in range(1,len(graph)):
            graph[i][1]=[i-1]
        global_graph.extend(graph)
    return global_graph

def train(nr_iter, examples,length):
    '''Return an averaged perceptron model trained on ``examples`` for
    ``nr_iter`` iterations.
    '''
    tt=['b','e','s','m']
    model = AveragedPerceptron()
    for j in range(int(length)):
        model.weights[str(j)]={tt[i]:0 for i in range(4)}
    for i in range(nr_iter):
        random.shuffle(examples)
        for features, class_ in examples:
            guess = model.predict(features)
            if guess != class_:
                model.update(class_, guess, features)
        print i,'iter done'
    model.average_weights()
    return model


def cut_the_text(index,src,model_dat,dst):
    graph2=test_fea(index,src)
    model=AveragedPerceptron()
    model.load(model_dat)
    f=codecs.open(src,'r','utf8')
    rs=f.readlines()
    data=[]
    for i in graph2:
        result=model.predict(i[3])
        data.append(result)
    num=0
    f2=file(dst,'w')
    for line in rs:
        line=line.strip()
        length=len(line)
        rs_line=[]
        for i in range(length):
            if data[num]=='e' or data[num]=='s':
                rs_line.append(line[i].encode('utf8')+' ')
            else:
                rs_line.append(line[i].encode('utf8'))
            num+=1
        rs_line=''.join(rs_line)
        print>>f2,rs_line
        print ''

def stats(my_rs,right_rs):
    f1=codecs.open(my_rs,'r','utf8')
    f2=codecs.open(right_rs,'r','utf8')
    my_rs=f1.readlines()
    right_rs=f2.readlines()
    t1=0#所有的
    t2=0#正确的
    t3=0#我的所有的
    t4=0
    for i,j in zip(my_rs,right_rs):
        i=i.strip()
        j=j.strip()
        num1=i.split()
        num2=j.split()
        my=dict()
        rg=dict()
        for i1 in num1:
            i1=i1.encode('utf8')            
            if i1 not in my:
                my[i1]=1
            else:
                my[i1]+=1
            t3+=1
        for i1 in num2:
            i1=i1.encode('utf8')
            if i1 not in rg:
                rg[i1]=1
            else:
                rg[i1]+=1
            t4+=1
        all_num=0
        right=0
        for i1 in rg.keys():
            all_num+=rg[i1]
            if i1 in my:
                right+=my[i1]
        print all_num,right
        t1+=all_num
        t2+=right
    p=t2*1.0/t1
    r=t2*1.0/t3
    print t1,t2,t3
    print p,r
    print 'Feasures:',2*p*r/(p+r)
    
if __name__ == '__main__':
    graph,length=main_fea('train-pku.txt','index4.json')
    examples=[]
    for i in graph:
        examples.append((i[3],i[2]))
    model=train(5, examples,length)
    model.save('model4.dat')
    cut_the_text('index4.json','test-pku.txt','model4.dat','rs4.txt')
    # stats('rs4.txt','test.answer.txt')