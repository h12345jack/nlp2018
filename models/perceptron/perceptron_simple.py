# coding=utf-8
__author__ = 'h12345jack'

import os
from collections import defaultdict
import pickle
import random

from tqdm import tqdm

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
UPPER_DIR = os.path.join(CUR_DIR, '..')
MODEL_DIR = os.path.join(CUR_DIR, 'model')
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


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
        return 

    def save(self, path):
        '''Save the pickled model weights.'''
        with open(path, 'wb') as f:
            pickle.dump(dict(self.weights), f)
        return 

    def load(self, path):
        '''Load the pickled model weights.'''
        with open(path, 'rb') as f:
            self.weights = pickle.load(f)
        return 


def _to_tags(sen):
    '''对应一个句子生成tags'''
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


class PerceptronCWS(object):
    """docstring for PerceptronCWS."""
    def __init__(self, dataset, iters=5):
        super(PerceptronCWS, self).__init__()
        self.dataset = dataset
        self.iters = iters
        pku = '{}_train'.format(dataset)
        self.train_src = os.path.join(UPPER_DIR, 'datas', pku +'.txt')
        self.model_path = os.path.join(CUR_DIR, 'model', pku + '.model')
        self.index_fpath = self.model_path[:self.model_path.rfind('.')] + '.index'


        self.dic = self.load_dic()
        self.inder_file = open(self.index_fpath, 'a')
        self.model = self.load_model()
    
    def load_dic(self):
        dic = dict()
        if os.path.exists(self.index_fpath):
            with open(self.index_fpath, encoding='utf8') as f:
                for line in f.readlines():
                    line=line.strip()
                    dic[line]=len(dic)
        return dic

    def index(self, key, read_only=True):
        if read_only:
            return self.dic.get(key, -1)
        else:
            if key not in self.dic:
                self.dic[key] = len(self.dic)
                print(key, file = self.inder_file)
                self.inder_file.flush()
            return self.dic[key]

    def load_model(self):
        if not os.path.exists(self.model_path):
            print('can not find model')
            print('begin index')
            self.inder_file.close()
            os.remove(self.index_fpath)
            self.inder_file = open(self.index_fpath, 'a', encoding='utf8')
            global_graph = []
            with open(self.train_src, encoding='utf8') as f:
                for line in tqdm(f.readlines()):
                    line = line.strip().split()
                    seq = ''.join(line)
                    graph = []
                    fs = [[self.index(k, read_only=False) for k in gen_keys(seq, x)] for x in range(len(seq))]
                    for c, v in zip(_to_tags(line), fs):
                        graph.append([0, [], c, v])
                    if not graph:
                        continue
                    graph[0][0] += 1
                    graph[-1][0] += 2
                    for i in range(1,len(graph)):
                        graph[i][1]=[i-1]
                    global_graph.extend(graph)
            print('index done!')
            examples = [(i[3], i[2]) for i in global_graph]
            model = self.train(self.iters, examples, len(self.dic))
            model.save(self.model_path)
        else:
            model=AveragedPerceptron()
            model.load(self.model_path)
        
        return model

    def train(self, nr_iter, examples, length):
        '''Return an averaged perceptron model trained on ``examples`` for
        ``nr_iter`` iterations.
        '''
        tt = ['b','e','s','m']
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

    def cut(self, sentence):
        seq = sentence.strip()
        graph = []
        fs = [list(filter(lambda x: x>=0, [self.index(k) for k in gen_keys(seq, x)])) for x in range(len(seq))]

        for c, v in zip(_to_tags(seq), fs):
            graph.append([0, [], c, v])
        if len(graph) == 0:
            return sentence
        graph[0][0]+=1
        graph[-1][0]+=2
        for i in range(1, len(graph)):
            graph[i][1]=[i-1]
        flags = [self.model.predict(i[3]) for i in graph]
        results = []
        for char, flag in zip(seq, flags):
            if flag == 'e' or flag == 's':
                results.append(char + ' ')
            else:
                results.append(char)
        return ''.join(results).strip()   
 

def test(dataset='pku'):
    model1 = PerceptronCWS(dataset)
    print('data loaded!')
    rs = open('../results/perceptron-{}.txt'.format(dataset), 'w', encoding='utf8')
    with open('../datas/{}_test.txt'.format(dataset), encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            print(model1.cut(line), file=rs)

def special_test():
    model = PerceptronCWS('pku')
    rs = open('../results/perceptron-pku2weibo.txt', 'w', encoding='utf8')
    with open('../datas/weibo_test.txt', encoding='utf8') as f:
        for line in f.readlines():
            line = line.strip()
            print(model.cut(line), file=rs)

def main():
    # datas = ['pku', 'msr', 'weibo']
    # for dataset in datas:
    #     test(dataset)
    test('weibo')
    special_test()

if __name__ == '__main__':
    main()