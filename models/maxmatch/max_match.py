#coding=utf8

import os
from collections import defaultdict
import pickle



class MaxMatch(object):
    def __init__(self, dataset, reverse=False):
        data_set_pku = '{}_train'.format(dataset)
        pku = data_set_pku+'-reverse.dict' if reverse else data_set_pku+'.dict'
        self.dict_file = os.path.join('.', 'model', pku)

        self.dataset = dataset
        self.dataset_path = os.path.join('..', 'datas', dataset+'.txt')
        self.reverse = reverse
        self.dict, self.max_len = self.build_dict(dataset)

    
    def build_dict(self, dataset):
        if not os.path.exists(self.dict_file):
            dic = set()
            max_len = 0
            with open(self.dataset_path, encoding='utf8') as f:
                for line in f.readlines():
                    line = line.strip()
                    wl = line.split()
                    for w in wl:
                        if w not in dic:
                            dic.add(w)
                            max_len = max(max_len, len(w))
            with open(self.dict_file, 'wb') as f:
                pickle.dump((dic, max_len), f)
        with open(self.dict_file, 'rb') as f:
            tmp = pickle.load(f)
            dic, max_len = tmp
            return dic, max_len
        
    def cut(self, sentence):
        now = 0
        if self.reverse:
            sentence = sentence[::-1]
        sen_len = len(sentence)
        wl = []
        while now < sen_len:
            pku = min(self.max_len, sen_len-now)
            for i in range(self.max_len, 0, -1):
                word = sentence[now: now+i]
                if word in self.dict or i == 1:
                    wl.append(word)
                    now += i
                    break
        if self.reverse:
            wl = wl[::-1]
        return wl

def special_cut():
    dataset = 'pku'
    tmp = MaxMatch(dataset, reverse=False)
    rs_f = open('../results/maxmatch-pku2weibo.txt', 'w', encoding='utf8')
    with open('../datas/weibo_test.txt', encoding='utf8') as f:
        for line in f.readlines():
            wl = tmp.cut(line.strip())
            print(" ".join(wl), file=rs_f)


def dataset_cut(dataset = 'weibo'):
    tmp = MaxMatch(dataset, reverse=False)
    rs_f = open('../results/maxmatch-{}.txt'.format(dataset), 'w', encoding='utf8')                   
    with open('../datas/{}_test.txt'.format(dataset), encoding='utf8') as f:
        for line in f.readlines():
            wl = tmp.cut(line.strip())
            print(" ".join(wl), file=rs_f)
def main():
    special_cut()


if __name__ == '__main__':
	main()