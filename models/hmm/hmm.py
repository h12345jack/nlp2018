import numpy as np
import pickle


class HiddenMarkovModel(object):
    def __init__(self, state_size, obsv_size, word2idx, tag2idx, idx2tag):
        self.state_size = state_size
        self.obsv_size = obsv_size
        self.pi_t = np.ones(state_size)
        self.trans_t = np.ones((state_size, state_size))
        self.emiss_t = np.ones((state_size, obsv_size))

        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

    # self.pi = np.zeros(state_size)
    # self.trans = np.zeros((state_size, state_size))
    # self.emiss = np.zeros((state_size, obsv_size))

    def fit(self, X, Y):
        for x, y in zip(X, Y):
            for i, _ in enumerate(x):
                self.emiss_t[y[i]][x[i]] += 1
                if i == 0:
                    self.pi_t[y[i]] += 1
                else:
                    self.trans_t[y[i - 1]][y[i]] += 1
        self.pi = self.pi_t / np.sum(self.pi_t)
        self.trans = np.array([self.trans_t[i] / np.sum(self.trans_t, 1)[i] \
                               for i, _ in enumerate(self.trans_t)])
        self.emiss = np.array([self.emiss_t[i] / np.sum(self.emiss_t, 1)[i] \
                               for i, _ in enumerate(self.emiss_t)])

        self.pi_l = np.log(self.pi)
        self.trans_l = np.log(self.trans)
        self.emiss_l = np.log(self.emiss)

    def predict(self, x):
        alpha_t = np.zeros(self.state_size)
        steps = []
        for state in range(self.state_size):
            alpha_t[state] = self.pi_l[state] + self.emiss_l[state][x[0]]
        for t in range(1, len(x)):
            tmp_t = np.full(self.state_size, -np.inf)
            step_t = []
            for i in range(self.state_size):
                step_t.append(0)
                for j in range(self.state_size):
                    if alpha_t[j] + self.trans_l[j][i] + self.emiss_l[i][x[t]] > tmp_t[i]:
                        tmp_t[i] = alpha_t[j] + self.trans_l[j][i] + self.emiss_l[i][x[t]]
                        step_t[i] = j
            alpha_t = tmp_t
            steps.append(step_t)
        p = np.max(alpha_t)
        step = np.argmax(alpha_t)
        ans = []
        ans.append(step)
        for t in range(len(x) - 2, -1, -1):
            step = steps[t][step]
            ans.append(step)
        return p, list(reversed(ans))


def load_training_data(filename='./models/hmm/data/icwb2-data/training/pku_training.utf8'):
    training_data = []
    with open(filename, encoding='utf8') as f:
        for line in f.readlines():
            sentence = list(''.join(line.split()))
            label = []
            for word in line.split():
                if len(word) == 1:
                    label += ['S']
                else:
                    tmp_label = ['M'] * len(word)
                    tmp_label[0], tmp_label[-1] = 'B', 'E'
                    label += tmp_label
            assert len(sentence) == len(label)
            # assert len(sentence) > 0
            if len(sentence) == 0:
                continue
            training_data.append((sentence, label))
    return training_data


def load_testing_data(filename='./models/hmm/data/icwb2-data/testing/pku_test.utf8'):
    testing_data = []
    with open(filename, encoding='utf8') as f:
        for line in f.readlines():
            sentence = list(''.join(line.split()))
            assert len(sentence) > 0
            testing_data.append((sentence,))
    return testing_data


def to_idx(training_data):
    word2idx = {'<OTHERS>': 0}
    tag2idx = {}
    for line in training_data:
        sentence, label = line
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        for tag in label:
            if tag not in tag2idx:
                tag2idx[tag] = len(tag2idx)
    return word2idx, tag2idx


def prepare_sequence(seq, to_idx):
    idxs = [to_idx[w] if w in to_idx else 0 for w in seq]
    return np.array(idxs, dtype=np.int)


def seg(hmm, word2idx, idx2tag, testing_data, name):
    with open('./models/hmm/seg_'+name+'.txt', 'w') as f:
        for line in testing_data:
            sentence = line[0]
            x = prepare_sequence(sentence, word2idx)
            p, steps = hmm.predict(x)
            cuts = ''
            for i, step in enumerate(steps):
                tag = idx2tag[step]
                if tag == 'B':
                    cuts += ' ' + sentence[i]
                elif tag == 'M':
                    cuts += sentence[i]
                elif tag == 'E':
                    cuts += sentence[i] + ' '
                elif tag == 'S':
                    cuts += ' ' + sentence[i] + ' '
            f.write('  '.join(cuts.split()) + '\n')


# def predict(sentence, hmm, word2idx, idx2tag):
#     x = prepare_sequence(sentence, word2idx)
#     p, steps = hmm.predict(x)
#     cuts = ''
#     for i, step in enumerate(steps):
#         tag = idx2tag[step]
#         if tag == 'B':
#             cuts += ' ' + sentence[i]
#         elif tag == 'M':
#             cuts += sentence[i]
#         elif tag == 'E':
#             cuts += sentence[i] + ' '
#         elif tag == 'S':
#             cuts += ' ' + sentence[i] + ' '
#     return ' '.join(cuts.split())


def save(model, name):
    file = open('./models/hmm/model/hmm_'+name+'.model', 'wb')
    pickle.dump(model, file)


def load(name):
    file = open('./models/hmm/model/hmm_'+name+'.model', 'rb')
    return pickle.load(file)


def main():
    training_data = load_training_data('./models/hmm/data/icwb2-data/training/pku_training.utf8')
    word2idx, tag2idx = to_idx(training_data)
    idx2tag = {}
    for key in tag2idx.keys():
        idx2tag[tag2idx[key]] = key
    print(tag2idx)
    hmm = HiddenMarkovModel(len(tag2idx), len(word2idx), word2idx, tag2idx, idx2tag)
    X = []
    Y = []
    for line in training_data:
        X.append(prepare_sequence(line[0], word2idx))
        Y.append(prepare_sequence(line[1], tag2idx))
    hmm.fit(X, Y)
    print(hmm.pi)
    print(hmm.trans)
    print(hmm.emiss)
    testing_data = load_testing_data('./models/hmm/data/icwb2-data/testing/pku_test.utf8')
    seg(hmm, word2idx, idx2tag, testing_data, 'pku')
    save(hmm, 'pku')


    training_data = load_training_data('./models/hmm/data/icwb2-data/training/msr_training.utf8')
    word2idx, tag2idx = to_idx(training_data)
    idx2tag = {}
    for key in tag2idx.keys():
        idx2tag[tag2idx[key]] = key
    print(tag2idx)
    hmm = HiddenMarkovModel(len(tag2idx), len(word2idx), word2idx, tag2idx, idx2tag)
    X = []
    Y = []
    for line in training_data:
        X.append(prepare_sequence(line[0], word2idx))
        Y.append(prepare_sequence(line[1], tag2idx))
    hmm.fit(X, Y)
    print(hmm.pi)
    print(hmm.trans)
    print(hmm.emiss)
    testing_data = load_testing_data('./models/hmm/data/icwb2-data/testing/msr_test.utf8')
    seg(hmm, word2idx, idx2tag, testing_data, 'msr')
    save(hmm, 'msr')


    training_data = load_training_data('./models/hmm/data/nlpcc2016-word-seg-train.dat.txt')
    word2idx, tag2idx = to_idx(training_data)
    idx2tag = {}
    for key in tag2idx.keys():
        idx2tag[tag2idx[key]] = key
    print(tag2idx)
    hmm = HiddenMarkovModel(len(tag2idx), len(word2idx), word2idx, tag2idx, idx2tag)
    X = []
    Y = []
    for line in training_data:
        X.append(prepare_sequence(line[0], word2idx))
        Y.append(prepare_sequence(line[1], tag2idx))
    hmm.fit(X, Y)
    print(hmm.pi)
    print(hmm.trans)
    print(hmm.emiss)
    testing_data = load_testing_data('./models/hmm/data/nlpcc2016-wordseg-dev.dat.txt')
    seg(hmm, word2idx, idx2tag, testing_data, 'weibo')
    save(hmm, 'weibo')


    training_data = load_training_data('./models/hmm/data/icwb2-data/training/pku_training.utf8')
    word2idx, tag2idx = to_idx(training_data)
    idx2tag = {}
    for key in tag2idx.keys():
        idx2tag[tag2idx[key]] = key
    print(tag2idx)
    hmm = HiddenMarkovModel(len(tag2idx), len(word2idx), word2idx, tag2idx, idx2tag)
    X = []
    Y = []
    for line in training_data:
        X.append(prepare_sequence(line[0], word2idx))
        Y.append(prepare_sequence(line[1], tag2idx))
    hmm.fit(X, Y)
    print(hmm.pi)
    print(hmm.trans)
    print(hmm.emiss)
    testing_data = load_testing_data('./models/hmm/data/nlpcc2016-wordseg-dev.dat.txt')
    seg(hmm, word2idx, idx2tag, testing_data, 'pku2weibo')
    save(hmm, 'pku2weibo')


# def cut(sentence):
#     model = load('pku')
#     return predict(sentence, model, model.word2idx, model.idx2tag)


class HMM(object):
    def __init__(self, data):
        self.model = None
        self.model = load(data)

    def cut(self, sentence):
        x = prepare_sequence(sentence, self.model.word2idx)
        p, steps = self.model.predict(x)
        cuts = ''
        for i, step in enumerate(steps):
            tag = self.model.idx2tag[step]
            if tag == 'B':
                cuts += ' ' + sentence[i]
            elif tag == 'M':
                cuts += sentence[i]
            elif tag == 'E':
                cuts += sentence[i] + ' '
            elif tag == 'S':
                cuts += ' ' + sentence[i] + ' '
        return ' '.join(cuts.split())


if __name__ == '__main__':
    main()
