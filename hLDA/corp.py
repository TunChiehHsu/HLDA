import numpy as np
import string

def sim_corpus(n):
    n_rows = n
    corpus = [[] for _ in range(n_rows)]
    for i in range(n_rows):
        n_cols = np.random.randint(10, 200, 1, dtype = 'int')[0]
        for j in range(n_cols):
            num = np.random.normal(0, 1, n_cols)
            word = 'w%s' % int(round(num[j], 1)*10)
            corpus[i].append(word)
    return corpus

def read_corpus(document_path):
    punc = ['`', ',', "'", '.', '!', '?']
    corpus = []
    with open(document_path, 'r') as f:
        for line in f:
            for x in punc:
                line = line.replace(x, '')
            line = line.strip('\n')
            word = line.split(' ')
            corpus.append(word)
    return(corpus)

def main():
    pass


if __name__ == '__main__':
    main()