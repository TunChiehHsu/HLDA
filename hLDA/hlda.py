import numpy as np
import pydot
from .crp import CRP
from .util import gibbs


def node_sampling(corpus_s, phi):
    '''Node sampling samples the number of topics for next level'''
    topic = []
    for corpus in corpus_s:
        for word in corpus:
            cm = CRP(topic, phi)
            theta = np.random.multinomial(1, (cm/sum(cm))).argmax()
            if theta == 0:
                # create new topic
                topic.append([word])
            else:
                # existing topic
                topic[theta-1].append(word)
    return topic


def hLDA(corpus_s, alpha, beta, phi, eta, ite, level):
    topic = node_sampling(corpus_s, phi)

    hLDA_tree = [[] for _ in range(level)]
    tmp_tree = []
    node = [[] for _ in range(level+1)]
    node[0].append(1)

    for i in range(level):
        if i == 0:
            wn_topic = gibbs(corpus_s, topic, alpha, beta, phi, eta, ite)
            topic = set([x for list in wn_topic[0] for x in list])
            hLDA_tree[0].append(topic)
            tmp_tree.append(wn_topic[1:])
            tmp_tree = tmp_tree[0]
            node[1].append(len(wn_topic[1:]))
        else:
            for j in range(sum(node[i])):
                if tmp_tree == []:
                    break
                wn_topic = gibbs(tmp_tree[0], topic, alpha, beta, phi, eta, ite)
                topic = set([x for list in wn_topic[0] for x in list])
                hLDA_tree[i].append(topic)
                tmp_tree.remove(tmp_tree[0])
                if wn_topic[1:] != []:
                    tmp_tree.extend(wn_topic[1:])
                node[i+1].append(len(wn_topic[1:]))

    return hLDA_tree, node[:level]


def draw_graph(hLDA_object, length=8):
    words = hLDA_object[0]
    struc = hLDA_object[1]

    graph = pydot.Dot(graph_type='graph')
    end_index = [np.insert(np.cumsum(i), 0, 0) for i in struc]
    for level in range(len(struc)-1):

        leaf_level = level + 1
        leaf_word = words[leaf_level]
        leaf_struc = struc[leaf_level]
        word = words[level]
        end_leaf_index = end_index[leaf_level]

        for len_root in range(len(word)):
            # print(list(word[len_root]))
            root_word = '\n'.join(str(v) for v in list(word[len_root])[:length])
            leaf_index = leaf_struc[len_root]
            start = end_leaf_index[len_root]
            end = end_leaf_index[len_root+1]
            lf = leaf_word[start:end]
            for l in lf:
                # print(list(l))
                leaf_w = '\n'.join(str(v) for v in list(l)[:length])
                edge = pydot.Edge(root_word, leaf_w)
                graph.add_edge(edge)
    return graph


if __name__ == '__main__':
    # Generate Corpus
    n_rows = 10
    n_cols = 10
    corpus = np.zeros((n_rows, n_cols), dtype=np.object)
    word_count = 0
    for i in range(n_rows):
        for j in range(n_cols):
            corpus[i, j] = 'w%s' % word_count
            word_count += 1

    graph = hLDA(corpus, 10, 0.01, 0.5, 1, 1000, 3)
    plt = Image(pdot.create_png())
    display(plt)
