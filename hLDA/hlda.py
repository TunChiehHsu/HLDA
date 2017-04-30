import numpy as np
import pydot
import pygraphviz
from collections import Counter
from .crp import CRP
from .util import gibbs

def node_sampling(corpus_s, phi):
    topic = []    
    for corpus in corpus_s:
        for word in corpus:
            cm = CRP(topic, phi)
            theta = np.random.multinomial(1, (cm/sum(cm))).argmax()
            if theta == 0:
                topic.append([word])
            else:
                topic[theta-1].append(word)
    return topic


def hLDA(corpus_s, alpha, beta, phi, eta, ite, level):
    topic = node_sampling(corpus_s, phi)
    print(len(topic))
    
    hLDA_tree = [[] for _ in range(level)]
    tmp_tree = []
    node = [[] for _ in range(level+1)]
    node[0].append(1)
    
    for i in range(level):
        if i == 0:
            wn_topic = gibbs(corpus_s, topic, alpha, beta, phi, eta, ite)
            node_topic = [x for word in wn_topic[0] for x in word]
            hLDA_tree[0].append(node_topic)
            tmp_tree.append(wn_topic[1:])
            tmp_tree = tmp_tree[0]
            node[1].append(len(wn_topic[1:]))
        else:
            for j in range(sum(node[i])):
                if tmp_tree == []:
                    break
                wn_topic = gibbs(tmp_tree[0], topic, alpha, beta, phi, eta, ite)
                node_topic = [x for word in wn_topic[0] for x in word]
                hLDA_tree[i].append(node_topic)
                tmp_tree.remove(tmp_tree[0])
                if wn_topic[1:] != []:
                    tmp_tree.extend(wn_topic[1:])
                node[i+1].append(len(wn_topic[1:]))
        
    return hLDA_tree, node[:level]


def HLDA_plot(hLDA_object, Len = 5, save = False):
    
    from IPython.display import Image, display
    def viewPydot(pdot):
        plt = Image(pdot.create_png())
        display(plt)

    words = hLDA_object[0]
    struc = hLDA_object[1]
      
    graph = pydot.Dot(graph_type='graph')
    end_index = [np.insert(np.cumsum(i),0,0) for i in struc]
    
    for level in range(len(struc)-1):
        leaf_level = level + 1
        leaf_word = words[leaf_level]
        leaf_struc = struc[leaf_level]
        word = words[level]
        end_leaf_index = end_index[leaf_level]

        for len_root in range(len(word)):
            root_word = '\n'.join([x[0] for x in Counter(word[len_root]).most_common(Len)])
            leaf_index = leaf_struc[len_root]  
            start = end_leaf_index[len_root]
            end = end_leaf_index[len_root+1]
            lf = leaf_word[start:end]  
            for l in lf:
                leaf_w = '\n'.join([x[0] for x in Counter(list(l)).most_common(Len)])
                edge = pydot.Edge(root_word, leaf_w)
                graph.add_edge(edge)
    if save == True:
        graph.write_png('graph.png')
    viewPydot(graph)

def main():
    pass


if __name__ == '__main__':
    main()
