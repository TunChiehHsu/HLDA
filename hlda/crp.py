import numpy as np


def CRP(topic, phi):
    '''CRP gives the probability of topic assignment for specific vocabulary'''
    '''Return a j * 1 vector, where j is the number of topic'''
    cm = np.empty(len(topic)+1)
    m = sum([len(x) for x in topic])
    # prob for new topic
    cm[0] = phi / (phi + m)
    for i, word in enumerate(topic):
        # prob for existing topics
        cm[i+1] = len(word) / (phi + m)
    return cm


def CRP_prior(corpus_s, doc, phi):
    cp = np.empty((len(corpus_s), len(doc)))
    for i, corpus in enumerate(corpus_s):
        p_topic = [[x for x in doc[j] if x != i] for j in range(len(doc))]
        tmp = CRP(p_topic, phi)
        cp[i,:] = tmp[1:]
    return cp
