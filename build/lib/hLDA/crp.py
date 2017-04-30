import numpy as np


def CRP(topic, phi):
    p_crp = np.empty(len(topic)+1)
    m = sum([len(x) for x in topic])
    p_crp[0] = phi / (phi + m)
    for i, word in enumerate(topic):
        p_crp[i+1] = len(word) / (phi + m)
    return p_crp