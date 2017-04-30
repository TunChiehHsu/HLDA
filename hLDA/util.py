import numpy as np
from scipy.special import gammaln
from math import exp
from collections import Counter
from .crp import CRP

def Z(corpus_s, topic, alpha, beta):
    n_vocab = sum([len(x) for x in corpus_s])
    t_zm = np.zeros(n_vocab).astype('int')
    z_topic = [[] for _ in topic]
    z_doc = [[] for _ in topic]
    z_tmp = np.zeros((n_vocab, len(topic)))
    assigned = np.zeros((len(corpus_s), len(topic)))
    n = 0
    for i in range(len(corpus_s)):
        for d in range(len(corpus_s[i])): 
            wi = corpus_s[i][d]   
            for j in range(len(topic)):
                lik = (z_topic[j].count(wi) + beta) / (assigned[i, j] + n_vocab * beta)
                pri = (len(z_topic[j]) + alpha) / ((len(corpus_s[i]) - 1) + len(topic) * alpha)
                z_tmp[n, j] = lik * pri
                t_zm[n] = np.random.multinomial(1, (z_tmp[n,:]/sum(z_tmp[n,:]))).argmax()
            z_topic[t_zm[n]].append(wi)
            z_doc[t_zm[n]].append(i)
            assigned[i, t_zm[n]] += 1
            n += 1
    z_topic = [x for x in z_topic if x != []]
    z_doc = [x for x in z_doc if x != []]
    return z_topic, z_doc

def most_common(x):
    return Counter(x).most_common(1)[0][0]

def CRP_prior(corpus_s, doc, phi):
    cp = np.empty((len(corpus_s), len(doc)))
    for i, corpus in enumerate(corpus_s):
        p_topic = [[x for x in doc[j] if x != i] for j in range(len(doc))]
        tmp = CRP(p_topic, phi)
        cp[i,:] = tmp[1:]
    return cp

def likelihood(corpus_s, topic, eta):
    w_m = np.empty((len(corpus_s), len(topic)))
    allword_topic = [word  for t in topic for word in t]
    n_vocab = sum([len(x) for x in corpus_s])
    for i, corpus in enumerate(corpus_s):
        prob_result = []
        for j in range(len(topic)):
            current_topic = topic[j]
            n_word_topic = len(current_topic)
            prev_dominator = 1
            later_numerator = 1
            prob_word = 1  

            overlap = [val for val in set(corpus) if val in current_topic]
            
            prev_numerator = gammaln(len(current_topic) - len(overlap) + n_vocab * eta)
            later_dominator = gammaln(len(current_topic) + n_vocab * eta)
            for word in corpus:                
                corpus_list = corpus                
                if current_topic.count(word) - corpus_list.count(word) < 0 :
                    a = 0
                else:
                    a = current_topic.count(word) - corpus_list.count(word)
                
                prev_dominator += gammaln(a + eta)
                later_numerator += gammaln(current_topic.count(word) + eta)
           
            prev = prev_numerator - prev_dominator
            later = later_numerator - later_dominator
            
            like = prev + later 
            w_m[i, j] = like
        w_m[i, :] = w_m[i, :] + abs(min(w_m[i, :]) + 0.1)
    w_m = w_m/w_m.sum(axis = 1)[:, np.newaxis]
    return w_m

def post(w_m, c_p):
    c_m = (w_m * c_p) / (w_m * c_p).sum(axis = 1)[:, np.newaxis]
    return np.array(c_m)

def wn(c_m, corpus_s):
    wn_ass = []
    for i, corpus in enumerate(corpus_s):
        for word in corpus:
            if c_m[i].sum != 1:
                c_m[i] = c_m[i]/c_m[i].sum()
            theta = np.random.multinomial(1, c_m[i]).argmax()
            wn_ass.append(theta)
    return np.array(wn_ass)

def gibbs(corpus_s, topic, alpha, beta, phi, eta, ite):
    n_vocab = sum([len(x) for x in corpus_s])
    gibbs = np.empty((n_vocab, ite)).astype('int')
   
    for i in range(ite):
        z_topic, z_doc = Z(corpus_s, topic, alpha, beta)
        c_p = CRP_prior(corpus_s, z_doc, phi)
        w_m = likelihood(corpus_s, z_topic, eta)
        c_m = post(w_m, c_p)
        gibbs[:, i] = wn(c_m, corpus_s) 
    # drop first 1/10 data
    gibbs = gibbs[:, int(ite/10):]
    theta = [most_common(gibbs[x]) for x in range(n_vocab)]
    
    n_topic = max(theta)+1
    
    wn_topic = [[] for _ in range(n_topic)]
    wn_doc_topic = [[] for _ in range(n_topic)]

    doc = 0
    n = 0
    for i, corpus_s in enumerate(corpus_s):
        if doc == i:
            for word in corpus_s:
                wn_doc_topic[theta[n]].append(word)
                n += 1
            for j in range(n_topic):
                if wn_doc_topic[j] != []:
                    wn_topic[j].append(wn_doc_topic[j])
        wn_doc_topic = [[] for _ in range(n_topic)]        
        doc += 1
    wn_topic = [x for x in wn_topic if x != []]
    return wn_topic



