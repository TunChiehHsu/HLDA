import numpy as np
from scipy.special import gammaln
from math import exp
from collections import Counter
from .crp import CRP_prior


def most_common(x):
    return Counter(x).most_common(1)[0][0]


def post(w_m, c_p):
    c_m_nume = (w_m * c_p)
    c_m_deno = c_m_nume.sum(axis=1)
    c_m = c_m_nume/c_m_deno[:, np.newaxis]
    return np.array(c_m)


def likelihood(corpus_s, topic, eta):
    w_m = np.empty((len(corpus_s), len(topic)))
    allword_topic = [word for t in topic for word in t]
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

            prev_numerator = gammaln(len(current_topic) - len(overlap) + n_vocab * 1)
            later_dominator = gammaln(len(current_topic) + n_vocab * 1)
            for word in corpus:
                corpus_list = corpus
                if current_topic.count(word) - corpus_list.count(word) < 0:
                    a = 0
                else:
                    a = current_topic.count(word) - corpus_list.count(word)

                prev_dominator += gammaln(a + 1)
                later_numerator += gammaln(current_topic.count(word) + 1)

            prev = prev_numerator - prev_dominator
            later = later_numerator - later_dominator

            like = prev + later
            w_m[i, j] = exp(like)
    return w_m


def wn(c_m, corpus_s, topic):
    wn_ass = []
    wn_topic = [[] for _ in range(len(topic))]
    for i, corpus in enumerate(corpus_s):
        for word in corpus:
            theta = np.random.multinomial(1, c_m[i]).argmax()
            wn_ass.append(theta)
            wn_topic[theta].append(word)
    return np.array(wn_ass), wn_topic


def gibbs(corpus_s, topic, alpha, beta, phi, eta, ite):
    n_vocab = sum([len(x) for x in corpus_s])
    gibbs = np.empty((n_vocab, ite)).astype('int')

    for i in range(ite):
        z_topic, z_doc = Z(corpus_s, topic, alpha, beta)
        c_p = CRP_prior(corpus_s, z_doc, phi)
        w_m = likelihood(corpus_s, z_topic, eta)
        c_m = post(w_m, c_p)
        gibbs[:, i], w_topic = wn(c_m, corpus_s, z_topic)
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


def Z(corpus_s, topic, alpha, beta):
    '''Z distributes each vocabulary to topics'''
    '''Return a n * 1 vector, where n is the number of vocabularies'''
    n_vocab = sum([len(x) for x in corpus_s])
    # zm: n * 1
    # return the assignment of each vocabulary
    t_zm = np.zeros(n_vocab).astype('int')
    # z_assigned: j * 1
    # return a list of list topic where stores assigned vocabularies in each sublist
    z_assigned = [[] for _ in topic]
    z_doc = [[] for _ in topic]
    z_tmp = np.zeros((n_vocab, len(topic)))
    assigned = np.zeros((len(corpus_s), len(topic)))
    n = 0
    for i in range(len(corpus_s)):
        for d in range(len(corpus_s[i])):
            wi = corpus_s[i][d]
            for j in range(len(topic)):
                lik = (z_assigned[j].count(wi) + beta) / (assigned[i, j] + n_vocab * beta)
                pri = (len(z_assigned[j]) + alpha) / ((len(corpus_s[i]) - 1) + len(topic) * alpha)
                z_tmp[n, j] = lik * pri
                t_zm[n] = np.random.multinomial(1, (z_tmp[n,:] / sum(z_tmp[n,:]))).argmax()
            z_assigned[t_zm[n]].append(wi)
            z_doc[t_zm[n]].append(i)
            assigned[i, t_zm[n]] += 1
            n += 1
    z_assigned = [x for x in z_assigned if x != []]
    z_doc = [x for x in z_doc if x != []]
    return z_assigned, z_doc
