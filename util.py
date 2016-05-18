from __future__ import division
import numpy as np
import random
import os
def safe_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def write_doc(words, file_name):
    doc = open(file_name, "w")
    for i in range(len(words)):
        # rounding shouldn't realy do anything
        for j in range(int(round(words[i]))):
            print >>doc, i
    doc.close()


def write_flat(words, file_name):
    doc = open(file_name, "w")
    for i in range(len(words)):
        print >>doc, words[i]
    doc.close()

def superflat_topic(r,K):
    topics = [0 for _ in range(K)]
    for i in range(r):
        topics[i] = 1 / r
    random.shuffle(topics)
    return topics

# make sure to seed random number generator first!
def sparse_topic(r, K):
    # k-sparse prior.
    topics = [0 for _ in range(K)]
    # dirichlet prior with all alpha=1 is just uniform prior over prob. simplex
    k_sparse_part = np.random.dirichlet(np.ones(r))
    for i in range(len(k_sparse_part)):
        topics[i] = k_sparse_part[i]
    random.shuffle(topics)
    return topics

def gen_topic_vector(r, K, USE_DIRICHLET, USE_SUPERFLAT_PRIOR):
    if USE_DIRICHLET:
        alpha = r/K
        return np.random.dirichlet(np.ones(K) * alpha)
    elif USE_SUPERFLAT_PRIOR:
        return superflat_topic(r,K)
    else:
        return sparse_topic(r,K)

def gen_doc(D, A, r, K, USE_DIRICHLET, USE_SUPERFLAT_PRIOR):
    topics = gen_topic_vector(r,K, USE_DIRICHLET, USE_SUPERFLAT_PRIOR)
    word_dist = np.dot(A, topics)
    words = np.random.multinomial(D, word_dist)
    return (words, topics)


