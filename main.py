from __future__ import division
import sys
import numpy as np
import random
from itertools import chain, izip
import scipy.optimize
from math import log, fabs
import scipy.spatial.distance as distance
import math
import scipy.io
import multiprocessing
from functools import partial
import os
import time
from optparse import OptionParser
from settings import *
from util import *

parser = OptionParser()
parser.add_option('-d', '--dirichlet', action="store_true",
                  dest="dirichlet", help="use dirichlet prior")
parser.add_option('-r', '--rtop', action="store_true",
                  dest="rtop", help="use top r thresholding")
(options, args) = parser.parse_args()
USE_DIRICHLET = options.dirichlet  # True
USE_TOPR = options.rtop

if len(args) < 3:
    print "Args: [infer|real_experiment] (topic-word-matrix-file) (inverse-file) {optional doc-word-matrix}"
    sys.exit(1)

# A has column for each topic, row for each word
A_file = args[1]
B_file = args[2]
A = np.loadtxt(A_file)
K = A.shape[1]
V = A.shape[0]
alpha = r/K
# Renormalize rows of A
# (Not really necessary.)
Ap = A.copy()
for i in range(V):
    Ap[i] = Ap[i] / sum(Ap[i])

# Need to add a way to disable this if it doesn't exist yet
#B = np.loadtxt('Bmatrix.txt.nips.nobias')
B = np.loadtxt(B_file)
B_norm = max(map(fabs, np.ndarray.flatten(B)))


# ReLu threshold
def threshold(x):
    if x < tau:
        return 0
    else:
        return x

# y is the document as bag-of-words, x is the hidden distribution on topics

# ignores the support parameter
def log_likelihood(y, x, alpha):
    # print x
    result = 0
    # log p(y | x)
    for i in range(len(y)):
        # if A[i].dot(x) == 0:
        #    return -float('inf')
        # print A[i], x
        if y[i] > 0: 
            result += y[i] * log(Ap[i].dot(x))
    # log p(x) for dirichlet, up to constant
    if USE_DIRICHLET:
        beta = 1 - alpha
        for i in range(len(x)):
            result += - beta * log(x[i])
    return np.array(result)

def fast_grad_log_likelihood(
        sparse_y, x, alpha, support_V, use_dirichlet_prior_gradient):
    result = np.zeros(K)
    prod = Ap.dot(x)
    for (i, y_i) in sparse_y:
        factor = y_i / prod[i]
        result += factor * Ap[i, :]
    if use_dirichlet_prior_gradient: 
        beta = 1 - alpha
        result += -beta / x
    return result * support_V

# Standard algorithm to project onto probability simplex. See e.g.
# http://ttic.uchicago.edu/~wwang5/papers/SimplexProj.pdf
def project(x):
    u = sorted(x, reverse=True)
    rho_list = [j for j in range(len(u))
                if u[j] + (1 - sum(u[:(j + 1)])) / (j + 1) > 0]
    rho = rho_list[-1] 
    lmbda = (1 - sum(u[:(rho + 1)])) / (rho + 1)
    result = np.zeros(K)
    for i in range(len(x)):
        result[i] = max(x[i] + lmbda, 1 / (2 * V)) 
    return result


def l1_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += fabs(x[i] - y[i])
    return sum


def projected_ascent(y, x0, alpha, support, use_dirichlet_prior_gradient):
    step_size = base_step_size / sum(y)
    x = project(x0)
    # unit_vector in direction of simplex on support space
    unit_vector = np.array([1 if i in support else 0
                            for i in range(len(x0))
                            ])
    unit_vector = unit_vector / np.linalg.norm(unit_vector)
    sparse_y = [(i, y[i]) for i in range(len(y)) if y[i] > 0]
    support_V = np.zeros(K)
    for j in support:
        support_V[j] = 1
    for i in range(iters):
        grad = fast_grad_log_likelihood(
            sparse_y, x, alpha, support_V, use_dirichlet_prior_gradient)
        x += step_size * grad  
        x = project(x)
    return x


def top(n, a):
    return sorted(range(len(a)), key=lambda i: a[i])[-n:]

def test_inference(D, doc_id):
    pid = 1
    #pid = multiprocessing.current_process()._identity[0]
    np.random.seed(doc_id * pid)
    sum_err_l1 = 0
    sum_err_lmax = 0

    topics = gen_topic_vector(r, K, USE_DIRICHLET, USE_SUPERFLAT_PRIOR)

    def l1_and_lmax_err(guess):
        return (distance.cityblock(topics, guess),
                distance.chebyshev(topics, guess))

    word_dist = np.dot(A, topics)
    words = np.random.multinomial(D, word_dist)
    # given words, best estimate for word_dist is just words,
    words_estimate = words / sum(words)
    print "WORDS ESTIMATE ERROR: ", np.linalg.norm(words_estimate - word_dist)
    print "L2 NORM OF WORD ESTIMATE: ", np.linalg.norm(words_estimate)
    topics_recovered = B.dot(words_estimate)
    if USE_TOPR:  # not USE_DIRICHLET:
        support = top(r, topics_recovered)
    else:
        tau = 2 * B_norm * (math.sqrt(math.log(K) / D)) / 4.5
        support = [i for i in range(len(topics_recovered))
                   if topics_recovered[i] > tau]
    print "ESTIMATED SUPPORT SIZE: ", len(support)

    err = distance.cityblock(topics, topics_recovered)
    print 'ERR PRE PROJECT', err

    print 'SUM PRE PROJECT', sum(topics_recovered)
    for i in range(len(topics_recovered)):
        if i not in support:
            topics_recovered[i] = 0

    print 'SUM POST PROJECT', sum(topics_recovered)
    topics_recovered_unnormalized = topics_recovered
    if DO_NORMALIZATION and topics_recovered.sum() > 0:
        topics_recovered = topics_recovered / topics_recovered.sum()
    print 'ERR', l1_and_lmax_err(topics_recovered)
    true_support = [i for i in range(len(topics)) if topics[i] > 0]
    print 'CORRECT SUPPORT', true_support
    print 'SUPPORT ', support
    print 'TOP5 TRUE', top(5,topics)
    print 'TOP5 RECOVERY', sorted(top(5,topics_recovered))

    # compute maximum likelihood estimate
    print "TO MAX LIKELIHOOD"
    unrestricted_support = range(len(topics_recovered))
    ascended_restricted = projected_ascent(words, topics_recovered, alpha,
                                           support, False)
    if USE_TLI_ESTIMATE:
        ascended_dirichlet = projected_ascent(words, topics_recovered, alpha,
                                              unrestricted_support, True)
    else:
        ascended_dirichlet = projected_ascent(words, np.ones(K) / K, alpha,
                                                  unrestricted_support, True)
    if True:
        ascended = ascended_dirichlet
        interesting = [i for i in range(len(topics_recovered)) if topics_recovered[
            i] > 0.06 or topics[i] > 0.01]
        print "SUMMARY VECTORS"
        print "OLD DIST: ", [topics_recovered[i] for i in interesting]
        print "NEW DIST: ", [ascended[i] for i in interesting]
        print "TRUE DIST: ", [topics[i] for i in interesting]
    return l1_and_lmax_err(topics_recovered_unnormalized) + l1_and_lmax_err(topics_recovered) + \
        l1_and_lmax_err(ascended_restricted) + \
        l1_and_lmax_err(ascended_dirichlet)


def estimate_support(words, size=5):
    words_estimate = words / sum(words)
    topics_recovered = B.dot(words_estimate)
    support = top(size, topics_recovered)
    return support

if __name__ == "__main__":
    if args[0] == "infer":
        results_err = []
        p = multiprocessing.Pool(PROC)
        for D in test_D:
            print max(map(fabs, np.ndarray.flatten(B.dot(A) - np.identity(B.shape[0]))))
            sum_err = None
            errs = map(partial(test_inference, D), range(DOCS))
            for err_tuple in errs:
                if sum_err is not None:
                    sum_err += np.array(err_tuple)
                else:
                    sum_err = np.array(err_tuple)

            results_err.append(
                str(D) + "\t" + "\t".join(map(str, sum_err / DOCS)))
        results = open("RESULTS." +
                       ("d" if USE_DIRICHLET else "s") + "." + A_file, "w")
        for line in results_err:
            print >>results, line
        results.close()
    elif args[0] == "real_experiment":
        if len(args) < 4:
            print "need additional argument: M file (usually named *.trunc.mat)"
            sys.exit(1)
        M_file = args[3]
        M = scipy.io.loadmat(M_file)['M']
        total_num_docs = M.shape[1]
        sample_size = 200
        sample_list_gen = range(total_num_docs)
        random.shuffle(sample_list_gen)
        sample_list = sample_list_gen[:sample_size]
        # we run experiment, get approximate topics, and compare
        safe_mkdir("real_doc")
        safe_mkdir("real_doc_support_guess")
        heldouts = []
        for doc in sample_list:
            words = np.ndarray.flatten(M[:, doc].toarray())
            print "Doc length: ", sum(words)
            estimated_support = estimate_support(words, size=r)
            write_doc(words, "real_doc/{0}".format(doc))
            support_guess_doc = open(
                "real_doc_support_guess/{0}".format(doc), "w")
            for s in estimated_support:
                print >>support_guess_doc, s
            support_guess_doc.close()
    else:
        print A.shape
