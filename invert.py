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
import cvxopt as cvx

from settings import *


parser = OptionParser()
(options, args) = parser.parse_args()
if len(args) < 2:
    print "Need name of matrix to invert, and destination file."
A_file = args[0]
A = np.loadtxt(A_file)
K = A.shape[1]
V = A.shape[0]
dest = args[1]

def interleave(l1, l2): return list(chain.from_iterable(izip(l1, l2)))

def process_row_delta(row):
    delta = 0
    n = len(A)
    d = len(A[0])
    print('Computing row ' + str(row + 1) + ' of ' + str(d))
    G_top = interleave(np.transpose(A), -np.transpose(A))
    G_bottom = interleave(np.identity(n), -np.identity(n))
    almost_G = map(lambda x: np.append(x, [0]), G_top) + map(
        lambda x: np.append(x, [-1]), G_bottom)

    c = cvx.matrix(np.append(np.zeros(n), [1]))  # default is minimize
    G = cvx.matrix(np.array(almost_G))
    h = cvx.matrix(np.array([delta] * row * 2 + [delta + 1, delta - 1] +
                            [delta] * 2 * (d - 1 - row) + [0] * 2 * n), tc='d')
    sol = cvx.solvers.lp(c, G, h, solver=LP_SOLVER)
    vars = np.array(sol['x'])
    print vars[-1]
    return np.transpose(vars[:-1])


def new_compute_B():
    n = len(A)
    d = len(A[0])
    print n, d
    print np.linalg.matrix_rank(A)
    bs = []  # each element of bs will be a row of B
    # note for GLPK, each process needs maybe 4G RAM
    p = multiprocessing.Pool(PROC)  
    bs = p.map(process_row_delta, range(d))

    B = np.vstack(bs)

    np.savetxt(dest, B)



if __name__ == "__main__":
    new_compute_B()
