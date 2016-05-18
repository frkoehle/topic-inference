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
(options, args) = parser.parse_args()
USE_DIRICHLET = options.dirichlet  # True
if len(args) < 1:
    print "Args: (topic-word-matrix-file)"
A_file = args[0]
A = np.loadtxt(A_file)
K = A.shape[1]
V = A.shape[0]

if __name__ == "__main__":
    print "SYNTHESIZING"
    for D in test_D:
        dir_name1 = "synth/{0}".format(D)
        safe_mkdir(dir_name1)
        dir_name2 = "synth_topics/{0}".format(D)
        safe_mkdir(dir_name2)
        if not os.path.exists(dir_name2):
            os.makedirs(dir_name2)
        for doc_id in range(DOCS):
            words, topics = gen_doc(D,A,r,K, USE_DIRICHLET, USE_SUPERFLAT_PRIOR)
            write_doc(words, "synth/{0}/{1}".format(D, doc_id))
            synth_doc_topic = open(
                "synth_topics/{0}/{1}".format(D, doc_id), "w")
            for t in topics:
                print >>synth_doc_topic, t
            synth_doc_topic.close()

