# If False, use an r-uniform sparse prior
USE_SUPERFLAT_PRIOR = False
DO_ASCENT = True
DO_NORMALIZATION = True
USE_TLI_ESTIMATE = True

# document sizes to use for synthetic experiments
test_D = range(200, 2000, 200)
DOCS = 200
#DOCS = 200

# Sparsity parameter; for Dirichlet, we set alpha = r/K
r = 5
# Inversion parameter: allowed bias
delta = 0

# This setting matters mostly for MAP computation
# For experiments we used 8e-4 for NY times, 2e-3 for other datasets
base_step_size = 2e-3
iters = 1600

#LP Solver to use, either glpk or mosek. Must install cvxopt package
# with appropriate support.
LP_SOLVER = 'glpk'

# max number of processes to use
PROC = 16
