from NN import Model
import matplotlib.pyplot as plt
import copy
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

data = []

params = {
    "logdir"      : "./log/pyro",
    "figdir"      : "./fig/pyro",
    "n_dim"       : 3,
    "L"           : 2,
    "lattice"     : "pyrochlore",
    "pbc"         : True,
    "scaling"     : 100,
    "hamiltonian" : "heisenberg",
    #"h"           : 1.0,
    #"j1"          : 1.00,
    #"j2"          : 0.00,
    #"lambda"      : 1,
    "model_type"  : "Jastrow",
    "alpha"       : 1,
    "n_iter"      : 5000,
    "iter_to_avg" : 200,
    "diag_shift"  : 0.01,
    "n_samples"   : 5120,
    "learning_rate" : 0.01,
    "verbose"     : True,
#   "observables" : [],
}

#"""
for alpha in [ 128 ]:
    for i in range(3):
        params["model_type"] = "SymmRBM"
        params["n_samples"] = 5120
        params["alpha"] = alpha
        params["earlystopping"] = {"min_delta": 0.0001, "patience": 100}
        model = Model( params )
        results = model.run( learning_rate = { "Phase": [0.01, 0.00001]} )
        data += [ results ]
        print( data )
#"""

"""
params["n_iter"] = 1000
for alpha in [ 1, 2, 4, 8 ]:
    for i in range(3):
        params["model_type"] = "SymmRBM"
        params["alpha"] = alpha
        model = Model( params )
        results = model.run( learning_rate = {"Phase": [0.01, 0.001]}, name="/SymmRBM-alpha"+str(alpha) )
        data += [ results ]
        print( data )
#"""
