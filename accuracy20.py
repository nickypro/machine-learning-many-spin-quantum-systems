from NN import Model
import matplotlib.pyplot as plt
import copy

data = []

params = {
    "logdir"      : "./log/20",
    "figdir"      : "./fig/20",
    "n_dim"       : 1,
    "L"           : 20,
    "lattice"     : "hypercube",
    "pbc"         : True,
    "scaling"     : 100,
    "hamiltonian" : "heisenberg",
    #"h"           : 1.0,
    #"j1"          : 1.00,
    #"j2"          : 0.00,
    #"lambda"      : 1,
    "model_type"  : "Jastrow",
    "alpha"       : 1,
    "n_iter"      : 500,
    "iter_to_avg" : 50,
    "diag_shift"  : 0.1,
    "n_samples"   : 1008,
    "learning_rate" : 0.01,
    "verbose"     : True,
    "observables" : [],
    "exact"       : -0.44521932649382184,
}

#"""
# Jastrow Ansatz
for i in range(3):
    model = Model( params )
    results = model.run( learning_rate = {"Phase": [0.05, 0.001]}, name="/Jastrow" )
    data += [ results ]
    print( data )
#"""

"""
# RBM
for alpha in [ 0.5, 1, 2, 4 ]:
    for i in range(3):
        params["alpha"] = alpha
        params["model_type"] = "RBM"
        model = Model( params )
        results = model.run( learning_rate = {"Phase": [0.01, 0.000001]}, name="/RBM" )
        data += [ results ]
        print( data )
#"""

"""
# Symm RBM
for alpha in [ 1, 2, 4 ]:
    for i in range(3):
        params["alpha"] = alpha
        params["model_type"] = "SymmRBM"
        model = Model( params )
        results = model.run( learning_rate = {"Phase": [0.01, 0.000001]}, name="/SymmRBM" )
        data += [ results ]
        print( data )
#"""

"""
for alpha in [ 1, 2, 4 ]:
    params["alpha"] = alpha
    params["model_type"] = "SymmNN"
    name = lambda params : "/SymmNN-alpha-" + str(params["alpha"]) + "-"
    for i in range(3):
        model = Model( params )
        results = model.run( learning_rate = {"Phase": [0.1, 0.00001], "Modulus": [ 0, 0.01 ]}, name=name(params) )
        data += [ results ]
        print( data )
#"""
