from NN import Model
import matplotlib.pyplot as plt
import copy

data = []

params = {
    "logdir"      : "./log/pyro2",
    "figdir"      : "./fig/pyro2",
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
    "model_type"  : "exact",
    "alpha"       : 1,
    "n_iter"      : 10000,
    "iter_to_avg" : 300,
    "diag_shift"  : 0.1,
    "n_samples"   : 1008,
    "learning_rate" : 0.01,
    "verbose"     : True,
    "observables" : [],
}


model = Model( params )
results = model.runExact() 
print( results )

print( "Finished" )
