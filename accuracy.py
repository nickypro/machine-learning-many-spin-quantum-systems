from NN import Model
import matplotlib.pyplot as plt
import copy

data = []
def dataprint( data ):
    print( "[" )
    for d in data:
        print( d )
        print( ", " )
    print( "]" )

params = {
    "logdir"      : "./log/n",
    "figdir"      : "./fig/n",
    "n_dim"       : 1,
    "L"           : 80,
    "lattice"     : "hypercube",
    "pbc"         : True,
    "scaling"     : 100,
    "hamiltonian" : "heisenberg",
    #"h"           : 1.0,
    #"j1"          : 1.00,
    #"j2"          : 0.00,
    #"lambda"      : 1,
    "model_type"  : "RBM",
    "alpha"       : 1,
    "n_iter"      : 3000,
    "iter_to_avg" : 50,
    "diag_shift"  : 0.1,
    "n_samples"   : 1008,
    "learning_rate" : 0.01,
    "verbose"     : True,
#   "observables" : [],
}


"""
params["n_iter"] = 1000
for alpha in [ 1, 2, 4 ]:
    for i in range(1):
        params["earlystopping"] = {"min_delta": 0.001, "patience": 100}
        params["model_type"] = "RBM"
        params["alpha"] = alpha
        params["n_samples"] = 5008
        params["L"] = 20
        params["scaling"] = 1000
        model = Model( params )
        results = model.run( learning_rate = {"Phase": [0.005, 0.0001]}, name="/SymmRBM-alpha"+str(alpha) )
        data += [ results ]
        print( data )
#"""

"""
for alpha in [ 4 ]:
    for i in range(3):
        params["model_type"] = "RBM"
        params["alpha"] = alpha
        params["n_iter"] = 2000
        params["n_samples"] = 5008
        params["earlystopping"] = {"min_delta": 0.001, "patience": 200}
        model = Model( params )
        name = lambda params : "/RBM-alpha" + str(params["alpha"]) + "-scaling" + str(params["scaling"]) + "-"
        results = model.run( learning_rate = {"Phase": [0.1, 0.001]}, name=name(params) )
        if results["out"]["E_90"] < -0.44:
            data += [ results ]
            print( data )
#"""

"""
for i in range(2):
    params["model_type"] = "Jastrow"
    model = Model( params )
    results = model.run( learning_rate = { "Phase": [0.1, 0.001]} )
    data += [ results ]
    print( data )
#"""

"""
for alpha in [ 1, 2, 4 ]:
    for L in [ 80 ]:
        didWork = 0
        while didWork < 3:
            params["model_type"] = "SymmRBM"
            name = "/SymmRBM" + "-alpha-" + str(alpha)
            params["n_samples"] = 1008
            params["earlystopping"] = {"min_delta": 0.1, "patience": 200}
            params["alpha"] = alpha
            params["L"] = L
            model = Model( params )
            results = model.run( learning_rate={"Phase":[0.01, 0.0001]}, name=name )
            if results["out"]["E_mean"] < -0.442:
                data += [ results ]
                didWork += 1
            dataprint( data )
#"""

"""
for alpha in [ 64, 128 ]:
    params["alpha"] = alpha
    params["model_type"] = "SymmNN"
    name = lambda params : "/SymmNN-alpha-" + str(params["alpha"]) + "-"
    for i in range(10):
        model = Model( params )
        results = model.run( learning_rate = {"Phase": [0.1, 0.000001], "Modulus": [ 0, 0.01 ]}, name=name(params) )
        data += [ results ]
        print( data )
#"""

"""
for alpha in [ 1, 2, 4 ]:
    params["alpha"] = alpha
    params["model_type"] = "SymmRBM"
    for i in range(3):
        model = Model( params )
        results = model.run( learning_rate = {"Phase": [0.1, 0.00001]}, name="/SymmRBM-alpha-"+str(alpha)+"-" )
        data += [ results ]
        print( data )

"""
"""
name = lambda params : "/GCNN" + str(params["alpha"])
params["model_type"] = "GCNN"
for i in range( 5 ):
    params["earlystopping"] = {"min_delta": 0.1, "patience": 200}
    model = Model( params )
    results = model.run( learning_rate = {"Phase": [0.1, 0.00001]}, name=name(params), feature_dims=(4,8,16) )
    data += [ results ]
    print( data )
#"""

"""
for alpha in [ 1, 2, 4, 8, 16, 32, 64, 128 ]:
    for i in range(3):
        params.update({"alpha": alpha})
        model = Model( params )
        out = model.runSymmRBM( learning_rate = {"Phase": [0.01, 0.0001]}, name="SymmRBM-alpha"+str(alpha) )
        addData( data["SymmRBM"], params, out, True )
"""

# out = model.runJastrow( plot="/Jastrow100" )
# out = model.runCNN( ( 4, 8, 16, 16 ), learning_rate={"Phase": [0.1, 0.001]}, name="/CNN" )
# addData( data["Jastrow"], out, True )


print( "Finished" )
