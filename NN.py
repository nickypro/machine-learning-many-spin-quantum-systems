from sys import argv
import os
import netket as nk
#import netket.experimental as nkx
from netket.operator.spin import sigmax, sigmaz, sigmay, sigmap, sigmam
from scipy.sparse.linalg import eigsh
import numpy as np
import jax
import optax
import flax
import flax.linen as nn
import netket.nn as nknn
import jax.numpy as jnp
import time
import datetime
import json
import copy
from matplotlib import pyplot as plt
from matplotlib import rcParams
log=nk.logging.RuntimeLog()
rcParams['font.size'] = 16
rcParams['figure.figsize'] = 8, 6

def getDefaultParams():
    return {
        "verbose"     : True,
        "logdir"      : "./log",
        "figdir"      : "./fig",
        "n_dim"       : 1,
        "L"           : 2,
        "lattice"     : "hypercube",
        "pbc"         : True,
        "scaling"     : 1.0,
        "hamiltonian" : "heisenberg",
        "h"           : 1.0,  # ising
        "j1"          : 1.00, # default nearest neighbor
        "j2"          : 0.00, # default next-nearest neighbor
        "lambda"      : 1, # default skew-symmetry
        "model_type"  : "RBM",
        "alpha"       : 1,
        "n_iter"      : 300,
        "iter_to_avg" : 50,
        "diag_shift"  : 0.1,
        "n_samples"   : 1008,
        "learning_rate" : 0.01,
        "observables" : set([]), # set(["Sx"])
        "obs"         : {},
        "exact"       : None,
        "earlystopping": None,
    }


def defaultParamDiff(  params ):
    default = getDefaultParams()
    diff = {}
    for key, value in params.items():
        if value == default[key]:
            continue
        diff[key] = copy.deepcopy( value )

    return diff


def getSecondEdges( edges ):
    first_edges = {}
    second_edges = set([])
    for (i, j) in edges:
        [ a, b ] = sorted([ i, j ])
        if not first_edges.get( a ):
            first_edges[ a ] = set([])
        first_edges[ a ].add( b )

    for key in first_edges:
        for first_edge in first_edges[ key ]:
            if not first_edges.get( first_edge ):
                continue
            for second_edge in first_edges[ first_edge ]:
                second_edges.add( ( key, second_edge ) )

    return second_edges

class Model():

    def __init__( self, params={} ):
        # constants
        print( "initialising..." )
        self.obs = {}
        self.diff = {}
        self.exact = None
        self.updateParams( getDefaultParams() )
       
        # non-default parameters
        self.updateParams( params )

        if params.get("verbose"):
            print( params )

        # We define an hilbert space for N spins with total magnetization fixed at 0.
        # Pyrochlore Lattice:
        if self.lattice == "pyrochlore":
            basis = np.array([ [0, 2, 2], [2, 0, 2], [2, 2, 0] ])
            ( a, b, c ) = ( basis[0]/2, basis[1]/2, basis[2]/2 )
            cell = np.array([ 0*a, a, b, c ])
            extent = [ self.L for _ in range(self.n_dim) ]
            self.graph = nk.graph.Lattice( basis_vectors=basis, site_offsets=cell, extent=extent, pbc=self.pbc)
        
        # Hypercube Lattice
        if self.lattice == "hypercube":
            self.graph = nk.graph.Hypercube(length=self.L, n_dim=self.n_dim, pbc=self.pbc)
       
        self.N = self.graph.n_nodes
        print( "Total number of nodes: ", self.N )
        self.hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=self.graph.n_nodes)

        # get Observables
        self.getObservables()

        # get Hamiltonian
        self.defineHamiltonian()

        # set magnetization to zero
        self.v = self.hi.random_state( jax.random.PRNGKey(0), (2,) )
        print( "The total magnetization for those two states are: ", self.v.sum(axis=-1) )

    def updateParams( self, params ):
        print( "updating params..." )
        self.logdir     = params[ "logdir" ] if "logdir" in params else self.logdir
        self.figdir     = params[ "figdir" ] if "figdir" in params else self.figdir
        self.n_dim      = params[ "n_dim" ] if "n_dim" in params else self.n_dim
        self.L          = params[ "L" ] if "L" in params else self.L
        self.lattice    = params[ "lattice" ] if "lattice" in params else self.lattice
        self.pbc        = params[ "pbc" ] if "pbc" in params else self.pbc
        self.scaling    = params[ "scaling" ] if "scaling" in params else self.scaling
        self.hamiltonian= params[ "hamiltonian" ] if "hamiltonian" in params else self.hamiltonian
        self.h          = params[ "h" ] if "h" in params else self.h
        self.j1         = params[ "j1" ] if "j1" in params else self.j1
        self.j2         = params[ "j2" ] if "j2" in params else self.j2
        self._lambda    = params[ "lambda" ] if "lambda" in params else self._lambda
        self.model_type = params[ "model_type" ] if "model_type" in params else self.model_type
        self.alpha      = params[ "alpha" ] if "alpha" in params else self.alpha
        self.n_iter     = params[ "n_iter" ] if "n_iter" in params else self.n_iter
        self.iter_to_avg    = params[ "iter_to_avg" ] if "iter_to_avg" in params else self.iter_to_avg
        self.diag_shift     = params[ "diag_shift" ] if "diag_shift" in params else self.diag_shift
        self.n_samples      = params[ "n_samples" ] if "n_samples" in params else self.n_samples
        self.learning_rate  = params[ "learning_rate" ] if "learning_rate" in params else self.learning_rate
        self.observables    = params[ "observables" ] if "observables" in params else self.observables
        self.exact          = params[ "exact" ] if "exact" in params else self.exact
        self.earlystopping  = params[ "earlystopping" ] if "earlystopping" in params else self.earlystopping

        # ensure directories exits:
        os.makedirs( self.logdir, exist_ok=True)
        os.makedirs( self.figdir, exist_ok=True)

        default = getDefaultParams()
        for key, value in params.items():
            if value == default[ key ]:
                if key in self.diff:
                    del self.diff[ key ]
                continue
            self.diff[ key ] = value

    def defineHamiltonian( self ):
        # scale to get the answer
        const = self.scaling / ( 4*self.graph.n_nodes )

        # Ising Hamiltonian
        if self.hamiltonian == "ising":
            print("# Using Ising Model")
            self.H = const * nk.operator.Ising(hilbert=self.hi, graph=self.graph, h=self.h)
            return self.H

        # Heisenberg Hamiltonian
        if self.hamiltonian == "heisenberg":
            print("# Using Heisenberg Model")
            self.H = const * nk.operator.Heisenberg( hilbert=self.hi, graph=self.graph )
            return self.H

        if self.hamiltonian == "skew":
            print("# Using Default Broken-Symmetry Heisenberg Model")
            # Default to General Skew-Heisenberg Hamiltonian
            def exchange(i, j):
                hi = self.hi
                return (
                      sigmax(hi, i) * sigmax(hi, j)
                    + sigmay(hi, i) * sigmay(hi, j)
                    + sigmaz(hi, i) * sigmaz(hi, j)
                )

            def hlambda( i, j ):
                hi = self.hi
                return (
                    0.25 * self._lambda * (
                        sigmap( hi, i ) * sigmam( hi, j )
                      + sigmap( hi, j ) * sigmam( hi, i )
                    )
                      + sigmaz( hi, i ) * sigmaz( hi, j )
                )

            self.first_edges = self.graph.edges()
            self.second_edges = getSecondEdges( self.first_edges )

            # define hamilotonian
            self.H = self.scaling * (
                ( self.j1 * sum([ exchange(i, j) for (i, j) in self.first_edges ]) \
              + ( self.j2 * sum([ hlambda(i, j) for (i, j) in self.second_edges ]) if self.j2 != 0 else 0 ) )/( 4 * self.graph.n_nodes )
            )
            return self.H
        
        Exception("hamiltnian not found:" + self.hamiltonian)

    def getObservables( self ):
        # create an observable
        self.obs = {}
        for obs in self.observables:
            if obs == "Sx":
                self.obs["Sx"] = sum([ nk.operator.spin.sigmax(self.hi, i) for i in range(self.L) ])
        return self.obs

    def getFilename( self ):
        return  "./figs2/L=%d__j2=%.2f__lambda=%1.2f__Its=%d__LR=%0.2f-%.3f__alpha=%d.png" % (
            self.L, self.j2, self._lambda, self.n_iter, self.phase_init, self.phase_final, self.alpha
        )

    def getOptimizer( self, learning_rate, simple=False ):
        # Define simple optimizer 
        if not learning_rate:
            return nk.optimizer.Sgd( learning_rate=self.learning_rate )

        if  type( learning_rate ) is float:
            return nk.optimizer.Sgd( learning_rate=learning_rate )

        [ phase_init,   phase_final   ] = learning_rate[ "Phase" ]
        [ modulus_init, modulus_final ] = learning_rate.get( "Modulus", [ 0, 0.01 ] )
        self.phase_init  = phase_init
        self.phase_final = phase_final

        # Define more advanced optimizer
        # A linear schedule varies the learning rate from 0 to 0.01 across 600 steps.
        modulus_schedule = optax.linear_schedule( modulus_init, modulus_final, self.n_iter )

        # The phase starts with a larger learning rate and then is decreased.
        phase_schedule = optax.linear_schedule( phase_init, phase_final, self.n_iter )

        # Combine the linear schedule with SGD
        optm = optax.sgd( modulus_schedule )
        optp = optax.sgd( phase_schedule   )

        if simple:
            return optp

        # The multi-transform optimizer uses different optimisers for different parts of the
        # parameters.
        optimizer = optax.multi_transform(
            {'o1': optm, 'o2': optp},
            flax.core.freeze({"Modulus":"o1", "Phase":"o2"})
        )

        return optimizer

    def runExact( self ):
        # compute exact energy
        print("### Exact Diagonalization...")
        start = time.time()
        sp_h = self.H.to_sparse().real
        eig_vals = np.sort( eigsh(sp_h, k=4, which="SA", return_eigenvectors=False, tol=1.0e-8) )
        end = time.time()
        
        print("Exact eigenvalues with scipy sparse:", eig_vals)
        self.exact_gs_energy = eig_vals[ 0 ]

        out = { 
            "dur": end-start,
            "E_mean": self.exact_gs_energy/self.scaling,
            "E_90": self.exact_gs_energy/self.scaling,
            "E_min": self.exact_gs_energy/self.scaling,
        }
        self.exact = self.exact_gs_energy
        
        results = { "in": self.diff, "out": out }
        return results

    def makeGS( self, model, learning_rate, simple=False ):
        self.diff[ "learning_rate" ] = learning_rate
        sampler = nk.sampler.MetropolisExchange(self.hi, graph=self.graph)

        self.vstate = nk.vqs.MCState( sampler, model, n_samples=self.n_samples )

        optimizer = self.getOptimizer( learning_rate, simple )

        gs = nk.VMC(
            hamiltonian = self.H,
            optimizer = optimizer,
            variational_state = self.vstate,
            preconditioner = nk.optimizer.SR( diag_shift=self.diag_shift )
        )
        return gs

    def runGS( self, gs, log ):
        print('Model has', nk.jax.tree_size(self.vstate.parameters), 'parameters')
        callback = None
        if self.earlystopping:
            callback = nk.callbacks.EarlyStopping(
                    min_delta=self.earlystopping["min_delta"],
                    patience=self.earlystopping["patience"] )
        
        now = datetime.datetime.now()
        start = time.time()
        name = log + str( now )
        logFileName = self.logdir + name
        gs.run( self.n_iter, out=logFileName, obs=self.obs, callback=callback )
        end = time.time()

        dur = end-start
        logFile = json.load( open(logFileName+".log") )
        
        out = self.doStats( logFile, name )
        out["dur"] = dur
        out["logFile"] = logFileName+".log"

        print( out )
        results = { "in": self.diff, "out": out }

        return results, logFile

    def doStats( self, data, graphName=False, makeNewFigures=True ):
        out = {}
        Energy = data["Energy"]
        iters = self.iter_to_avg
        
        E = Energy["Mean"]
        if type(E) is dict:
            E = E["real"]
        E = np.array( E ) * ( 1/self.scaling )
        N = [ i for i in range(len(E)) ]
        Errors = np.array( Energy["Sigma"] ) * ( 1/self.scaling )

        # "last few" energies found
        Err_last = Errors[ -iters: ]
        N_last = N[ -iters: ]
        E_last   = np.array( E[ -iters: ] )
        
        # Find the mean and std of the last few energies found
        out["E_mean"]     = np.mean( E_last )
        out["E_mean_std"] = np.mean( Err_last )

        ps = np.argsort( E_last )

        p100 = ps[0]
        out["E_min"]     = E_last[ p100 ]
        out["E_min_std"] = Err_last[ p100 ]
        
        p95  = ps[ iters//20 ]
        out["E_95"]      = E_last[ p95 ]
        out["E_95_std"]  = Err_last[ p95 ]
        
        p90  = ps[ iters//10 ]
        out["E_90"]      = E_last[ p90 ]
        out["E_90_std"]  = Err_last[ p90 ]

        p75  = ps[ iters//10 ]
        out["E_75"]      = E_last[ p75 ]
        out["E_75_std"]  = Err_last[ p75 ]

        def initFig( N, E, Err ):
            if makeNewFigures:
                plt.figure()
            plt.ylabel("Energy")
            plt.xlabel("Number of Iterations")
            plt.errorbar( N, E, yerr=Err )

        

        toPlot = [(N, E, Errors, "full")]
        if makeNewFigures:
            toPlot += [(N_last, E_last, Err_last, "final")]
        
        for ( N, E, Err, label ) in toPlot:
            initFig( N, E, Err )
            if self.exact:
                plt.axhline(y=self.exact, xmin=0, xmax=len(E), linewidth=2, color="k", label="Exact")
            if graphName:
                plt.tight_layout()
                if makeNewFigures:
                    plt.savefig( (self.figdir + graphName + label + ".png"), dpi=300  )
                    #plt.savefig( (self.figdir + graphName + ".svg")  )
            else:
                if makeNewFigures:
                    plt.show()

        return out
    
    def runRBM( self, learning_rate=None, name="/RBM" ):
        # define symm neural network
        model = nk.models.RBM( alpha=self.alpha )

        # compile the ground-state optimization loop
        gs = self.makeGS( model, learning_rate, simple=True )

        print('### RBM')
        results, self.logRBM = self.runGS( gs, name )
        
        return results
        
    def runSymmRBM( self, learning_rate=None, name="/SymmRBM" ):
        # define symm neural network
        model = nk.models.RBMSymm( symmetries=self.graph.translation_group(), alpha=self.alpha )

        # compile the ground-state optimization loop
        gs = self.makeGS( model, learning_rate, simple=True )

        print('### Symm RBM')
        results, self.logSymmRBM = self.runGS( gs, name )
        
        return results

    def runNN( self, learning_rate=None, name="/NN" ):
        alpha = self.alpha
        # define symm neural network
        class NNModel(nn.Module):
            @nn.compact
            def __call__(this, x):

                # We use a simple dense layer for modulus
                rho = nknn.Dense(
                    features=alpha*self.graph.n_nodes,
                    dtype=float,
                    kernel_init=nn.initializers.normal(stddev=0.001),
                    name="Modulus"
                )(x)
                rho = nn.relu(rho)

                # We use nknn.Dense and not nn.Dense because the latter has a bug
                # with complex number inputs
                phase = nknn.Dense(
                    features=alpha*self.graph.n_nodes,
                    dtype=float,
                    kernel_init=nn.initializers.normal(stddev=0.001),
                    name="Phase"
                )(x)
                phase = nn.relu(phase)

                return jnp.sum(rho, axis=(-1)) + 1.0j*jnp.sum(phase, axis=(-1))

        # create model with alpha=1
        model = NNModel()

        # compile the ground-state optimization loop
        gs = self.makeGS( model, learning_rate )

        print('### Simple NN')
        results, self.logNN = self.runGS( gs, name )
        
        return results

    def runSymmNN( self, learning_rate=None, name="/SymmNN" ):
        alpha = self.alpha
        # define symm neural network
        class NNModel(nn.Module):
            @nn.compact
            def __call__(this, x):

                # We use a symmetrized dense layer, and the symmetries are given
                # by the translational group on our lattice
                rho = nknn.DenseSymm(
                    symmetries=self.graph.translation_group(),
                    features=alpha,
                    dtype=float,
                    kernel_init=nn.initializers.normal(stddev=0.001),
                    name="Modulus"
                )(x)
                rho = nn.relu(rho)

                # We use nknn.Dense and not nn.Dense because the latter has a bug
                # with complex number inputs
                phase= nknn.Dense(
                    features=alpha*self.graph.n_nodes,
                    dtype=float,
                    kernel_init=nn.initializers.normal(stddev=0.001),
                    name="Phase"
                )(x)
                phase = nn.relu(phase)

                return jnp.sum(rho, axis=(-1, -2)) + 1.0j*jnp.sum(phase, axis=(-1))

        # create model with alpha=1
        model = NNModel()

        # compile the ground-state optimization loop
        gs = self.makeGS( model, learning_rate )

        print('### Symm NN')
        results, self.logNN = self.runGS( gs, name )
        
        return results

    def runJastrow( self, learning_rate=None, name="Jastrow" ):

        class Jastrow(nknn.Module):
            @nknn.compact
            def __call__(self, x):
                x = jnp.atleast_2d(x)
                return jax.vmap(self.single_evaluate, in_axes=(0))(x)

            def single_evaluate(self, x):
                v_bias = self.param(
                    "visible_bias", nn.initializers.normal(), (x.shape[-1],), complex
                )

                J = self.param(
                    "kernel", nn.initializers.normal(), (x.shape[-1],x.shape[-1]), complex
                )

                return x.T@J@x + jnp.dot(x, v_bias)

        model = Jastrow()

        # compile the ground-state optimization loop
        gs = self.makeGS( model, learning_rate, simple=True )

        print('### Jastrow calculation')
        results, self.logJastrow = self.runGS( gs, name )
        
        return results
    
    def runCNN( self, learning_rate=None, name="/CNN" ):
        features = 4 
        # define symm neural network
        class NNModel(nn.Module):
            @nn.compact
            def __call__(this, x):
                
                rho = nknn.Conv(
                        features=4,
                        kernel_size=3,
                        use_bias=True,
                        strides=1,
                        dtype=float,
                        kernel_init=nk.nn.initializers.normal(stddev=0.1),
                        bias_init=nk.nn.initializers.normal(stddev=0.1)
                    )(x)
                rho = nknn.Conv(
                        features=16,
                        kernel_size=2,
                        use_bias=True,
                        strides=1,
                        dtype=float,
                        kernel_init=nk.nn.initializers.normal(stddev=0.1),
                        bias_init=nk.nn.initializers.normal(stddev=0.1)
                    )(rho)
                rho = nknn.Dense(
                    features=alpha,
                    kernel_init=nn.initializers.normal(stddev=0.001),
                    name="Modulus"
                )( rho )
                rho = nknn.relu( rho )

                phase = nknn.Conv(
                        features=4,
                        kernel_size=3,
                        use_bias=True,
                        strides=1,
                        dtype=float,
                        kernel_init=nk.nn.initializers.normal(stddev=0.1),
                        bias_init=nk.nn.initializers.normal(stddev=0.1)
                    )(x)
                phase = nknn.Conv(
                        features=16,
                        kernel_size=2,
                        use_bias=True,
                        strides=1,
                        dtype=float,
                        kernel_init=nk.nn.initializers.normal(stddev=0.1),
                        bias_init=nk.nn.initializers.normal(stddev=0.1)
                    )(phase)
                phase = nknn.Dense(
                    features=alpha,
                    kernel_init=nn.initializers.normal(stddev=0.001),
                    name="Phase"
                )( phase )
                phase = nknn.relu( phase )


                return jnp.sum(rho, axis=(-1, -2, -3, -4)) + 1.0j*jnp.sum( phase, axis=(-1, -2, -3, -4))

        # create model with alpha=1
        model = NNModel()

        # compile the ground-state optimization loop
        gs = self.makeGS( model, learning_rate )

        print('### Symm NN')
        results, self.logNN = self.runGS( gs, name )
        
        return results

    def runGCNN( self, feature_dims, learning_rate=None, name="/GCNN" ):
        #Feature dimensions of hidden layers, from first to last
        #Number of layers
        num_layers = len( feature_dims )

        #Define the GCNN
        model = nk.models.GCNN(symmetries = self.graph.translation_group() , layers = num_layers, features = feature_dims)

        # compile the ground-state optimization loop
        gs = self.makeGS( model, learning_rate, simple=True )

        print('### Convolutional Neural Network')
        results, self.logCNN = self.runGS( gs, name )
        
        return results

    def run( self, name=None, learning_rate=None, feature_dims=None ):
        m = self.model_type
        if m == "Exact":
            return self.runExact()
        if m == "Jastrow":
            name = "/Jastrow" if not name else name
            return self.runJastrow( learning_rate, name=name )
        if m == "NN":
            name = "/NN" if not name else name
            return self.runNN( learning_rate, name=name )
        if m == "SymmNN":
            name = "/SymmNN" if not name else name
            return self.runSymmNN( learning_rate, name=name )
        if m == "CNN":
            name = "/CNN" if not name else name
            return self.runCNN( learning_rate, name=name )
        if m == "GCNN":
            name = "/GCNN" if not name else name
            return self.runGCNN( feature_dims, learning_rate, name=name )
        if m == "RBM":
            name = "/RBM" if not name else name
            return self.runRBM( learning_rate, name=name )
        if m == "SymmRBM":
            name = "/SymmRBM" if not name else name
            return self.runSymmRBM( learning_rate, name=name )

if __name__ == "__main__":
    if len(argv) < 2:
        print("""NN Package. Usages:
    python3 ./NN log ./logfile.log
    python3 ./NN log ./logfile.log
    """)
        exit(0)
    
    params = {}
    command = []
    i = 1
    while i < len(argv):
        x = argv[i]
        if x[:2] == "--":
            print( "param: ", x[2:] ) 
            params[ x[2:] ] = float( argv[i+1] )
            i+=2
            continue
        command += [ x ] 
        i+=1

    action = command[0]
    print( command )

    model = Model(params) 

    if action == "log":
        for logFile in command[1:]:
            print( "opening ", logFile )
            log = json.load( open(logFile+".log") )
            print( model.doStats( log, makeNewFigures=False ) )
        plt.show()
        print( model.exact )
