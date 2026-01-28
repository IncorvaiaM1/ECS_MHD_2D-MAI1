'''
Written by Matthew Golden some time in 2025

PURPOSE:
The purpose of this script is to find a null vector of a Jacobian for the purpose of continuation.
This way, we can make an educated guess. To avoid numerical instability, I plan on using shifted power iteration.

Jv = 0*v -> (J+I)v = 1*v.
'''


import time
import jax
import jax.flatten_util
import jax.numpy as jnp

#import lib.mhd_jax as mhd_jax
import lib.loss_functions as loss_functions
from lib.linalg import gmres
import lib.dictionaryIO as dictionaryIO
import lib.utils as utils

from jax.experimental import io_callback

import os

os.makedirs( "temp_data/newton", exist_ok=True)

###############################
# Construct numerical grid
###############################

print(jax.devices())

precision = jnp.float64  # Double or single precision
# If you want double precision, change JAX defaults
if (precision == jnp.float64):
    jax.config.update("jax_enable_x64", True)

input_dict, param_dict = dictionaryIO.load_dicts("solutions/Re40/RPO1.npz")

#param_dict['forcing_str'] = "lambda x,y : -4*jnp.cos(4*y)"
param_dict = dictionaryIO.recompute_grid_information(input_dict, param_dict)

##################
# NEWTON OPTIONS
##################
mode = "Lawson_RK4"

# Verify that we have a solution to the governing equations to begin with
obj = lambda input_dict, param_dict : loss_functions.objective_RPO(input_dict, param_dict, mode)

def print_info():
    f = obj(input_dict, param_dict)

    norm = lambda x : jnp.linalg.norm(jnp.reshape(x, [-1]))
    rel_err = norm(f["fields"]) / norm(input_dict["fields"])

    print("Info for base solution:")
    print(f"T = {input_dict["T"]:.3e}")
    print(f"sx= {input_dict["sx"]:.3e}")
    print(f"rot= {param_dict["rot"]}")
    print(f"shift_reflect_ny= {param_dict["shift_reflect_ny"]}")
    
    print(f"relative error for RPO is {rel_err:.3e}")
    print(f"max pointwise error is {jnp.max( jnp.abs( f["fields"] )):.3e}")

print_info()





#Modify the input_dict and param_dict to contain the bifurcation parameter
input_dict.update({"b0": param_dict["b0"]})
del param_dict["b0"]

#Define a new function that we want null vectors of.
def continuation_objective(input_dict, param_dict):
    #Re-add b0 to param_dict
    param_dict.update({"b0": input_dict["b0"]})

    #Evaluate RPO conditions as usual
    out_dict = obj(input_dict, param_dict)

    #Add a zero for b0
    out_dict.update( {"b0": 0 * param_dict['b0']} )
    return out_dict

#Capture param_dict
obj2 = lambda input_dict : continuation_objective(input_dict, param_dict)

#Define JVP of the continuation objective
jac = jax.jit( lambda primal, tangent: jax.jvp( obj2, (primal,), (tangent,))[1] )

flatten = lambda x : jax.flatten_util.ravel_pytree(x)[0]
unflatten = jax.flatten_util.ravel_pytree(input_dict)[1]

def lin_op(v):
    #This linear operator should correspond to (J+I)*v
    
    #Apply J
    Jv = flatten(jac(input_dict, unflatten(v)))

    #Compute (J+I)*v
    Jv = Jv - v

    #Dealias
    Jv = unflatten(Jv)
    Jv['fields'] = param_dict['mask'] * jnp.fft.rfft2(Jv['fields'])
    Jv['fields'] = jnp.fft.irfft2(Jv['fields'])

    #Kill Bx since we want to do continuation along By
    Jv['b0'] = Jv['b0'].at[0].set(0.0)

    #Back to 1D vector form
    Jv = flatten(Jv)

    return Jv

lin_op_parallel = jax.jit( jax.vmap( lin_op ))

#Get the number of degrees of freedom
n = jnp.size(flatten(input_dict))

r = 12 #number of vectors I want to iterate

key = jax.random.PRNGKey(0)
V = jax.random.normal(key, shape=(r,n))

maxit = 128 + 32

for i in range(maxit):
    print(i)

    #Apply shifted Jacobian
    V = lin_op_parallel(V)

    #Orthonormalize
    Q, _ = jnp.linalg.qr( V.T )

    V = Q.T

# After iteration, evaluate the Jacobian action one final time
JV = lin_op_parallel(V)
R = V @ JV.T #hopefully upper triangular

# Compute the eigendecomposition of R
w, v = jnp.linalg.eig(R)

# Find the closest eigenvalue to -1 in this case
index = jnp.argmin( jnp.abs(w + 1) )

print(f"Closest eigenvalue to -1 is {w[index]}")


from scipy.io import savemat

import numpy as np
savemat("null_vectors/R.mat", {"R" : np.asarray(jax.device_get(R))})

for i in range(r):
    mini_dict = unflatten(V[i,:])
    savemat(f"null_vectors/{i}.mat", {key: np.asarray(jax.device_get(val)) for key, val in mini_dict.items()})

null_vec = V.T @ v[:,index]
mini_dict = unflatten(null_vec)
savemat(f"null_vectors/null_vec.mat", {key: np.asarray(jax.device_get(val)) for key, val in mini_dict.items()})