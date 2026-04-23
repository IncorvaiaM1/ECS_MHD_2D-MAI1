'''
This will be the junkyard of my code. If a function is useful, but I don't think you should care much how it works,
it goes here.
'''

import jax
import jax.flatten_util
import jax.numpy as jnp

import lib.loss_functions as loss_functions






def line_search_unravel(x, step, obj_fn, unravel_fn, b, max_iters=20):
    """
    Armijo backtracking line search in the Newton direction.

    Halves `damp` until ||f(x - damp*step)|| < ||b|| or max_iters is reached.

    Parameters
    ----------
    x          : flat parameter vector (jnp array)
    step       : Newton step (flat jnp array, GMRES solution to J*s = f)
    obj_fn     : objective callable, input_dict -> residual_dict (already JIT'd)
    unravel_fn : converts flat vector back to input_dict
    b          : current residual as flat vector (used for ||f|| comparison)
    max_iters  : maximum number of halving steps

    Returns
    -------
    x_new : updated flat parameter vector
    damp  : damping factor used (jnp scalar)
    """
    flatten = lambda d: jax.flatten_util.ravel_pytree(d)[0]
    norm_b = float(jnp.linalg.norm(b))
    damp = 1.0
    for _ in range(max_iters):
        x_try = x - damp * step
        norm_try = float(jnp.linalg.norm(flatten(obj_fn(unravel_fn(x_try)))))
        if norm_try < norm_b:
            return x_try, jnp.array(damp)
        damp *= 0.5
    return x - damp * step, jnp.array(damp)


def create_state_from_turb( turb_dict, idx, param_dict ):
    #Get conditions for RPO guess
    f = turb_dict['fs'][idx[0]-1,:,:,:]
    f = jnp.fft.irfft2(f)

    #Period
    T = param_dict['dt'] * param_dict['ministeps'] * (idx[1] - idx[0])

    #spatial shift
    #Eventually do a search for a good intial guess
    sx = 0.0

    #number of timesteps 
    steps = param_dict['ministeps'] * (idx[1] - idx[0])
    steps = int(steps) #JAX complains if we do not cast steps

    param_dict.update({ 'steps': steps } )

    #Create a dictionary of optimizable field
    input_dict = {"fields": f, "T": T, "sx": sx}

    #Delete keys from the turbulent trajectory param_dict that we won't need anymore to avoid confusion
    del param_dict['dt']
    del param_dict['ministeps']

    return input_dict, param_dict






def compile_objective_and_Jacobian( input_dict, param_dict, obj ):
    
    #Capture param_dict and JIT the objective function
    objective = jax.jit( lambda input_dict: obj(input_dict, param_dict) )

    #Compile the objective function.
    f = objective(input_dict)

    import time
    start = time.time()
    f = objective(input_dict)
    stop = time.time()
    walltime0 = stop - start

    #Define the Jacobian action and compile it
    jac = jax.jit( lambda primal, tangent: jax.jvp( objective, (primal,), (tangent,))[1] )
    _ = jac( input_dict, input_dict )

    start = time.time()
    Jf = jac( input_dict, input_dict )
    stop = time.time()
    walltime1 = stop - start

    print(f"Evaluating objective: {walltime0:.3} seconds")
    print(f"Evaluating Jacobian: {walltime1:.3} seconds")
    #print(f"Evaluating Jacobian transpose: {walltime2:.3} seconds")
    return objective, jac





def choose_objective_fn( shooting_mode, integrate_mode, param_dict, num_checkpoints, adaptive_dict ):
    #Check that all options are within their allowed values
    assert shooting_mode in {"single_shooting", "multi_shooting"}
    assert integrate_mode in {"adaptive", "fixed_timesteps"}


    if shooting_mode == "single_shooting" and integrate_mode == "fixed_timesteps":
        print(f"Choosing single shooting with fixed timesteps:")
        print(f"steps = {param_dict['steps']}")
        print(f"num_checkpoints = {num_checkpoints}")
        #define number of segements for memory checkpointing
        param_dict.update(  {"ministeps": int(param_dict["steps"]//num_checkpoints), "num_checkpoints": int(num_checkpoints)})
        #Define the RPO objective function
        obj = loss_functions.objective_RPO_with_checkpoints

    if shooting_mode == "single_shooting" and integrate_mode == "adaptive":
        print(f"Choosing single shooting with adaptive timestepping:")
        obj = lambda input_dict, param_dict: loss_functions.objective_RPO_adaptive( input_dict, param_dict, adaptive_dict )

    if shooting_mode == "multi_shooting":
        print(f"ERROR: multishooting is experimental and not quite implemented. Yell at Matt.")
        exit()
        #Define the RPO objective function
        #obj = loss_functions.objective_RPO_multishooting

    return obj, param_dict
