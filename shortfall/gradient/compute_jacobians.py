from shortfall.sim import Simulator, SimConfig
from shortfall.miners.repay_proportional import RepayProportionalShortfallMinerState
from shortfall.miners.burn_no_lease_no_fee import BurnShortfallMinerState
from shortfall.miners.repay_ratchet import RepayRatchetShortfallMinerState
from shortfall.miners.base import BaseMinerState
from shortfall.network import *
from shortfall.strategy import *
from shortfall.consts import *

import shortfall.gradient.jax_base as jax_base
import shortfall.gradient.jax_repay_ratchet as jax_repay_ratchet
import shortfall.gradient.jax_burn as jax_burn

import pandas as pd

from functools import partial
import numpy as np

from joblib import Parallel, delayed
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev
import multiprocessing

from tqdm.auto import tqdm

import pickle
import os
from shortfall.utils import tqdm_joblib
import argparse

days = 3 * YEAR + 1

# from:  https://github.com/google/jax/issues/3171#issuecomment-1140299630
@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)

def base_miner_experiment():
    initial_pledge_projection_period_days_vec = np.arange(18,23,1).astype(np.float32)
    supply_lock_target_vec = np.asarray([0.3, 0.4, 0.5])
    token_lease_fee_vec = np.arange(0.18, 0.22, 0.01)

    keys = []
    inputs = []
    total = 0
    for initial_pledge_projection_period_days in initial_pledge_projection_period_days_vec:
        for supply_lock_target in supply_lock_target_vec:
            for token_lease_fee in token_lease_fee_vec:
                x_in = jnp.asarray([initial_pledge_projection_period_days, 
                                                supply_lock_target,
                                                token_lease_fee]).astype(jnp.float32)
                key = '%0.03f,%0.03f,%0.03f' % (
                    initial_pledge_projection_period_days,
                    supply_lock_target,
                    token_lease_fee,
                )
                inputs.append(x_in)
                keys.append(key)
                total += 1
    result_fp = 'base.pkl'
    inputs_dict = {
        'initial_pledge_projection_period_days': initial_pledge_projection_period_days_vec,
        'supply_lock_target': supply_lock_target_vec,
        'token_lease_fee': token_lease_fee_vec,
    }
    sim_fn = jax_base.run_sim
    jacobian_fn = jacfwd(jax_base.jax_wrapper)
    
    return inputs, keys, total, result_fp, inputs_dict, sim_fn, jacobian_fn

def repay_ratchet_miner_experiment():
    initial_pledge_projection_period_days_vec = np.arange(18,23,1).astype(np.float32)
    supply_lock_target_vec = np.asarray([0.3, 0.4, 0.5])
    token_lease_fee_vec = np.arange(0.18, 0.22, 0.01)
    max_repayment_term_vec = np.asarray([900, 365*3, 1300]).astype(np.float32)  # 2.5, 3, 3.5 Y
    max_fee_reward_fraction_vec = np.asarray([0.23, 0.24, 0.25, 0.26, 0.27]).astype(np.float32)

    keys = []
    inputs = []
    total = 0
    for initial_pledge_projection_period_days in initial_pledge_projection_period_days_vec:
        for supply_lock_target in supply_lock_target_vec:
            for token_lease_fee in token_lease_fee_vec:
                for max_repayment_term in max_repayment_term_vec:
                    for max_fee_reward_fraction in max_fee_reward_fraction_vec:
                        x_in = jnp.asarray([initial_pledge_projection_period_days, 
                                                supply_lock_target,
                                                token_lease_fee,
                                                max_repayment_term,
                                                max_fee_reward_fraction]).astype(jnp.float32)
                        key = '%0.03f,%0.03f,%0.03f,%0.03f,%0.03f' % (
                            initial_pledge_projection_period_days,
                            supply_lock_target,
                            token_lease_fee,
                            max_repayment_term,
                            max_fee_reward_fraction
                        )
                        inputs.append(x_in)
                        keys.append(key)
                        total += 1
    result_fp = 'repay_ratchet.pkl'
    inputs_dict = {
        'initial_pledge_projection_period_days': initial_pledge_projection_period_days_vec,
        'supply_lock_target': supply_lock_target_vec,
        'token_lease_fee': token_lease_fee_vec,
        'max_repayment_term': max_repayment_term_vec,
        'max_fee_reward_fraction': max_fee_reward_fraction_vec,
    }
    sim_fn = jax_repay_ratchet.run_sim
    jacobian_fn = jacfwd(jax_repay_ratchet.jax_wrapper)

    return inputs, keys, total, result_fp, inputs_dict, sim_fn, jacobian_fn

def burn_miner_experiment():
    initial_pledge_projection_period_days_vec = np.arange(18,23,1).astype(np.float32)
    supply_lock_target_vec = np.asarray([0.3, 0.4, 0.5])
    token_lease_fee_vec = np.arange(0.18, 0.22, 0.01)
    max_shortfall_fraction_vec = np.asarray([0.3, 0.4, 0.5, 0.6, 0.7])

    keys = []
    inputs = []
    total = 0
    for initial_pledge_projection_period_days in initial_pledge_projection_period_days_vec:
        for supply_lock_target in supply_lock_target_vec:
            for token_lease_fee in token_lease_fee_vec:
                for max_shortfall_fraction in max_shortfall_fraction_vec:
                        x_in = jnp.asarray([initial_pledge_projection_period_days, 
                                                supply_lock_target,
                                                token_lease_fee,
                                                max_shortfall_fraction]).astype(jnp.float32)
                        key = '%0.03f,%0.03f,%0.03f,%0.03f' % (
                            initial_pledge_projection_period_days,
                            supply_lock_target,
                            token_lease_fee,
                            max_shortfall_fraction,
                        )
                        inputs.append(x_in)
                        keys.append(key)
                        total += 1
    result_fp = 'burn.pkl'
    inputs_dict = {
        'initial_pledge_projection_period_days': initial_pledge_projection_period_days_vec,
        'supply_lock_target': supply_lock_target_vec,
        'token_lease_fee': token_lease_fee_vec,
        'max_shortfall_fraction': max_shortfall_fraction_vec,
    }
    sim_fn = jax_burn.run_sim
    jacobian_fn = jacfwd(jax_burn.jax_wrapper)

    return inputs, keys, total, result_fp, inputs_dict, sim_fn, jacobian_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='base', 
                        choices=['base', 'repay_ratchet', 'burn'], required=False)
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join(os.environ['HOME'], 'shortfall_jax_results'), required=False)
    parser.add_argument('--n-jobs', type=int, default=0, required=False, 
                        help='Number of jobs to run in parallel, if 0 will do ncores-2')
    args = parser.parse_args()

    if args.experiment == 'base':
        inputs, keys, total, result_fname, inputs_dict, sim_fn, jacobian_fn = base_miner_experiment()
    elif args.experiment == 'repay_ratchet':
        inputs, keys, total, result_fname, inputs_dict, sim_fn, jacobian_fn = repay_ratchet_miner_experiment()
    elif args.experiment == 'burn':
        inputs, keys, total, result_fname, inputs_dict, sim_fn, jacobian_fn = burn_miner_experiment()
    else:
        raise NotImplemented("Not implemented yet")

    os.makedirs(args.output_dir, exist_ok=True)
    results_fp = os.path.join(args.output_dir, result_fname)
    n_jobs = multiprocessing.cpu_count()-2 if args.n_jobs == 0 else args.n_jobs

    if not os.path.exists(results_fp):
        # run the simulation for the given configurations
        val_results = {}
        jacobian_results = {}
        
        with tqdm_joblib(tqdm(desc="Eval", total=total)) as progress_bar:
            val_results_list = Parallel(n_jobs=n_jobs)(delayed(sim_fn)(x) for x in inputs)
        with tqdm_joblib(tqdm(desc="Jacobian", total=total)) as progress_bar:
            jacobian_results_list = Parallel(n_jobs=n_jobs)(delayed(jacobian_fn)(x) for x in inputs)
        
        # convert results to dictionary
        # NOTE: https://stackoverflow.com/a/60932855
        # we rely on the fact that joblib preserves order.  That way, the call to Jax is not corrupted
        # with additional outputs indicating which input was used to make the call.
        # this seems roundabout - so investigate
        master_idx = 0
        for k in keys:
            val_results[k] = pd.DataFrame(val_results_list[master_idx])
            jacobian_results[k] = jacobian_results_list[master_idx]
            master_idx += 1
        
        # store results
        store_dict = {
            'inputs': inputs_dict,
            'outputs': {
                'val_results': val_results,
                'jacobian_results': jacobian_results,
            }
        }

        with open(results_fp, 'wb') as f:
            pickle.dump(store_dict, f)
    
    print("Done!")
