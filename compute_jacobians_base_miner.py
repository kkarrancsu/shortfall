from sim import Simulator, SimConfig
from miners.repay_proportional import RepayProportionalShortfallMinerState
from miners.burn import BurnShortfallMinerState
from miners.repay_ratchet import RepayRatchetShortfallMinerState
from miners.base import BaseMinerState
from network import *
from strategy import *
from consts import *

import dataclasses
import pandas as pd

from functools import partial
import numpy as np

import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev
from jax import random

from tqdm.auto import tqdm

import pickle
import os

days = 3 * YEAR + 1

def run_sim(x):
    initial_pledge_projection_period_days = x[0]
    supply_lock_target = x[1]
    token_lease_fee = x[2]

    network = dataclasses.replace(MAINNET_APR_2023,
        token_lease_fee=token_lease_fee,
        reward_decay=REWARD_DECAY, # or try REWARD_DECAY + BASELINE_GROWTH
        initial_pledge_projection_period_days=initial_pledge_projection_period_days,
        supply_lock_target=supply_lock_target
    )

    miner_factory = BaseMinerState.factory(balance=0)
    cfg = SimConfig(
        network=network,
        strategy=StrategyConfig.pledge_limited(1000.0, 3 * YEAR, True),
        miner_factory=miner_factory,
    )
    stats_interval = DAY
    stats = Simulator(cfg).run_all(days, stats_interval)
    return stats

# from:  https://github.com/google/jax/issues/3171#issuecomment-1140299630
@partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: jax.lax.dynamic_slice(a, (start,), (size,)))(starts)

def jax_wrapper(x):
    stats = run_sim(x)
    
    # JAX doesn't like DataFrames, so we extract the output quantity that we're interested in directly
    # from the dictionary

#     returns_raw = jnp.array([(x['reward_earned'] - (x['fee_burned'] + x['lease_fee_accrued'])) 
#                             for x in stats])
    # i think these are cumulative quantities rather than by "day" quantities,
    # so we need to index at the right time to get the total returns rather than window-sum
#     returns_1y = jnp.sum(moving_window(returns_raw, 365), axis=1)
#     return returns_1y

    indices = np.arange(365, len(stats), 365)
    returns_raw = [(stats[ii]['reward_earned'] - (stats[ii]['fee_burned'] + stats[ii]['lease_fee_accrued'])) 
                   for ii in indices]
                
    returns_raw = jnp.array(returns_raw)
    return returns_raw

jacobian = jacfwd(jax_wrapper)  # forward mode differentiation seems faster

if __name__ == "__main__":
    initial_pledge_projection_period_days_vec = np.arange(18,23,1).astype(np.float32)
    supply_lock_target_vec = np.asarray([0.3, 0.4, 0.5])
    token_lease_fee_vec = np.arange(0.18, 0.22, 0.01)

    results_dir = os.path.join(os.environ['HOME'], 'shortfall_jax_results')
    os.makedirs(results_dir, exist_ok=True)
    results_fp = os.path.join(results_dir, 'base_miner_strategy.pkl')

    if not os.path.exists(results_fp):
        # compute the miner metrics for this sweep
        val_results = {}
        jacobian_results = {}
        pbar = tqdm(total = 
                    len(initial_pledge_projection_period_days_vec)*
                    len(supply_lock_target_vec)*
                    len(token_lease_fee_vec)
        )
        for initial_pledge_projection_period_days in initial_pledge_projection_period_days_vec:
            for supply_lock_target in supply_lock_target_vec:
                for token_lease_fee in token_lease_fee_vec:
                    x_in = jnp.asarray([initial_pledge_projection_period_days, 
                                        supply_lock_target,
                                        token_lease_fee]).astype(jnp.float32)
                    stats = run_sim(x_in)
                    stats_df = pd.DataFrame(data=stats)
                    jc = jacobian(x_in)
                    key = '%0.03f,%0.03f,%0.03f' % (
                        initial_pledge_projection_period_days,
                        supply_lock_target,
                        token_lease_fee
                    )
                    val_results[key] = stats_df
                    jacobian_results[key] = np.asarray(jc)
                    pbar.update(1)
        pbar.close()

        # store results
        store_dict = {
            'inputs': {
                'initial_pledge_projection_period_days_vec': initial_pledge_projection_period_days_vec,
                'supply_lock_target_vec': supply_lock_target_vec,
                'token_lease_fee_vec': token_lease_fee_vec,
            },
            'outputs': {
                'val_results': val_results,
                'jacobian_results': jacobian_results,
            }
        }

        with open(results_fp, 'wb') as f:
            pickle.dump(store_dict, f)
    
    print("Done!")
