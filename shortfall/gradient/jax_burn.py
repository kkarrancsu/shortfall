from shortfall.sim import Simulator, SimConfig
from shortfall.miners.repay_proportional import RepayProportionalShortfallMinerState
from shortfall.miners.burn import BurnShortfallMinerState
from shortfall.miners.repay_ratchet import RepayRatchetShortfallMinerState
from shortfall.miners.base import BaseMinerState
from shortfall.network import *
from shortfall.strategy import *
from shortfall.consts import *

import dataclasses
import pandas as pd

from functools import partial
import numpy as np

from joblib import Parallel, delayed
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev
from jax import random
import multiprocessing

from tqdm.auto import tqdm

import pickle
import os
from shortfall.utils import tqdm_joblib

days = 3 * YEAR + 1

def run_sim(x):
    initial_pledge_projection_period_days = x[0]
    supply_lock_target = x[1]
    token_lease_fee = x[2]
    max_shortfall_fraction = x[3]

    network = dataclasses.replace(MAINNET_APR_2023,
        token_lease_fee=token_lease_fee,
        reward_decay=REWARD_DECAY, # or try REWARD_DECAY + BASELINE_GROWTH
        initial_pledge_projection_period_days=initial_pledge_projection_period_days,
        supply_lock_target=supply_lock_target
    )

    miner_factory = BurnShortfallMinerState.factory(
        balance=0,
        max_shortfall_fraction=max_shortfall_fraction,
    )
    cfg = SimConfig(
        network=network,
        strategy=StrategyConfig.pledge_limited(1000.0, 3 * YEAR, True),
        miner_factory=miner_factory,
    )
    stats_interval = DAY
    stats = Simulator(cfg).run_all(days, stats_interval)
    return stats

def jax_wrapper(x):
    stats = run_sim(x)
    
    # JAX doesn't like DataFrames, so we extract the output quantity that we're interested in directly
    # from the dictionary

    # returns_raw = jnp.array([(x['reward_earned'] - (x['fee_burned'] + x['lease_fee_accrued'])) 
    #                         for x in stats])
    # i think these are cumulative quantities rather than by "day" quantities,
    # so we need to index at the right time to get the total returns rather than window-sum
    # returns_1y = jnp.sum(moving_window(returns_raw, 365), axis=1)
    # return returns_1y

    indices = np.arange(365, len(stats), 365)
    returns_raw = [(stats[ii]['reward_earned'] - (stats[ii]['fee_burned'] + stats[ii]['lease_fee_accrued'])) 
                   for ii in indices]
                
    returns_raw = jnp.array(returns_raw)
    return returns_raw