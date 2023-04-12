#!/usr/bin/env python

import os
import argparse
import dataclasses

from shortfall.sim import Simulator, SimConfig
from shortfall.miners.repay_proportional import RepayProportionalShortfallMinerState
from shortfall.miners.burn import BurnShortfallMinerState
from shortfall.miners.repay_ratchet import RepayRatchetShortfallMinerState
from shortfall.miners.base import BaseMinerState
from shortfall.network import *
from shortfall.strategy import *
from shortfall.consts import *

import shortfall.gradient.base_cost_models as bcm
import pandas as pd

from functools import partial
import numpy as np

from joblib import Parallel, delayed
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev, value_and_grad
import multiprocessing

from tqdm.auto import tqdm

import pickle
import os
from shortfall.utils import tqdm_joblib
import argparse

days = 3 * YEAR + 1
stats_interval = DAY

# don't use these for params for optimization
initial_pledge_projection_period_days = 20
supply_lock_target = 0.3

BORROW_AMT = 0.75
APY = 0.25
SAMPLING_RATE_DAYS = 30

def compute_repay_ratchet(x):
    token_lease_fee = x[0]
    max_fee_reward_fraction = x[1]
    max_repayment_term = x[2]

    repay_miner_factor = RepayRatchetShortfallMinerState.factory(
        balance=0.,
        max_repayment_term=max_repayment_term,
        max_fee_reward_fraction=max_fee_reward_fraction,
        reward_projection_decay=REWARD_DECAY + BASELINE_GROWTH
    )

    network = dataclasses.replace(MAINNET_APR_2023,
        token_lease_fee=token_lease_fee,
        reward_decay=REWARD_DECAY, # or try REWARD_DECAY + BASELINE_GROWTH
        initial_pledge_projection_period_days=initial_pledge_projection_period_days,
        supply_lock_target=supply_lock_target
    )

    burn_cfg = SimConfig(
        network=network,
        strategy=StrategyConfig.pledge_limited(1000.0, 3 * YEAR, True),
        miner_factory=repay_miner_factor,
    )
    burn_stats = Simulator(burn_cfg).run_all(days, stats_interval)
    return burn_stats
    
def compute_loss(x, income_base):
    burn_stats = compute_repay_ratchet(x)
    # extract the quantities of interest
    sampling_rate = 30  # sampling rate of the income curve
    indices = np.arange(sampling_rate, len(burn_stats), sampling_rate)
    income_burn = jnp.asarray([burn_stats[ii]['net_equity'] for ii in indices])
    
    # try the 10% lower to start with
    burn_target = income_base * 0.9
    burn_actual = income_burn

    mse_loss = jnp.mean(jnp.power(burn_target - burn_actual, 2)) / len(indices)

    return mse_loss

def optimize_parameters():
    # starting params
    token_lease_fee = 0.2
    max_fee_reward_fraction = 0.25
    max_repayment_term = 3 * 365

    base_stats = bcm.compute_baseline(initial_pledge_projection_period_days, supply_lock_target, token_lease_fee)
    income_base = bcm.compute_income(base_stats, borrow_amt=BORROW_AMT, apy=APY, sampling_rate_days=SAMPLING_RATE_DAYS)

    x = jnp.asarray([token_lease_fee, max_fee_reward_fraction, max_repayment_term])
    loss_grad = value_and_grad(compute_loss, argnums=0)
    # l, g, = loss_grad(x)
    # print(l)
    # print(g)
    print(compute_loss(x, income_base))
    
    n_iter = 100
    alpha = 1e-5
    for i in range(n_iter):
        loss, grads = loss_grad(x, income_base)
        print(i, loss, grads, x)
        x = x - alpha * grads

    # return x

if __name__ == '__main__':
    optimize_parameters()