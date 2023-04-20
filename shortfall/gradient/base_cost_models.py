import numpy as np
import jax.numpy as jnp

import dataclasses

from shortfall.sim import Simulator, SimConfig
from shortfall.miners.repay_proportional import RepayProportionalShortfallMinerState
from shortfall.miners.burn import BurnShortfallMinerState
from shortfall.miners.repay_ratchet import RepayRatchetShortfallMinerState
from shortfall.miners.base import BaseMinerState
from shortfall.network import *
from shortfall.strategy import *
from shortfall.consts import *

BORROW_AMT = 0.75
APY = 0.25
SAMPLING_RATE_DAYS = 30

def compute_income_borrow_apy(stats_list, pledge_borrowed, apy=0.25, sampling_rate_days=30):
    indices = np.arange(sampling_rate_days, len(stats_list), sampling_rate_days)
    t = indices / 365.
    # pledge_borrowed = borrow_amt * stats_list[0]['pledge_locked']
    pledge_repayment_schedule = pledge_borrowed * jnp.exp(jnp.log(1.0+apy) * t)
    costs = pledge_repayment_schedule - pledge_borrowed
    baseline_income = jnp.asarray([stats_list[ii]['reward_earned']-costs[c_idx] for c_idx, ii in enumerate(indices)])
    return baseline_income

def compute_baseline(initial_pledge_projection_period_days, 
                     supply_lock_target, 
                     strategy,
                     days = 3 * YEAR + 1, stats_interval = DAY, ):
    token_lease_fee = np.nan  # this should be a noop for base strategy which does not take shortfall
                              # set to NaN to ensure it doesn't get used, otherwise we'll see an error
    
    base_miner_factory = BaseMinerState.factory(balance=0)
    network = dataclasses.replace(MAINNET_APR_2023,
        token_lease_fee=token_lease_fee,
        reward_decay=REWARD_DECAY, # or try REWARD_DECAY + BASELINE_GROWTH
        initial_pledge_projection_period_days=initial_pledge_projection_period_days,
        supply_lock_target=supply_lock_target
    )
    baseline_cfg = SimConfig(
        network=network,
        strategy=strategy,
        miner_factory=base_miner_factory,
    )
    base_stats = Simulator(baseline_cfg).run_all(days, stats_interval)
    return base_stats