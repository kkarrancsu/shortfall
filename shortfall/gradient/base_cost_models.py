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


def compute_income(stats_list, borrow_amt=0.75, apy=0.25, sampling_rate_days=30):
    indices = np.arange(sampling_rate_days, len(stats_list), sampling_rate_days)
    t = indices / 365.
    pledge_borrowed = borrow_amt * stats_list[0]['pledge_locked']
    pledge_repayment_schedule = pledge_borrowed * jnp.exp(jnp.log(1.0+apy) * t)
    costs = pledge_repayment_schedule - pledge_borrowed
    baseline_income = jnp.asarray([stats_list[ii]['reward_earned']-costs[c_idx] for c_idx, ii in enumerate(indices)])
    return baseline_income

def compute_baseline(initial_pledge_projection_period_days, token_lease_fee, supply_lock_target, 
                     days = 3 * YEAR + 1, stats_interval = DAY):
    base_miner_factory = BaseMinerState.factory(balance=0)
    network = dataclasses.replace(MAINNET_APR_2023,
        token_lease_fee=token_lease_fee,
        reward_decay=REWARD_DECAY, # or try REWARD_DECAY + BASELINE_GROWTH
        initial_pledge_projection_period_days=initial_pledge_projection_period_days,
        supply_lock_target=supply_lock_target
    )
    baseline_cfg = SimConfig(
        network=network,
        strategy=StrategyConfig.pledge_limited(1000.0, 3 * YEAR, False),
        miner_factory=base_miner_factory,
    )
    base_stats = Simulator(baseline_cfg).run_all(days, stats_interval)
    return base_stats