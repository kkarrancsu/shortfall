import numpy as np
import jax.numpy as jnp

import dataclasses

from shortfall.sim import Simulator, SimConfig
from shortfall.miners.repay_proportional import RepayProportionalShortfallMinerState
from shortfall.miners.burn_no_lease_no_fee import BurnShortfallMinerState
from shortfall.miners.repay_ratchet import RepayRatchetShortfallMinerState
from shortfall.miners.base import BaseMinerState
from shortfall.network import *
from shortfall.strategy import *
from shortfall.consts import *

BORROW_AMT = 0.75
APY = 0.25
DEFAULT_SAMPLING_RATE_DAYS = 30

def compute_income_borrow_apy(stats_list, pledge_borrowed, apy=0.25, 
                              sampling_rate_days=DEFAULT_SAMPLING_RATE_DAYS):
    """
    """
    indices = np.arange(sampling_rate_days, len(stats_list), sampling_rate_days)
    t = indices / 365.
    
    pledge_repayment_schedule = pledge_borrowed * jnp.exp(jnp.log(1.0+apy) * t)
    costs = pledge_repayment_schedule - pledge_borrowed
    baseline_income = jnp.asarray([stats_list[ii]['reward_earned']-costs[c_idx] for c_idx, ii in enumerate(indices)])
    return baseline_income

def compute_income_take_rewards(stats_list, pledge_borrowed, repayment_cap=1.5, take_rate=0.40, 
                                sampling_rate_days=DEFAULT_SAMPLING_RATE_DAYS):
    """
    """
    # apply the revenue take on a day-by-day basis, then apply subsampling at the end
    total_repay_obligation = pledge_borrowed * repayment_cap
    repayment_vector = []
    income_vector = []
    repay_remaining_vector = []
    total_repaid = 0
    day_reward = 0
    repay_remaining = total_repay_obligation
    for ii in range(len(stats_list)):
        total_reward = stats_list[ii]['reward_earned']
        if ii == 0:
            day_reward = total_reward
        else:
            day_reward = total_reward - stats_list[ii-1]['reward_earned']
        
        repay_remaining = total_repay_obligation - total_repaid
        if repay_remaining > 0:
            # print(ii, repay_remaining, day_reward)
            day_repay = min(day_reward * take_rate, repay_remaining)
            reward_remaining = day_reward - day_repay
            total_repaid += day_repay
            repayment_vector.append(day_repay)
        else:
            repayment_vector.append(0)
            reward_remaining = day_reward
        income_vector.append(reward_remaining)
        repay_remaining_vector.append(repay_remaining)

    indices = np.arange(sampling_rate_days, len(stats_list), sampling_rate_days)
    baseline_income = jnp.asarray([income_vector[ii] for ii in indices]).cumsum()
    repay_remaining_vector = jnp.asarray([repay_remaining_vector[ii] for ii in indices])

    # return how much is remaining to be repaid, b/c this could be > 0 if:
    #  - the take rate is too low
    #  - the repayment cap is too high

    return baseline_income, repay_remaining_vector

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