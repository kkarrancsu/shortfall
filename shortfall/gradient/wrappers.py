import numpy as np
import jax.numpy as jnp

import pandas as pd

from shortfall.sim import Simulator, SimConfig
from shortfall.miners.base import BaseMinerState
from shortfall.network import *
from shortfall.strategy import *
from shortfall.consts import *

from shortfall.miners.burn_with_fee import BurnWithFeeShortfallMinerState
from shortfall.miners.repay_ratchet import RepayRatchetShortfallMinerState

BORROW_AMT = 0.75
APY = 0.25
DEFAULT_SAMPLING_RATE_DAYS = 1

def get_income(stats_list,):
    df = pd.DataFrame(stats_list)
    return df.iloc[1:]['reward_earned'] - df.iloc[1:]['fee_burned'] - df.iloc[1:]['lease_fee_accrued'] 

def get_net_equity(stats_list):
    df = pd.DataFrame(stats_list)
    return df.iloc[1:]['net_equity']

def get_sector_duration_fofr(df,norm=True):
    income = get_income(df)
    norm_factor = float(df.iloc[0]['pledge_locked']) if norm else 1
    income_trajectory = income/norm_factor
    return income_trajectory.iloc[-1]

def compute_income_borrow_apy(stats_list, pledge_borrowed, apy=0.25, 
                              sampling_rate_days=DEFAULT_SAMPLING_RATE_DAYS):
    """
    """
    indices = np.arange(sampling_rate_days, len(stats_list), sampling_rate_days)
    t = indices / 365.
    
    pledge_repayment_schedule = pledge_borrowed * jnp.exp(jnp.log(1.0+apy) * t)
    costs = pledge_repayment_schedule - pledge_borrowed
    baseline_income = np.asarray([stats_list[ii]['reward_earned']-costs[c_idx] for c_idx, ii in enumerate(indices)])
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
    baseline_income = np.asarray([income_vector[ii] for ii in indices]).cumsum()
    repay_remaining_vector = np.asarray([repay_remaining_vector[ii] for ii in indices])

    # return how much is remaining to be repaid, b/c this could be > 0 if:
    #  - the take rate is too low
    #  - the repayment cap is too high

    return baseline_income, repay_remaining_vector

def compute_baseline(network_config, power, sector_duration=3*YEAR+1, shortfall_frac = 0.0):
    base_strategy = StrategyConfig.power_limited(power * TIBIBYTE / EXBIBYTE, sector_duration, shortfall_frac)
    base_miner_factory = BaseMinerState.factory(balance=0)
    baseline_cfg = SimConfig(
        network=network_config,
        strategy=base_strategy,
        miner_factory=base_miner_factory,
    )
    base_stats = Simulator(baseline_cfg).run_all(sector_duration, DAY)
    return base_stats

def get_burn_stats(network_config, power=50, 
                   max_shortfall_possible=BurnWithFeeShortfallMinerState.DEFAULT_MAX_SHORTFALL_FRACTION, 
                   shortfall_pct_pow=BurnWithFeeShortfallMinerState.DEFAULT_SHORTFALL_TAKE_RATE_EXPONENT, 
                   sector_duration=3*YEAR+1, shortfall_frac=0.0, miner_balance=0, 
                   network_uptake=0.0, fee_structure='linear', max_fee_frac=0.0,
                   stats_interval=DAY):
    """
    max_shortfall_possible - the maximum amount of shortfall that is possible at the protocol level
    shortfall_frac - a value between [0,max_shortfall_possible] that is the shortfall to take
    token_lease_fee - the borrowing rate for the tokens that are not acquired through shortfall
    """
    burn_miner_factory = BurnWithFeeShortfallMinerState.factory(
        balance=miner_balance, 
        max_shortfall_fraction=max_shortfall_possible,
        shortfall_take_rate_exponent=shortfall_pct_pow,
        network_uptake=network_uptake,
        fee_structure = fee_structure,
        fee_kwargs= {'max_fee_frac': max_fee_frac}
    )

    burn_pl_cfg = SimConfig(
        network=network_config,
        strategy=StrategyConfig.power_limited(power * TIBIBYTE / EXBIBYTE, sector_duration, shortfall_frac),
        miner_factory=burn_miner_factory,
    )
    stats_burn = Simulator(burn_pl_cfg).run_all(sector_duration, stats_interval)
    # return pd.DataFrame(data=stats_burn)
    return stats_burn

def get_repay_stats(network_config, power=50,
                    max_fee_reward_fraction=RepayRatchetShortfallMinerState.DEFAULT_MAX_FEE_REWARD_FRACTION,
                    sector_duration=3*YEAR+1, shortfall_frac=0.0, miner_balance=0,
                    stats_interval=DAY):
    max_repayment_term = 3. * 365 * DAY
    repay_miner_factory = RepayRatchetShortfallMinerState.factory(
        balance=miner_balance,
        max_repayment_term=max_repayment_term,
        max_fee_reward_fraction=max_fee_reward_fraction,
        reward_projection_decay=REWARD_DECAY + BASELINE_GROWTH
    )

    repay_cfg = SimConfig(
        network=network_config,
        strategy=StrategyConfig.power_limited(power * TIBIBYTE / EXBIBYTE, sector_duration, shortfall_frac),
        miner_factory=repay_miner_factory,
    )
    stats_repay = Simulator(repay_cfg).run_all(sector_duration, stats_interval)
    # return pd.DataFrame(data=stats_repay)
    return stats_repay