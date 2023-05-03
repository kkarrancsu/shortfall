import math
from dataclasses import dataclass

from .consts import DAY, SECTOR_SIZE, YEAR, EXBIBYTE, EPOCHS_PER_DAY

import jax
import jax.numpy as jnp
import jax.lax as lax

import mechafil.sim as sim
from datetime import date, timedelta
import pandas as pd

# SUPPLY_LOCK_TARGET = 0.30
INITIAL_PLEDGE_PROJECTION_PERIOD = 20 * DAY

# Reward at epoch = initial reward * (1-r)^epochs
REWARD_DECAY = 1 - math.exp(math.log(1/2)/(6*YEAR))
# Baseline at epoch = initial baseline * (1+b)^epochs
BASELINE_GROWTH = math.exp(math.log(3)/YEAR) - 1


def compute_mechafil_trajectories(bearer_token, 
                                  start_date=None, 
                                  network_shortfall_rate=0.1,
                                  simulation_len=1000, 
                                  rbp=6, 
                                  rr=0.6, 
                                  fpr=0.8, 
                                  sector_duration=360):
    print('Computing mechafil trajectories')
    if start_date is None:
        start_date = date.today() - timedelta(days=2)
    network_data_start_date = date(2021, 3, 16)
    cil_df = sim.run_simple_sim(
        network_data_start_date,
        start_date,
        simulation_len,
        rr,
        rbp,
        fpr,
        sector_duration,
        bearer_token,
        qap_method='basic'
    )
    # cil_df = sim.run_shortfall_sim(
    #     network_data_start_date,
    #     start_date,
    #     simulation_len,
    #     rr,
    #     rbp,
    #     fpr,
    #     sector_duration,
    #     bearer_token,
    #     shortfall_rate=network_shortfall_rate,
    #     qap_method='basic'
    # )
    cil_df['day_rewards_per_sector'] = SECTOR_SIZE * cil_df.day_network_reward / cil_df.network_QAP

    # get the DF from forecast date start only, and convert the day index to match this simulation for easy indexing
    cil_df_forecast = cil_df[pd.to_datetime(cil_df['date']) >= pd.to_datetime(start_date)]
    cil_df_forecast['days'] = range(0,len(cil_df_forecast))
    cil_df_forecast.reset_index(inplace=True)
    print('Finished computing mechafil trajectories')
    return cil_df_forecast

@dataclass
class NetworkConfig:
    cil_df: pd.DataFrame

    # Fee p.a. on externally leased tokens.
    token_lease_fee: float
    supply_lock_target: float

DEFAULT_NETWORK_CONFIG = NetworkConfig(
    cil_df = None,
    token_lease_fee=0.20,
    supply_lock_target=0.30,
)

@dataclass
class NetworkState:
    cil_df_forecast: pd.DataFrame

    token_lease_fee: float
    supply_lock_target: float
    
    def __init__(self, cfg: NetworkConfig):
        self.cil_df_forecast = cfg.cil_df
        self.token_lease_fee = cfg.token_lease_fee
        self.supply_lock_target = cfg.supply_lock_target
      
        self.day = 0
        idx = self.get_df_idx()
        self.day_reward = self.cil_df_forecast.iloc[idx]['day_network_reward']
        self.power = self.cil_df_forecast.iloc[idx]['network_QAP'] / EXBIBYTE

    def get_df_idx(self):
        # return self.cil_df_forecast.index[self.cil_df_forecast['days'] == self.day].tolist()[0]
        return self.day

    def handle_day(self):
        self.day += 1
        idx = self.get_df_idx()
        self.day_reward = self.cil_df_forecast.iloc[idx]['day_network_reward']
        self.power = self.cil_df_forecast.iloc[idx]['network_QAP'] / EXBIBYTE

    def initial_pledge_for_power(self, power: float) -> float:
        """The initial pledge requirement for an incremental power addition."""
        storage = self.expected_reward_for_power(power, INITIAL_PLEDGE_PROJECTION_PERIOD)
        
        idx = self.get_df_idx()
        circulating_supply = self.cil_df_forecast.iloc[idx]['circ_supply']
        network_qa_power_EIB = self.cil_df_forecast.iloc[idx]['network_QAP'] / EXBIBYTE
        network_baseline_power_EIB = self.cil_df_forecast.iloc[idx]['network_baseline'] / EXBIBYTE
        consensus = circulating_supply * power * self.supply_lock_target / max(network_qa_power_EIB, network_baseline_power_EIB)
        return storage + consensus

    def power_for_initial_pledge(self, pledge: float) -> int:
        """The maximum power that can be committed for an nominal pledge."""
        rewards = self.projected_reward(INITIAL_PLEDGE_PROJECTION_PERIOD)

        idx = self.get_df_idx()
        circulating_supply = self.cil_df_forecast.iloc[idx]['circ_supply']
        network_qa_power_EIB = self.cil_df_forecast.iloc[idx]['network_QAP'] / EXBIBYTE
        
        power = pledge * network_qa_power_EIB / (rewards + circulating_supply * self.supply_lock_target)
        return power

    def expected_reward_for_power(self, power: float, duration: float) -> float:
        """Projected rewards for some power over a period, taking reward decay into account."""
        # power is in EiB, convert to # of sectors
        num_sectors = power * EXBIBYTE / SECTOR_SIZE
        # use the day_rewards_per_sector to get the total expected reward
        idx = int(self.get_df_idx())
        ii_start = int(idx)
        ii_end = int(idx+duration)
        total_expected_reward_per_sector = self.cil_df_forecast.iloc[ii_start:ii_end]['day_rewards_per_sector'].sum()
        total_expected_reward = total_expected_reward_per_sector * num_sectors
        return total_expected_reward

    def projected_reward(self, duration: float) -> float:
        """Projects a per-epoch reward into the future, taking decay into account"""
        idx = self.get_df_idx()
        return self.cil_df_forecast.iloc[idx:int(idx+duration)]['day_network_reward'].sum()

    def fee_for_token_lease(self, amount: float, duration: float) -> float:
        return amount * self.token_lease_fee * duration / YEAR


def sum_over_exponential_decay(duration: float, decay: float) -> float:
    # SUM[(1-r)^x] for x in 0..duration
    # return (1 - math.pow(1 - decay, duration) + decay * math.pow(1 - decay, duration)) / decay
    return (1 - jnp.power(1. - decay, duration) + decay * jnp.power(1. - decay, duration)) / decay
