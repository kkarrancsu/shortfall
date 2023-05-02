import math
from dataclasses import dataclass

from .consts import DAY, SECTOR_SIZE, YEAR, EXBIBYTE, EPOCHS_PER_DAY

import jax
import jax.numpy as jnp
import jax.lax as lax

# SUPPLY_LOCK_TARGET = 0.30
INITIAL_PLEDGE_PROJECTION_PERIOD = 20 * DAY

# Reward at epoch = initial reward * (1-r)^epochs
REWARD_DECAY = 1 - math.exp(math.log(1/2)/(6*YEAR))
# Baseline at epoch = initial baseline * (1+b)^epochs
BASELINE_GROWTH = math.exp(math.log(3)/YEAR) - 1


@dataclass
class NetworkConfig:
    day: float
    qa_power: float
    raw_byte_power: float
    baseline_power: float
    day_reward: float
    reward_decay: float
    circulating_supply: float
    # Fee p.a. on externally leased tokens.
    token_lease_fee: float

    initial_pledge_projection_period_days: float
    supply_lock_target: float

# 2023-02-01, epoch 2563440
MAINNET_FEB_2023 = NetworkConfig(
    day=0,
    qa_power=21530229500983050000. / EXBIBYTE,
    raw_byte_power=16006761814138290000. / EXBIBYTE,
    baseline_power=15690691297578078000. / EXBIBYTE,
    day_reward=5*19.0057947578366*EPOCHS_PER_DAY,
    reward_decay=REWARD_DECAY,
    circulating_supply=434191286.621853,
    token_lease_fee=0.20,
    initial_pledge_projection_period_days=20 * DAY,
    supply_lock_target=0.30,
)

# 2023-04-01, epoch 2733360
MAINNET_APR_2023 = NetworkConfig(
    day=0,
    qa_power=22436033270683107000. / EXBIBYTE,
    raw_byte_power=14846032093347054000. / EXBIBYTE,
    baseline_power=17550994139680311000. / EXBIBYTE,
    day_reward=5*16.7867382504675*EPOCHS_PER_DAY,
    reward_decay=REWARD_DECAY,
    circulating_supply=456583469.869076,
    token_lease_fee=0.20,
    initial_pledge_projection_period_days=20 * DAY,
    supply_lock_target=0.30,
)

@dataclass
class NetworkState:
    day: float
    power: float
    power_baseline: float
    circulating_supply: float
    epoch_reward: float
    reward_decay: float
    token_lease_fee: float
    supply_lock_target: float

    def __init__(self, cfg: NetworkConfig):
        self.day = cfg.day
        self.power = cfg.qa_power
        self.power_baseline = cfg.baseline_power
        self.circulating_supply = cfg.circulating_supply
        self.day_reward = cfg.day_reward

        self.reward_decay = cfg.reward_decay

        # these are candidates for optimization
        self.token_lease_fee = cfg.token_lease_fee
        self.supply_lock_target = cfg.supply_lock_target

    def handle_day(self):
        self.day += 1
        self.day_reward *= (1-self.reward_decay)
        self.power_baseline *= (1+BASELINE_GROWTH)

    def initial_pledge_for_power(self, power: float) -> float:
        """The initial pledge requirement for an incremental power addition."""
        storage = self.expected_reward_for_power(power, INITIAL_PLEDGE_PROJECTION_PERIOD)
        consensus = self.circulating_supply * power * self.supply_lock_target / max(self.power, self.power_baseline)
        return storage + consensus

    def power_for_initial_pledge(self, pledge: float) -> int:
        """The maximum power that can be committed for an nominal pledge."""
        rewards = self.projected_reward(self.day_reward, INITIAL_PLEDGE_PROJECTION_PERIOD)
        power = pledge * self.power / (rewards + self.circulating_supply * self.supply_lock_target)
        # return int((power // SECTOR_SIZE) * SECTOR_SIZE)
        return power

    def expected_reward_for_power(self, power: float, duration_days: float, decay=REWARD_DECAY) -> float:
        """Projected rewards for some power over a period, taking reward decay into account."""
        # Note this doesn't use alpha/beta filter estimate or take baseline rewards into account.
        if self.power <= 0:
            return self.projected_reward(self.day_reward, duration_days, decay)
        return self.projected_reward(self.day_reward * power / self.power, duration_days, decay)

    def projected_reward(self, epoch_reward: float, duration: float, decay=REWARD_DECAY) -> float:
        """Projects a per-epoch reward into the future, taking decay into account"""
        return epoch_reward * sum_over_exponential_decay(duration, decay)

    def fee_for_token_lease(self, amount: float, duration_days: float) -> float:
        return amount * self.token_lease_fee * duration_days / YEAR


def sum_over_exponential_decay(duration: float, decay: float) -> float:
    # SUM[(1-r)^x] for x in 0..duration
    # return (1 - math.pow(1 - decay, duration) + decay * math.pow(1 - decay, duration)) / decay
    return (1 - jnp.power(1. - decay, duration) + decay * jnp.power(1. - decay, duration)) / decay
