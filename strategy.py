from dataclasses import dataclass

from consts import SECTOR_SIZE, EXBIBYTE
from miner import MinerState
from network import NetworkState

@dataclass
class StrategyConfig:
    # The maximum amount of storage power available at any one time.
    max_power: int
    # The maximum total amount of onboarding to perform ever.
    # Prevents re-investment after this amount (even after power expires).
    max_power_onboard: int
    # The maximum total tokens to pledge ever.
    # Prevents re-investment after this amount (even after pledge is returned).
    max_pledge_onboard: float
    # Commitment duration for onboarded power.
    commitment_duration: int
    # Maximum tokens to lease from external party.
    max_pledge_lease: float
    # Whether to use a pledge shortfall (always at maximum available).
    take_shortfall: bool

    @staticmethod
    def power_limited(power: int, commitment: int, shortfall: False):
        """A strategy limited by power onboarding rather than tokens."""
        return StrategyConfig(
            max_power=power,
            max_power_onboard=power,
            max_pledge_onboard=1e18,
            commitment_duration=commitment,
            max_pledge_lease=1e28,
            take_shortfall=shortfall,
        )

    @staticmethod
    def pledge_limited(pledge: float, commitment: int, shortfall: False):
        """A strategy limited by pledge tokens (from balance or borrowed) rather than power."""
        return StrategyConfig(
            max_power=1000 * EXBIBYTE,
            max_power_onboard=1000 * EXBIBYTE,
            max_pledge_onboard=pledge,
            commitment_duration=commitment,
            max_pledge_lease=1e18,
            take_shortfall=shortfall,
        )

    @staticmethod
    def pledge_lease_limited(lease: float, commitment: int, shortfall: False):
        """A strategy limited by pledge tokens borrowable."""
        return StrategyConfig(
            max_power=1000 * EXBIBYTE,
            max_power_onboard=1000 * EXBIBYTE,
            max_pledge_onboard=1e18,
            commitment_duration=commitment,
            max_pledge_lease=lease,
            take_shortfall=shortfall,
        )

class MinerStrategy:
    def __init__(self, cfg: StrategyConfig):
        self.cfg = cfg
        self._onboarded = 0
        self._pledged = 0.0

    def act(self, net: NetworkState, m: MinerState):
        available_lock = m.available_balance() + (self.cfg.max_pledge_lease - m.lease)
        available_lock = min(available_lock, self.cfg.max_pledge_onboard - self._pledged)
        if self.cfg.take_shortfall:
            available_pledge = net.max_pledge_for_tokens(available_lock)
        else:
            available_pledge = available_lock

        target_power = min(self.cfg.max_power - m.power, self.cfg.max_power_onboard - self._onboarded)
        power_for_pledge = net.power_for_initial_pledge(available_pledge)

        # Set power and lock amounts depending on which is the limiting factor.
        if target_power <= power_for_pledge:
            # Limited by power, so pledge either all available, or zero (which will result in minimum with shortfall)
            if self.cfg.take_shortfall:
                lock = 0
            else:
                lock = available_lock
        else:
            # Limited by pledge
            lock = available_lock
            target_power = power_for_pledge

        # Round power to a multiple of sector size.
        target_power = (target_power // SECTOR_SIZE) * SECTOR_SIZE

        if target_power > 0:
            power, pledge = m.activate_sectors(net, target_power, self.cfg.commitment_duration, lock=lock)
            self._onboarded += power
            self._pledged += pledge
