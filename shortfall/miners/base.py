from collections import defaultdict
from typing import NamedTuple, Callable

from ..consts import SECTOR_SIZE
from ..network import NetworkState

class SectorBunch(NamedTuple):
    power: float
    pledge: float

class BaseMinerState:
    """Miner with leased tokens but no pledge shortfall behaviour."""

    def __init__(self, balance: float, supply_lock_target=0.3):
        self.power: float = 0
        self.balance: float = balance
        self.lease: float = 0.0
        self.pledge_locked: float = 0.0

        self.reward_earned: float = 0.0
        self.fee_burned: float = 0.0
        self.lease_fee_accrued = 0.0

        self.supply_lock_target = supply_lock_target

        # Scheduled expiration of power, by epoch.
        self._expirations: dict[int, list[SectorBunch]] = defaultdict(list[SectorBunch])

    @staticmethod
    def factory(balance: float):
        """Returns a function that creates new miner states."""
        return lambda: BaseMinerState(balance=balance)

    def summary(self):
        net_equity = self.balance - self.lease
        return {
            'power': self.power,
            'balance': self.balance,
            'lease': self.lease,
            'pledge_locked': self.pledge_locked,
            'available': self.available_balance(),
            'net_equity': net_equity,

            'reward_earned': self.reward_earned,
            'fee_burned': self.fee_burned,
            'lease_fee_accrued': self.lease_fee_accrued,
        }

    def available_balance(self) -> float:
        return self.balance - self.pledge_locked

    def max_pledge_for_tokens(self, net: NetworkState, available_lock: float, duration: float) -> float:
        """The maximum incremental initial pledge commitment allowed for an incremental locking."""
        return available_lock

    def activate_sectors(self, net: NetworkState, power: float, duration: float, lock: float = float("inf")):
        """
        Activates power and locks a specified pledge.
        Lock must be at least the pledge requirement; it's a parameter only so subclasses can be more generous.
        If available balance is insufficient for the specified locking, the tokens are leased.
        Returns the power and pledge locked.
        """
        # assert power % SECTOR_SIZE == 0

        pledge_requirement = net.initial_pledge_for_power(power)

        if lock >= pledge_requirement:
            lock = pledge_requirement
        else:
            raise RuntimeError(f"lock {lock} is less than minimum pledge {pledge_requirement}")
        self._lease(max(lock - self.available_balance(), 0))

        self.power += power
        self.pledge_locked += lock
        expiration = net.day + duration
        self._expirations[expiration].append(SectorBunch(power, pledge_requirement))

        return power, lock

    def receive_reward(self, net: NetworkState, reward: float):
        # Vesting is ignored.
        self._earn_reward(reward)

        # Repay lease if possible.
        self._repay(min(self.lease, self.available_balance()))

    def handle_day(self, net: NetworkState):
        """Executes end-of-day state updates"""
        # Accrue token lease fees.
        # The fee is added to the repayment obligation. If the miner has funds, it will pay it next epoch.
        fee = net.fee_for_token_lease(self.lease, 1)
        self._accrue_lease_fee(fee)

        # Expire power.
        expiring_now = self._expirations.pop(net.day, [])
        for sb in expiring_now:
            self.handle_expiration(sb)

    def handle_expiration(self, sectors: SectorBunch):
        self.power -= sectors.power
        self.pledge_locked -= sectors.pledge

    def _earn_reward(self, v: float):
        assert v >= 0
        self.balance += v
        self.reward_earned += v

    def _burn_fee(self, v: float):
        assert v >= 0
        # assert v <= self.available_balance(), f"fee {v} exceeds available balance {self.available_balance()}"
        self.balance -= v
        self.fee_burned += v

    def _lease(self, v: float):
        assert v >= 0
        self.balance += v
        self.lease += v

    def _repay(self, v: float):
        # assert v >= 0, f"repay {v} is negative"
        assert v <= self.lease
        assert v <= self.available_balance()
        self.balance -= v
        self.lease -= v

    def _accrue_lease_fee(self, v: float):
        assert v >= 0
        self.lease += v
        self.lease_fee_accrued += v

