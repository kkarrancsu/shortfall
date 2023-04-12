from typing import Callable

from ..consts import SECTOR_SIZE
from .base import BaseMinerState, SectorBunch
from ..network import NetworkState

class BurnShortfallMinerState(BaseMinerState):
    """A miner that burns an equivalent amount to the shortfall, but never pledges it."""

    DEFAULT_MAX_SHORTFALL_FRACTION = 0.50

    @staticmethod
    def factory(balance: float, max_shortfall_fraction: float = DEFAULT_MAX_SHORTFALL_FRACTION) -> Callable[[], BaseMinerState]:
        """Returns a function that creates new miner states."""
        return lambda: BurnShortfallMinerState(balance=balance, max_shortfall_fraction=max_shortfall_fraction)

    def __init__(self, balance: float, max_shortfall_fraction: float = DEFAULT_MAX_SHORTFALL_FRACTION):
        super().__init__(balance)
        self.fee_pending: float = 0
        self.max_shortfall_fraction = max_shortfall_fraction

    def summary(self):
        summary = super().summary()
        summary.update({
            'fee_pending': self.fee_pending,
        })
        return summary

    # Override
    def max_pledge_for_tokens(self, net: NetworkState, available_lock: float, duration: float) -> float:
        """The maximum incremental initial pledge commitment allowed for an incremental locking."""
        return available_lock / self.max_shortfall_fraction

    # Overrides
    def activate_sectors(self, net: NetworkState, power_eib: float, duration_days: float, lock: float = float("inf")):
        """
        Activates power and locks a specified pledge.
        Lock may be 0, meaning to lock the minimum (after shortfall), or inf to lock the full pledge requirement.
        If available balance is insufficient for the specified locking, the tokens are leased.
        Returns the power and pledge locked.
        """
        # assert power % SECTOR_SIZE == 0

        pledge_requirement = net.initial_pledge_for_power(power_eib)
        minimum_pledge = pledge_requirement * (1 - self.max_shortfall_fraction)

        if lock == 0:
            lock = minimum_pledge
        elif lock > pledge_requirement:
            lock = pledge_requirement
        # elif lock < minimum_pledge:
        #     raise RuntimeError(f"lock {lock} is less than minimum pledge {pledge_requirement}")
        self._lease(max(lock - self.available_balance(), 0))

        self.power_eib += power_eib
        self.pledge_locked += lock  # Only the initially locked amount is ever required to be pledged
        self.fee_pending += pledge_requirement - lock  # Pending fee captures the difference to the notional initial pledge

        expiration = net.day + duration_days
        self._expirations.setdefault(expiration, []).append(SectorBunch(power_eib, lock))

        return power_eib, lock

    # Override
    def receive_reward(self, net: NetworkState, reward: float):
        # Vesting is ignored.
        self._earn_reward(reward)

        # Calculate and burn shortfall fee
        if self.fee_pending > 0:
            collateral_target = self.pledge_locked + self.fee_pending
            collateral_pct = self.pledge_locked / collateral_target
            available_pct = collateral_pct * collateral_pct
            fee_take_rate = 1 - available_pct
            assert fee_take_rate >= 0
            assert fee_take_rate <= 1.0
            if fee_take_rate > 0:
                # Burn the fee
                fee_amount = min(reward * fee_take_rate, self.fee_pending)
                self._burn_fee(fee_amount)
                self.fee_pending -= fee_amount

        # Repay lease if possible.
        self._repay(min(self.lease, self.available_balance()))

    # Override
    def handle_day(self, net: NetworkState):
        """Executes end-of-day state updates"""
        # Accrue token lease fees.
        # The fee is added to the repayment obligation. If the miner has funds, it will pay it next epoch.
        fee = 0  # per discussion w/ tmellan

        # Expire power.
        # NOTE: I wonder if popping from a dictionary will mess up Jax's differentiation.
        # Do we need to pop, or is that for efficiency?
        expiring_now = self._expirations.pop(net.day, [])
        for sb in expiring_now:
            self.handle_expiration(sb)

    def handle_expiration(self, sectors: SectorBunch):
        # Reduce the outstanding fee in proportion to the power represented.
        # XXX it's not clear that this is appropriate policy.
        remaining_power_frac = (self.power_eib - sectors.power_eib) / self.power_eib
        self.fee_pending *= remaining_power_frac

        self.power_eib -= sectors.power_eib
        self.pledge_locked -= sectors.pledge
