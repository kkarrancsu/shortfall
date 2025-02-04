import json
import sys
import time

from shortfall.consts import TIBIBYTE
from shortfall.miners.burn import BurnShortfallMinerState
from shortfall.miners.repay_ratchet import RepayRatchetShortfallMinerState
from shortfall.network import *
from shortfall.sim import SimConfig, Simulator
from shortfall.strategy import StrategyConfig

def main(args):
    # TODO: argument processing
    epochs = 3 * YEAR + 1
    stats_interval = DAY

    # miner_factory=BurnShortfallMinerState.factory(balance=0)
    miner_factory = RepayRatchetShortfallMinerState.factory(
        balance=0,
        max_repayment_term=3 * YEAR,
        max_fee_reward_fraction=0.25,
        reward_projection_decay=REWARD_DECAY + BASELINE_GROWTH
    )
    cfg = SimConfig(
        network=MAINNET_APR_2023,
        strategy=StrategyConfig.pledge_limited(1000.0, 3 * YEAR, shortfall=1.0),
        # strategy=StrategyConfig.power_limited(100 * TIBIBYTE, 3 * YEAR, shortfall=1.0),
        miner_factory=miner_factory,
    )
    sim = Simulator(cfg)

    start_time = time.perf_counter()
    stats = sim.run_all(epochs, stats_interval)
    end_time = time.perf_counter()

    print(cfg.strategy)
    for s in stats:
        print(json.dumps(s))
    latency = end_time - start_time
    print("Simulated {} epochs in {:.1f} sec".format(epochs, latency))

if __name__ == '__main__':
    main(sys.argv)
