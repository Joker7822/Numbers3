#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numbers3_evo_agent as agent
import numbers3_runner as runner
from numbers3_champion import apply_to_agent, load_champion


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--champion", default="state/evolution_champion.json")
    known, rest = p.parse_known_args()
    champion = load_champion(Path(known.champion))
    cfg = apply_to_agent(agent, champion["config"])

    original = agent.AgentConfig

    def champion_agent_config(validation_draws=1000, top_k=20, **kwargs):
        return original(
            validation_draws=validation_draws,
            top_k=top_k,
            half_life_draws=cfg["half_life_draws"],
            evolution_rate=cfg["evolution_rate"],
            weight_temperature=cfg["weight_temperature"],
            random_state=cfg["random_state"],
        )

    runner.AgentConfig = champion_agent_config
    sys.argv = [sys.argv[0]] + rest
    runner.main()


if __name__ == "__main__":
    main()
