# Safe RL with Temporal Logic Constraints

Code for PPO with LTL constraints in safety-critical navigation tasks.

## Installation

```bash
pip install -r requirements.txt
pip install safety-gymnasium
```

## Quick Start

**Important:** Run all commands from the project root directory and set PYTHONPATH:

```bash
export PYTHONPATH="$PWD/src:$PWD/CRL:$PYTHONPATH"
```

### Train PPO (Baseline)
```bash
python CRL/train_ppo.py --algorithm PPO --env ZoneEnv-v0 --seed 100 --total-steps 200000 --n-envs 4
```

### Train PPO-Shield (with action shielding)
```bash
python CRL/train_ppo.py --algorithm PPOShield --env ZoneEnv-v0 --seed 100 --total-steps 200000 --n-envs 4
```

### Train PPO-LTL (with LTL constraints)
```bash
python CRL/train_ppo.py --algorithm PPOLag --env ZoneEnv-v0 --seed 100 --total-steps 200000 --n-envs 4 --formula "(!blue U green) & F yellow" --finite --cost-limit 0.05 --lagrangian-lr 0.01
```

## Configuration

Key parameters:
- `--seed`: Random seed
- `--total-steps`: Total training timesteps
- `--n-envs`: Number of parallel environments
- `--formula`: LTL specification (for PPO-LTL)
- `--cost-limit`: Maximum allowed cost (for PPO-LTL)
- `--lagrangian-lr`: Lagrangian multiplier learning rate (for PPO-LTL)

## LTL to Automata

The `src/` directory contains tools for converting LTL formulas to deterministic Büchi automata.

## License

See LICENSE file.
