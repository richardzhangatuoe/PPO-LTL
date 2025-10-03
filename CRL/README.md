# Safe RL with LTL Constraints (CARLA)

Code for PPO with Lagrangian constraints in CARLA autonomous driving scenarios.

## Installation

### 1. Download CARLA

Download and install CARLA 0.9.13 from the [official release page](https://github.com/carla-simulator/carla/releases/tag/0.9.13).

### 2. Setup Environment

```bash
# Create conda environment
conda create -y -n vlm-rl python=3.8
conda activate vlm-rl

# Install PyTorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# Install requirements
pip install -r requirements.txt
```

### 3. Configure CARLA Path

If using `--start_carla`, update `CARLA_ROOT` in `carla_env/envs/carla_route_env.py` to your CARLA installation path.

Alternatively, start CARLA server manually:
```bash
./CARLA_0.9.13/CarlaUE4.sh -quality_level=Low -benchmark -fps=15 -RenderOffScreen -prefernvidia -carla-world-port=2000
```

## Quick Start

### Train PPO (Baseline)
```bash
python train.py --config crl_ppo_vanilla_seed1 --total_timesteps 100000 --device cuda:0 --port 2000 --start_carla --no_render
```

### Train PPO-Shield (with action shielding)
```bash
python train.py --config crl_ppo_shield_seed1 --total_timesteps 100000 --device cuda:0 --port 2000 --start_carla --no_render
```

### Train PPO-LTL (Lagrangian with LTL constraints)
```bash
python train.py --config crl_ppo_A_seed1 --total_timesteps 100000 --device cuda:0 --port 2000 --start_carla --no_render
```

## Configuration

Key parameters:
- `--total_timesteps`: Training duration
- `--device`: `cpu`, `cuda:0`, etc.
- `--port`: CARLA server port (default: 2000)
- `--start_carla`: Automatically start CARLA server
- `--no_render`: Disable rendering for faster training

## License

See LICENSE file.

