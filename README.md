<div align="center">    
 
# SpanLuke: Enhance Legal Entity Recognition with LUKE and SpanMarker
![](https://github.com/lambdavi/L-NER/blob/final/media/logo_temp.png?raw=True)
</div>

## Description   
The goal of this project is to identify entities in legal text.

This repository starts from the code of "PoliToHFI at SemEval-2023 Task 6: Leveraging Entity-Aware and Hierarchical Transformers For Legal Entity Recognition and Court Judgement Prediction" submitted to the SemEval-2023, Task 6.

## How to run   
First, install dependencies (python==3.10 required)
```bash
# clone project   
git clone https://github.com/lambdavi/L-NER.git

# install requirements   
cd L-NER 
pip install -r requirements.txt

# 
 ```   

# Available scripts
 ```bash
# run training script (example: training on PickAndPlace-v3 task)   
python train.py --env_id PandaPickAndPlace-v3 --algo ddpg

# run hyperparameters tuning (example: on PandaReach-v3 with SAC) 
python tuning.py --env_id PandaReach-v3 --algo sac

# eval your agent
python eval.py --env_id PandaReach-v3 --algo ddpg --path models/PandaReach_DDPG_50000_steps.zip

# just visualize the environment (random actions)
python visualize.py --env_id PandaReach-v3
```

#### Arguments for train:

- `--lr`: Learning rate
  - Type: float
  - Default: 0.001

- `--gamma`: Gamma value
  - Type: float
  - Default: 0.99

- `--buffer_size`: Buffer size
  - Type: int
  - Default: 1,000,000

- `--batch_size`: Batch size
  - Type: int
  - Default: 100

- `--tau`: Tau value
  - Type: float
  - Default: 0.005

- `--learning_starts`: When to start the learning
  - Type: int
  - Default: 100

- `--steps`: Number of steps
  - Type: int
  - Default: 50,000

- `--env_id`: Environment ID
  - Type: str
  - Default: "PandaReach-v3"
  - Choices: ["PandaReach-v3", "PandaReachDense-v3", "PandaPickAndPlace-v3", "PandaPickAndPlaceDense-v3"]

- `--algo`: Algorithm to solve the task
  - Type: str
  - Default: "ddpg"
  - Choices: ["ddpg", "sac", "dqn"]

### Citation   
```
@article{gallouedec2021pandagym,
    title        = {{panda-gym: Open-Source Goal-Conditioned Environments for Robotic Learning}},
    author       = {Gallou{\'e}dec, Quentin and Cazin, Nicolas and Dellandr{\'e}a, Emmanuel and Chen, Liming},
    year         = 2021,
    journal      = {4th Robot Learning Workshop: Self-Supervised and Lifelong Learning at NeurIPS},
    }
```   