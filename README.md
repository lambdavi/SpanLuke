<div align="center">    
 
# SpanLuke: Enhance Legal Entity Recognition with LUKE, SpanMarker and LoRA
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

# run training script  
python main.py \
    --ds_train_path data/NER_TRAIN/NER_TRAIN_ALL.json \
    --ds_valid_path data/NER_DEV/NER_DEV_ALL.json \
    --output_folder results/ \
    --batch 16 \
    --acc_step 4 \
    --num_epochs 5 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.06 \
    --model_path studio-ousia/luke-base \
    --span
```

<!--
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
-->