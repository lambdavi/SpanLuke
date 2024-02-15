from typing import Any
from typing import Dict
import panda_gym
import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DDPG, SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch


N_TRIALS = 100
N_STARTUP_TRIALS = 20
N_EVALUATIONS = 2
N_TIMESTEPS = int(1e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

DEFAULT_HYPERPARAMS = {
    "batch_size": 16,
    "ds_train_path": "data/NER_TRAIN/NER_TRAIN_ALL.json",
    "ds_valid_path": "data/NER_DEV/NER_DEV_ALL.json",
    "output_folder": "results/",
    "num_epochs":1
}


def sample_hp(trial: optuna.Trial) -> Dict[str, Any]:

    learning_rate = trial.suggest_float("lr", 5e-6, 1, log=True)
    peft_mode = trial.suggest_categorical("peft", ["lora", "adalora", "ia3"])
    lora_dropout = trial.suggest_float("dropout", 0.001, 1, log=True)
    lora_rank = 2**trial.suggest_int("rank", 2, 9)
    bool_alpha = trial.suggest_int("bool_alpha", 0, 1)
    lora_alpha = bool_alpha*lora_rank if bool_alpha==1 else int(lora_rank/2)

    # Display true values.
    trial.set_user_attr("lora_rank_", lora_rank)
    trial.set_user_attr("lora_alpha_", lora_alpha)

    return {
        "learning_rate": learning_rate,
        "peft_mode": peft_mode,
        "lora_dropout": lora_dropout,
        "lora_rank":lora_rank,
        "lora_alpha":lora_alpha 
    }


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: gymnasium.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    # Create the RL model.
    
    kwargs.update(sample_hp(trial))
    model = DDPG(**kwargs)
    elif ALGO=="sac":
        kwargs.update(sample_sac_params(trial))
        model = SAC(**kwargs)
    elif ALGO=="ppo":
        kwargs.update(sample_ppo_params(trial))
        model = PPO(**kwargs)
    else:
        raise NotImplementedError
    # Create env used for evaluation.
    eval_env = Monitor(gymnasium.make(ENV_ID))
    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))