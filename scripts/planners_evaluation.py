"""Usage: planners_evaluation.py [options]

Compare performances of several planners

Options:
  -h --help
  --generate <true or false>  Generate new data [default: True].
  --show <true_or_false>      Plot results [default: True].
  --missing <true_or_false>   Generate missing data for data_file [default: False].
  --directory <path>          Specify directory path [default: ./out/planners].
  --data_file <path>          Specify output data file name [default: data.csv].
  --budgets <start,end,N>     Computational budgets available to planners, in logspace [default: 1,3,100].
  --seeds <(s,)n>             Number of evaluations of each configuration, with an optional first seed [default: 10].
  --processes <p>             Number of processes [default: 4]
  --chunksize <c>             Size of data chunks each processor receives [default: 1]
  --range <start:end>         Range of budgets to be plotted
  --brange <start:end>        Index of the budget to be computed
  --test                      Run just test and do not execute training
"""
from argparse import ArgumentError
from ast import literal_eval
from pathlib import Path

import tqdm
from docopt import docopt
from collections import OrderedDict
from itertools import product
from multiprocessing.pool import Pool

import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import logging
import random

sns.set(font_scale=1.5, rc={"text.usetex": True})

from rl_agents.agents.common.factory import load_environment, agent_factory
from rl_agents.trainer.evaluation import Evaluation

logger = logging.getLogger(__name__)

gamma = 0.7
SEED_MAX = 1e9


def env_configs():
    # return ['configs/CartPoleEnv/env.json']
    # return ['configs/HighwayEnv/env.json']
    # return ['configs/GridWorld/collect.json']
    # return ['configs/FiniteMDPEnv/env_garnet.json']
    # return ['configs/SailingEnv/env.json']
    # return [Path("configs") / "DummyEnv" / "line_env.json"]
    return [Path("configs") / "InvertedPendulumsEnv/env2rewardB.json"]


def mcts_agent_config(
    sufix="", gamma=None, temperature=200, step_strategy="reset", horizon=None
):
    if gamma == None:
        _g = [0.8, 0.85, 0.9, 0.95]
    else:
        _g = gamma
    ma = dict()
    for i, g in enumerate(_g):
        ma[f"MCTS{i + 1}{sufix}"] = {
            "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
            "max_depth": 30,
            "temperature": temperature,
            "gamma": g,
            "step_strategy": step_strategy,
            "horizon": horizon,
        }
    return ma


def old_default_agents_configs():
    various_agents = {
        "random": {
            "__class__": "<class 'rl_agents.agents.simple.random.RandomUniformAgent'>"
        },
        "KL-OLOP": {
            "__class__": "<class 'rl_agents.agents.tree_search.olop.OLOPAgent'>",
            "gamma": gamma,
            "upper_bound": {"type": "kullback-leibler", "threshold": "1*np.log(time)"},
            "lazy_tree_construction": True,
            "continuation_type": "uniform",
        },
        "OPD": {
            "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
            "gamma": gamma,
        },
        "MDP-GapE": {
            "__class__": "<class 'rl_agents.agents.tree_search.mdp_gape.MDPGapEAgent'>",
            "gamma": gamma,
            "accuracy": 0,
            "confidence": 1,
            "upper_bound": {
                "type": "kullback-leibler",
                "time": "global",
                "threshold": "1*np.log(time)",
                "transition_threshold": "0.1*np.log(time)",
            },
            "max_next_states_count": 2,
            "continuation_type": "uniform",
            "step_strategy": "reset",
        },
        "BRUE": {
            "__class__": "<class 'rl_agents.agents.tree_search.brue.BRUEAgent'>",
            "gamma": gamma,
            "step_strategy": "reset",
        },
        "UCT": {
            "__class__": "<class 'rl_agents.agents.tree_search.mcts.MCTSAgent'>",
            "gamma": gamma,
            "closed_loop": False,
        },
        "GBOP": {
            "__class__": "<class 'rl_agents.agents.tree_search.graph_based_stochastic.StochasticGraphBasedPlannerAgent'>",
            "gamma": gamma,
            "upper_bound": {
                "type": "kullback-leibler",
                "threshold": "0*np.log(time)",
                "transition_threshold": "0.1*np.log(time)",
            },
            "max_next_states_count": 3,
            "accuracy": 5e-2,
        },
        "GBOP-D": {
            "__class__": "<class 'rl_agents.agents.tree_search.graph_based.GraphBasedPlannerAgent'>",
            "gamma": gamma,
        },
        "value_iteration": {
            "__class__": "<class 'rl_agents.agents.dynamic_programming.value_iteration.ValueIterationAgent'>",
            "gamma": gamma,
            "iterations": int(3 / (1 - gamma)),
        },
    }
    return OrderedDict(various_agents)


def agent_configs():
    # generates MCTS1 to MCTS4 with gamma [0.8, 0.85, 0.9, 0.95], "reset", temperature 200, no horizon
    agents = mcts_agent_config()
    # generates MCTS1s to MCTS8s for "subtree"
    agents.update(
        mcts_agent_config(
            sufix="s",
            gamma=[0.8, 0.85, 0.9, 0.95, 0.94, 0.96, 0.97, 0.98,],
            step_strategy="subtree",
        )
    )
    return OrderedDict(agents)


def selected_agents():
    return [
        # "MCTS1",
        # "MCTS2",
        "MCTS3",
        "MCTS4",
        # "MCTS1s",
        # "MCTS2s",
        "MCTS3s",
        "MCTS4s",
    ]


latex_names = {
    "simple_regret": "simple regret $r_n$",
    "total_reward": "total reward $R$",
    "mean_return": "mean return $E[R]$",
    "1/epsilon": r"${1}/{\epsilon}$",
    "MDP-GapE": r"\texttt{MDP-GapE}",
    "KL-OLOP": r"\texttt{KL-OLOP}",
    "BRUE": r"\texttt{BRUE}",
    "GBOP": r"\texttt{GBOP}",
    "UCT": r"\texttt{UCT}",
    "budget": r"budget $n$",
    "MCTS1": "MCTS $\gamma=0.8$",
    "MCTS2": "MCTS $\gamma=0.85$",
    "MCTS3": "MCTS $\gamma=0.9$",
    "MCTS4": "MCTS $\gamma=0.95$",
    "MCTS1s": "MCTS Subtree $\gamma=0.8$",
    "MCTS2s": "MCTS Subtree $\gamma=0.85$",
    "MCTS3s": "MCTS Subtree $\gamma=0.9$",
    "MCTS4s": "MCTS Subtree $\gamma=0.95$",
    "MCTS5s": "MCTS Subtree $\gamma=0.94$",
    "MCTS6s": "MCTS Subtree $\gamma=0.96$",
    "MCTS7s": "MCTS Subtree $\gamma=0.97$",
    "MCTS8s": "MCTS Subtree $\gamma=0.98$",
}


def evaluate(experiment):
    # Prepare workspace
    seed, budget, agent_config, env_config, path = experiment
    gym.logger.set_level(gym.logger.DISABLED)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Make environment
    env = load_environment(env_config)

    # Make agent
    agent_name, agent_config = agent_config
    agent_config["budget"] = int(budget)
    agent = agent_factory(env, agent_config)

    logger.debug(
        "Evaluating agent {} with budget {} on seed {}".format(agent_name, budget, seed)
    )

    # Compute true value
    compute_regret = False
    compute_return = True
    if compute_regret:
        env.seed(seed)
        observation = env.reset()
        vi = agent_factory(env, old_default_agents_configs()["value_iteration"])
        best_action = vi.act(observation)
        action = agent.act(observation)
        q = vi.state_action_value
        simple_regret = q[vi.mdp.state, best_action] - q[vi.mdp.state, action]
        gap = q[vi.mdp.state, best_action] - np.sort(q[vi.mdp.state, :])[-2]
    else:
        simple_regret = 0
        gap = 0

    if compute_return:
        # Evaluate
        evaluation = Evaluation(
            env,
            agent,
            directory=Path("out") / "planners" / agent_name,
            num_episodes=1,
            sim_seed=seed,
            display_env=False,
            display_agent=False,
            display_rewards=False,
        )
        evaluation.test()
        # rewards = evaluation.monitor.stats_recorder.episode_rewards_[0]
        # length = evaluation.monitor.stats_recorder.episode_lengths[0]
        rewards = evaluation.episode_rewards_
        length = evaluation.wrapped_env.length_queue[0]
        total_reward = np.sum(rewards)
        cum_discount = lambda signal: np.sum(
            [gamma ** t * signal[t] for t in range(len(signal))]
        )
        return_ = cum_discount(rewards)
        mean_return = np.mean([cum_discount(rewards[t:]) for t in range(len(rewards))])
    else:
        length = 0
        total_reward = 0
        return_ = 0
        mean_return = 0

    # Save results
    result = {
        "agent": agent_name,
        "budget": budget,
        "seed": seed,
        "total_reward": total_reward,
        "return": return_,
        "mean_return": mean_return,
        "length": length,
        "simple_regret": simple_regret,
        "gap": gap,
    }

    df = pd.DataFrame.from_records([result])
    with open(path, "a") as f:
        df.to_csv(f, sep=",", encoding="utf-8", header=f.tell() == 0, index=False)


def _get_budget(budget, budget_range):
    budgets = np.logspace(*literal_eval(budget)).astype(int)
    if budget_range:
        start, end = budget_range.split(":")
        budgets = budgets[int(start) : int(end)]
    budgets = np.unique(budgets)
    return budgets


def _get_seeds(seeds):
    seeds = seeds.split(",")
    first_seed = (
        int(seeds[0]) if len(seeds) == 2 else np.random.randint(0, SEED_MAX, dtype=int)
    )
    seeds_count = int(seeds[-1])
    seeds = (first_seed + np.arange(seeds_count)).tolist()
    return seeds


def prepare_experiments(budgets, seeds, path, budget_range, envs, agents):
    budgets = _get_budget(budgets, budget_range)
    seeds = _get_seeds(seeds)

    paths = [path]
    experiments = list(product(seeds, budgets, agents.items(), envs, paths))
    return experiments


def prepare_missing_experiments(budgets, seeds, path, budget_range, envs, agents):
    budgets = _get_budget(budgets, budget_range)
    seeds = _get_seeds(seeds)

    # Check what is missing
    expected = product(budgets, seeds, agents.keys())
    dft = pd.read_csv(path)
    dft = dft[~dft.agent.isin(["agent"])].apply(pd.to_numeric, errors="ignore")

    # experiments = list(product(seeds, budgets, agents.items(), envs, paths))
    experiments = []
    n_total = 0
    for _b, _s, _sa in expected:
        n_total += 1
        dfe = dft.query("budget == @_b and seed == @_s and agent == @_sa")
        if dfe.empty:
            for e in envs:
                experiments.append((_s, _b, (_sa, agents[_sa]), e, path))
    return experiments, n_total


def rename_df(df):
    df = df.rename(columns=latex_names)
    for key, value in latex_names.items():
        df["agent"] = df["agent"].replace(key, value)
    return df


def rename(value, latex=True):
    return latex_names.get(value, value) if latex else value


def plot_all(data_file, directory, data_range=None, legend_loc="lower right"):
    print("Reading data from {}".format(directory / data_file))
    df = pd.read_csv(str(directory / data_file))
    plot_all_df(df, directory, data_range, legend_loc)


def plot_all_df(df, directory, data_range=None, legend_loc="lower right", sufix=""):
    df = df[~df.agent.isin(["agent"])].apply(pd.to_numeric, errors="ignore")
    df = df.sort_values(by="agent")

    m = df.loc[df["simple_regret"] != np.inf, "simple_regret"].max()
    df["simple_regret"].replace(np.inf, m, inplace=True)

    if data_range:
        start, end = data_range.split(":")
        df = df[df["budget"].between(int(start), int(end))]
    print("Number of seeds found: {}".format(df.seed.nunique()))

    df = rename_df(df)
    with sns.axes_style("ticks"):
        sns.set_palette("colorblind")
        fig, ax = plt.subplots()
        for field in ["length"]:
            ax.set_xscale("symlog")
            if field in ["simple_regret"]:
                ax.set_yscale("symlog", linthreshy=1e-3)

            sns.lineplot(
                x=rename("budget"),
                y=rename(field),
                ax=ax,
                hue="agent",
                style="agent",
                data=df,
            )
            # ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,)))
            # ax.yaxis.grid(True, which='minor', linestyle='-')
            plt.legend(loc=legend_loc)

            file_name = field + sufix
            field_path = directory / "{}.pdf".format(file_name)
            fig.savefig(field_path, bbox_inches="tight")
            field_path = directory / "{}.png".format(file_name)
            fig.savefig(field_path, bbox_inches="tight")
            print("Saving {} plot to {}".format(file_name, field_path))

    custom_processing(df, directory)


def custom_processing(df, directory):
    pass


def main(
    args, envs=env_configs(), selected_agents=selected_agents(), experiment_agents=None
):
    # raise NotImplementedError('The Monitor wrapper which previously recorded statistics has been replaced by '
    #                           'RecordEpisodeStatistics, so this file needs to be updated.')
    data_file_path = Path(args["--directory"]) / args["--data_file"]
    experiments = []
    agents = (
        {
            agent: config
            for agent, config in agent_configs().items()
            if agent in selected_agents
        }
        if experiment_agents is None
        else experiment_agents
    )
    if args["--missing"] == "True":
        if not data_file_path.is_file():
            raise ArgumentError("Data file does not exists - do not know what to do!")
        experiments, n_total = prepare_missing_experiments(
            args["--budgets"],
            args["--seeds"],
            str(data_file_path),
            args["--brange"],
            envs,
            agents,
        )
    elif args["--generate"] == "True":
        if data_file_path.is_file():
            raise ArgumentError("Data file exists - will not create new generate!")

        experiments = prepare_experiments(
            args["--budgets"],
            args["--seeds"],
            str(data_file_path),
            args["--brange"],
            envs,
            agents,
        )
        n_total = len(experiments)
    if args["--test"]:
        print(
            "TEST: "
            + str(len(experiments))
            + " results out of total "
            + str(n_total)
            + " will be saved to "
            + str(data_file_path)
        )
        return
    if experiments:
        print(
            str(len(experiments)) + " results will be saved to " + str(data_file_path)
        )
        random.shuffle(experiments)
        chunksize = int(args["--chunksize"])
        with Pool(processes=int(args["--processes"])) as p:
            list(
                tqdm.tqdm(
                    p.imap_unordered(evaluate, experiments, chunksize=chunksize),
                    total=len(experiments),
                )
            )
        print("End processing for a file " + str(data_file_path))
    if args["--show"] == "True":
        plot_all(args["--data_file"], Path(args["--directory"]), args["--range"])


if __name__ == "__main__":
    # print(agent_configs())
    budget = _get_budget("3,6,60", "1:30")
    print(budget)
    print([b // 40 for b in budget])
    # arguments = docopt(__doc__)
    # main(arguments)

    # args = docopt(__doc__)
    # experiments = prepare_experiments(args["--budgets"], args['--seeds'],
    #                                       str(Path(args["--directory"]) / args["--data_file"]))
    # evaluate(experiments[67])

    # args = docopt(__doc__)
    # plot_all('data_1pendulum.csv', Path(args["--directory"]), args["--range"])
