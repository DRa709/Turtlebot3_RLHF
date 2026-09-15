"""Consistency checks for the published aggregate tables.

These checks do not replace episode-level audit or simulator reproduction.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ORDER = ['DQN', 'DoubleDQN', 'DuelingDoubleDQN', 'RainbowDQN', 'DiscreteSAC', 'SDSAC']
TABLES = ['summary', 'seed_results', 'evaluation_curves', 'checkpoint_summary',
          'seed_summary', 'comparison_budget_metrics', 'policy_diagnostics']


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def load_audited_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        require(actual == expected, f'Provenance hash mismatch: {name}')
    tables = {name: pd.read_csv(directory / f'{name}.csv') for name in TABLES}
    validate_data(tables)
    return tables


def validate_data(tables):
    summary, seeds, curves = [tables[name] for name in TABLES[:3]]
    require(set(summary.algorithm) == set(ORDER) and len(summary) == 6,
            'Expected exactly six benchmark methods')
    for name in ORDER:
        row = summary[summary.algorithm == name].iloc[0]
        learners = seeds[seeds.algorithm == name]
        checkpoints = curves[curves.algorithm == name].sort_values('checkpoint_step')
        require(len(learners) == 5 and learners.learning_seed.nunique() == 5,
                f'{name}: expected five distinct learners')
        require(row.final_episodes == 100 and row.n_runs == 5, f'{name}: final denominator')
        require(np.isclose(row.final_success, row.final_goals / row.final_episodes),
                f'{name}: success/count mismatch')
        require(np.isclose(row.final_success, learners.final_success.mean()),
                f'{name}: learner mean mismatch')
        require(np.isclose(row.final_sd_pp, learners.final_success.std(ddof=1) * 100),
                f'{name}: learner SD mismatch')
        require(np.isclose(row.final_success + row.safety + row.collision + row.timeout, 1),
                f'{name}: exclusive outcomes must sum to one')
        require(len(checkpoints) == 20 and checkpoints.checkpoint_step.nunique() == 20,
                f'{name}: checkpoint count')
        require(np.isclose(row.final_success, checkpoints.iloc[-1].success),
                f'{name}: final curve/table mismatch')
        require(np.isclose(row.all20_success, checkpoints.success.mean()),
                f'{name}: all-checkpoint mean mismatch')

    continuation = tables['checkpoint_summary']
    require((continuation[['goal', 'safety', 'timeout', 'collision']].sum(axis=1)
             == continuation.n).all(), 'Continuation outcomes must sum to trial count')
    require(np.allclose(continuation.goal_pct, 100 * continuation.goal / continuation.n),
            'Continuation goal percentages disagree with counts')
    require(np.allclose(continuation.safety_pct, 100 * continuation.safety / continuation.n),
            'Continuation stop percentages disagree with counts')
    by_seed = tables['seed_summary']
    keys = ['study_arm', 'added_steps', 'suite', 'policy_mode']
    for key, group in by_seed.groupby(keys):
        match = continuation
        for column, value in zip(keys, key):
            match = match[match[column] == value]
        require(len(match) == 1, 'Continuation checkpoint/seed table key mismatch')
        for column in ['n', 'goal', 'safety', 'timeout', 'collision']:
            require(group[column].sum() == match.iloc[0][column],
                    f'Continuation learner aggregate mismatch: {key} / {column}')

    budgets = tables['comparison_budget_metrics'].sort_values('comparisons')
    require(budgets.comparisons.tolist() == [20, 50, 100, 150, 200], 'Budget coverage')
    require((budgets.strict_n == 22).all() and (budgets.pairs == 37).all(),
            'Common prediction-validation denominators')
    require((budgets.fitting_pairs + budgets.snapshot_validation_pairs + budgets.cannot_judge
             == budgets.comparisons).all(), 'Comparison cost accounting')
    require(np.allclose(budgets.strict_accuracy_pct,
                        100 * budgets.strict_correct / budgets.strict_n),
            'Prediction accuracy/count mismatch')
    return {'methods': 6, 'learners_per_method': 5, 'final_trials_per_method': 100,
            'comparison_budgets': budgets.comparisons.tolist(),
            'scope': 'Aggregate table consistency and recorded hashes; no new experiments'}
