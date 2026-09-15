"""Validate exclusive navigation outcomes without treating a stop as contact."""
import pandas as pd

OUTCOMES = ('goal', 'safety', 'collision', 'timeout')


def episode_outcomes(frame):
    """Accept recorded outcomes or four exclusive binary terminal flags."""
    flag_names = ('goal', 'safety', 'collision', 'truncated')
    flags = None
    if all(name in frame.columns for name in flag_names):
        flags = frame[list(flag_names)].apply(pd.to_numeric, errors='raise')
        if not flags.isin([0, 1]).all().all() or not flags.sum(axis=1).eq(1).all():
            raise ValueError('Expected exactly one binary terminal outcome per episode')
    if 'outcome' in frame.columns:
        result = frame['outcome'].astype(str).str.strip().str.lower()
        result = result.replace({'truncated': 'timeout'})
        if not result.isin(OUTCOMES).all():
            raise ValueError('Unknown or missing terminal outcome; cannot report rates')
        if flags is not None:
            expected = flags.idxmax(axis=1).replace({'truncated': 'timeout'})
            if not result.eq(expected).all():
                raise ValueError('Outcome text disagrees with the recorded terminal flags')
        return result
    if flags is None:
        raise ValueError('Provide outcome or all four goal/safety/collision/truncated flags')
    return flags.idxmax(axis=1).replace({'truncated': 'timeout'})


def outcome_counts(frame):
    result = episode_outcomes(frame)
    return {name: int(result.eq(name).sum()) for name in OUTCOMES}
