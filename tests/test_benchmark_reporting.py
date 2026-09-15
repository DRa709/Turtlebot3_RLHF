"""Regression checks for contact/stop separation and result reporting."""
from pathlib import Path
import json
import sys
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from benchmarks.outcomes import episode_outcomes, outcome_counts
from benchmarks.result_data import load_results, validate_data
from benchmarks import generate_benchmark_suite as legacy
from benchmarks.build_report import report_tables

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def episode_csv(tmp_path):
    path = tmp_path / 'episodes.csv'
    pd.DataFrame({'outcome': ['goal'] * 4 + ['safety'] * 3 + ['collision'] * 2 + ['timeout'],
                  'end_env_step': list(range(1000, 11000, 1000)),
                  'length': [1000] * 10, 'return': [10] * 10}).to_csv(path, index=False)
    return path


def test_contact_and_stop_counts_are_distinct(episode_csv):
    frame = legacy.load_episodes_dataframe(str(episode_csv))
    assert outcome_counts(frame) == {'goal': 4, 'safety': 3, 'collision': 2, 'timeout': 1}
    assert frame.iloc[-1].rolling_collision == pytest.approx(20)
    assert frame.iloc[-1].rolling_safety == pytest.approx(30)


def test_saved_training_summary_separates_contacts(episode_csv, tmp_path):
    result = legacy.plot_individual_algorithm_figures(
        'discretesac', {101: {'episodes': str(episode_csv)}}, str(tmp_path / 'report'), dpi=45)
    saved = json.loads((tmp_path / 'report/individual/discretesac/discretesac_metrics_summary.json').read_text())
    assert saved['seed_breakdown']['101']['collisions'] == 2
    assert saved['seed_breakdown']['101']['safety_stops'] == 3
    assert result['metrics']['data_source'] == 'recorded_training_episodes'


def test_binary_flags_are_accepted_without_outcome_text():
    frame = pd.DataFrame({'goal': [1, 0, 0, 0], 'safety': [0, 1, 0, 0],
                          'collision': [0, 0, 1, 0], 'truncated': [0, 0, 0, 1]})
    assert episode_outcomes(frame).tolist() == ['goal', 'safety', 'collision', 'timeout']


@pytest.mark.parametrize('frame', [
    pd.DataFrame({'outcome': ['unknown']}),
    pd.DataFrame({'reward': [1]}),
    pd.DataFrame({'goal': [1], 'safety': [1], 'collision': [0], 'truncated': [0]}),
    pd.DataFrame({'goal': [0], 'safety': [0], 'collision': [0], 'truncated': [0]}),
    pd.DataFrame({'outcome': ['safety'], 'goal': [0], 'safety': [0],
                  'collision': [1], 'truncated': [0]}),
])
def test_unknown_or_conflicting_outcomes_are_rejected(frame):
    with pytest.raises(ValueError):
        episode_outcomes(frame)


def test_missing_training_axis_is_not_fabricated(tmp_path):
    path = tmp_path / 'missing_steps.csv'
    pd.DataFrame({'outcome': ['goal'], 'return': [10]}).to_csv(path, index=False)
    assert legacy.load_episodes_dataframe(str(path)) is None


def test_97_percent_is_reproduced_from_five_learners():
    tables = load_results(ROOT / 'results')
    learners = tables['seed_results'].query("algorithm == 'DiscreteSAC'").sort_values('learning_seed')
    assert (learners.final_success * 20).round().tolist() == [20, 20, 17, 20, 20]
    assert sum((learners.final_success * 20).round()) == 97
    headers, rows = report_tables(tables)['RESULTS_BENCHMARK']
    row = next(row for row in rows if row[0] == 'Discrete SAC')
    assert row[1:] == ['97/100', '97', '6.71', '3', '0', '0']


def test_altered_success_percentage_is_rejected():
    tables = load_results(ROOT / 'results')
    mask = tables['summary'].algorithm == 'DiscreteSAC'
    tables['summary'].loc[mask, 'final_success'] = 0.946
    with pytest.raises(ValueError, match='success/count mismatch'):
        validate_data(tables)


def test_continuation_cannot_merge_safety_into_contact():
    tables = load_results(ROOT / 'results')
    tables['checkpoint_summary'].loc[0, 'collision'] = 2
    with pytest.raises(ValueError, match='outcomes must sum'):
        validate_data(tables)


def test_demo_marks_standalone_figures(monkeypatch):
    monkeypatch.setattr(legacy, 'DEMO_MODE', True)
    figure = plt.figure()
    legacy.mark_figure(figure)
    assert any('SYNTHETIC DEMO - NOT EXPERIMENTAL RESULTS' in t.get_text() for t in figure.texts)
    plt.close(figure)


def test_demo_does_not_mix_real_inputs(tmp_path, monkeypatch):
    captured = {}
    def fake_demo(cache):
        (Path(cache) / 'demo_dataset').mkdir(parents=True)
    def fake_discovery(paths):
        captured['paths'] = paths
        return {'dqn': {101: {}}}
    def fake_html(results, output):
        (Path(output) / 'index.html').write_text('<html><body>Preview</body></html>')
    monkeypatch.setattr(legacy, 'generate_demo_dataset', fake_demo)
    monkeypatch.setattr(legacy, 'unpack_archives_if_needed', lambda paths, cache: [])
    monkeypatch.setattr(legacy, 'find_algorithm_runs', fake_discovery)
    monkeypatch.setattr(legacy, 'plot_individual_algorithm_figures', lambda *a, **k: {})
    monkeypatch.setattr(legacy, 'plot_comparative_suite', lambda *a, **k: None)
    monkeypatch.setattr(legacy, 'generate_interactive_html_dashboard', fake_html)
    monkeypatch.setattr(legacy, 'DEMO_MODE', False)
    monkeypatch.setattr(sys, 'argv', ['reporter', '--demo', '--input-dirs', '/real/runs',
                                    '--output-dir', str(tmp_path)])
    legacy.main()
    assert len(captured['paths']) == 1
    assert Path(captured['paths'][0]).name == 'demo_dataset'
    assert '/real/runs' not in captured['paths']
    output = tmp_path / 'synthetic_demo'
    assert 'SYNTHETIC DEMO' in (output / 'index.html').read_text()
    assert json.loads((output / 'provenance.json').read_text())['data_source'] == 'synthetic_demo'


def test_training_report_requires_explicit_inputs(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['reporter'])
    monkeypatch.setattr(legacy, 'DEMO_MODE', False)
    with pytest.raises(SystemExit) as error:
        legacy.main()
    assert error.value.code == 2
