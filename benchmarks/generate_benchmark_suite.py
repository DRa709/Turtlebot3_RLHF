#!/usr/bin/env python3
"""
TurtleBot3 Phase-1 Multi-Algorithm Benchmark Suite & Publication Figure Generator
==================================================================================
Generates publication-grade (300 DPI) individual and comparative benchmarking figures
across all Phase-1 Deep Reinforcement Learning navigation algorithms:
  - DQN
  - Double DQN
  - Dueling Double DQN
  - Rainbow DQN
  - Discrete SAC
  - SD-SAC

Features:
  1. Automated Data Ingestion:
     - Discovers cluster results, local sync folders, or zipped archives (*.zip).
     - Auto-extracts archives and normalizes heterogeneous column headers.
  2. Multi-Seed Statistical Aggregation:
     - Aggregates over seeds (e.g., 101, 202, 303, 404, 505) using uniform grid interpolation.
     - Generates fleet mean trajectories with shaded +/- 1 SD empirical ribbons.
  3. Output Organization:
     - Dedicated subdirectories for individual algorithms: individual/<algo>/
     - Dedicated comparative directory: comparative/
     - Interactive standalone HTML dashboard: index.html
     - Full metrics summary: CSV, JSON, and Markdown reports.
  4. Demo / Pending Simulation Mode:
     - Use --demo to populate synthetic comparative curves for algorithms still running.
"""

import os
import sys
import glob
import re
import json
import argparse
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Headless mode for cluster execution (ARC / Slurm)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
if __package__:
    from .outcomes import episode_outcomes
else:
    from outcomes import episode_outcomes

DEMO_MODE = False


def mark_figure(figure):
    label = ('SYNTHETIC DEMO - NOT EXPERIMENTAL RESULTS' if DEMO_MODE else
             'Training-episode statistics; not held-out policy evaluation')
    figure.text(0.5, 0.005, label, ha='center', va='bottom', fontsize=9,
                color='#a00000' if DEMO_MODE else '#333333')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Configure publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# Standardized algorithm metadata & color-blind friendly palette
ALGORITHM_CONFIGS = {
    'dqn': {
        'name': 'DQN',
        'full_name': 'Deep Q-Network',
        'color': '#1f77b4',       # Deep Blue
        'linestyle': '-',
        'type': 'value_based'
    },
    'doubledqn': {
        'name': 'DoubleDQN',
        'full_name': 'Double Deep Q-Network',
        'color': '#ff7f0e',       # Safety Orange
        'linestyle': '-',
        'type': 'value_based'
    },
    'duelingdoubledqn': {
        'name': 'DuelingDoubleDQN',
        'full_name': 'Dueling Double DQN',
        'color': '#2ca02c',       # Emerald Green
        'linestyle': '-',
        'type': 'value_based'
    },
    'rainbowdqn': {
        'name': 'RainbowDQN',
        'full_name': 'Rainbow DQN',
        'color': '#d62728',       # Crimson Red
        'linestyle': '-',
        'type': 'value_based'
    },
    'discretesac': {
        'name': 'DiscreteSAC',
        'full_name': 'Discrete Soft Actor-Critic',
        'color': '#9467bd',       # Royal Purple
        'linestyle': '-',
        'type': 'actor_critic'
    },
    'sdsac': {
        'name': 'SDSAC',
        'full_name': 'SD-SAC (categorical adaptation)',
        'color': '#8c564b',       # Sienna Brown
        'linestyle': '-',
        'type': 'actor_critic'
    }
}

DEFAULT_SEEDS = [101, 202, 303, 404, 505]


# ==============================================================================
# 1. DATA EXTRACTION & DISCOVERY
# ==============================================================================

def normalize_algo_key(text: str) -> Optional[str]:
    """Maps various naming variants to standardized algorithm keys."""
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
    mapping = {
        'duelingdoubledqn': 'duelingdoubledqn',
        'duelingddqn': 'duelingdoubledqn',
        'dueling': 'duelingdoubledqn',
        'doubledqn': 'doubledqn',
        'ddqn': 'doubledqn',
        'rainbowdqn': 'rainbowdqn',
        'rainbow': 'rainbowdqn',
        'discretesac': 'discretesac',
        'dsac': 'discretesac',
        'sdsac': 'sdsac',
        'dqn': 'dqn'
    }
    # Match longest patterns first so 'doubledqn' doesn't get caught by 'dqn'
    for k in sorted(mapping.keys(), key=len, reverse=True):
        if k in cleaned:
            return mapping[k]
    return None


def unpack_archives_if_needed(search_paths: List[str], cache_dir: str) -> List[str]:
    """Discovers and extracts *.zip and *.tar.gz archives into the cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    extracted_paths = []
    
    archive_patterns = ["*.zip", "*.tar.gz", "*.tgz"]
    found_archives = []
    for sp in search_paths:
        if os.path.isfile(sp):
            found_archives.append(sp)
        elif os.path.isdir(sp):
            for pat in archive_patterns:
                found_archives.extend(glob.glob(os.path.join(sp, pat)))
    
    for arc in set(found_archives):
        base_name = Path(arc).stem
        if base_name.endswith('.tar'):
            base_name = Path(base_name).stem
        
        # Only unpack archives related to RL benchmarks
        lower_name = base_name.lower()
        if not any(kw in lower_name for kw in ['dqn', 'sac', 'turtle', 'tb3', 'phase1', 'result', 'controlled', 'benchmark']):
            continue
        
        # Check if algorithm name is in archive
        algo_key = normalize_algo_key(base_name)
        dest_folder_name = algo_key or base_name
        dest_dir = os.path.join(cache_dir, dest_folder_name)
        
        # Unpack if destination is empty or doesn't exist
        if not os.path.exists(dest_dir) or not os.listdir(dest_dir):
            print(f"[Archive Extraction] Extracting {os.path.basename(arc)} -> {dest_dir} ...")
            os.makedirs(dest_dir, exist_ok=True)
            try:
                if arc.endswith('.zip'):
                    with zipfile.ZipFile(arc, 'r') as zf:
                        zf.extractall(dest_dir)
                elif arc.endswith(('.tar.gz', '.tgz')):
                    with tarfile.open(arc, 'r:gz') as tf:
                        tf.extractall(dest_dir)
                print(f"  Successfully extracted {os.path.basename(arc)}")
            except Exception as e:
                print(f"  Warning: failed to extract {arc}: {e}")
        
        if os.path.exists(dest_dir):
            extracted_paths.append(dest_dir)
            
    return extracted_paths


def find_algorithm_runs(search_paths: List[str]) -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    Recursively scans search paths to identify algorithm runs organized by:
    results[algo_key][seed] = {'episodes': path, 'updates': path, 'manifest': path}
    """
    results: Dict[str, Dict[int, Dict[str, str]]] = {k: {} for k in ALGORITHM_CONFIGS.keys()}
    
    for base in search_paths:
        if not os.path.exists(base):
            continue
        
        # Recursively look for episodes.csv
        for root, dirs, files in os.walk(base):
            if 'episodes.csv' in files:
                episodes_path = os.path.join(root, 'episodes.csv')
                updates_path = os.path.join(root, 'updates.csv') if 'updates.csv' in files else None
                manifest_path = os.path.join(root, 'run_identity.json') if 'run_identity.json' in files else None
                
                algo_key = None
                seed = None
                
                # Try reading run_identity.json for ground truth
                if manifest_path and os.path.exists(manifest_path):
                    try:
                        with open(manifest_path) as mf:
                            meta = json.load(mf)
                            if 'algorithm' in meta:
                                algo_key = normalize_algo_key(meta['algorithm'])
                            if 'learning_seed' in meta:
                                seed = int(meta['learning_seed'])
                    except Exception:
                        pass
                
                # Path heuristics fallback
                path_parts = Path(root).parts
                if not algo_key:
                    for part in reversed(path_parts):
                        ak = normalize_algo_key(part)
                        if ak:
                            algo_key = ak
                            break
                            
                if not seed:
                    for part in path_parts:
                        m = re.search(r'seed_?(\d+)', part, re.IGNORECASE)
                        if m:
                            seed = int(m.group(1))
                            break
                            
                if algo_key and seed:
                    if algo_key not in results:
                        results[algo_key] = {}
                    
                    # If multiple jobs exist, keep the one with largest file size or latest mtime
                    if seed in results[algo_key]:
                        existing_size = os.path.getsize(results[algo_key][seed]['episodes'])
                        new_size = os.path.getsize(episodes_path)
                        if new_size > existing_size:
                            results[algo_key][seed] = {
                                'episodes': episodes_path,
                                'updates': updates_path,
                                'manifest': manifest_path,
                                'dir': root
                            }
                    else:
                        results[algo_key][seed] = {
                            'episodes': episodes_path,
                            'updates': updates_path,
                            'manifest': manifest_path,
                            'dir': root
                        }

    # Prune empty algorithms
    return {k: v for k, v in results.items() if len(v) > 0}


# ==============================================================================
# 2. DATA PROCESSING & ROLLING AGGREGATION
# ==============================================================================

def load_episodes_dataframe(csv_path: str) -> Optional[pd.DataFrame]:
    """Loads and computes rolling metrics for an episode CSV."""
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
            
        # Determine cumulative environment steps
        if 'end_env_step' in df.columns:
            df['cum_steps'] = df['end_env_step']
        elif 'total_steps' in df.columns:
            df['cum_steps'] = df['total_steps']
        elif 'cum_steps' in df.columns:
            pass
        elif 'length' in df.columns:
            df['cum_steps'] = df['length'].cumsum()
        else:
            raise ValueError('Missing recorded step counts; refusing to invent a training axis')
            
        # Standardize return column
        if 'return' not in df.columns:
            if 'reward' in df.columns:
                df['return'] = df['reward']
            elif 'total_reward' in df.columns:
                df['return'] = df['total_reward']
            else:
                df['return'] = np.nan  # Unrecorded reward is not a zero return.
                
        # Keep physical contacts distinct from proximity-triggered safety stops.
        df['outcome'] = episode_outcomes(df)
            
        # Standardize length
        if 'length' not in df.columns:
            if 'steps' in df.columns:
                df['length'] = df['steps']
            else:
                df['length'] = np.nan  # Missing episode length remains missing.
                
        # Rolling averages (window = 50 episodes)
        df['rolling_return'] = df['return'].rolling(50, min_periods=5).mean()
        df['rolling_success'] = (df['outcome'] == 'goal').rolling(50, min_periods=5).mean() * 100.0
        df['rolling_length'] = df['length'].rolling(50, min_periods=5).mean()
        df['rolling_collision'] = (df['outcome'] == 'collision').rolling(50, min_periods=5).mean() * 100.0
        df['rolling_safety'] = (df['outcome'] == 'safety').rolling(50, min_periods=5).mean() * 100.0
        
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def interpolate_algorithm_fleet(
    seed_dfs: Dict[int, pd.DataFrame],
    eval_steps: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolates rolling metrics onto a uniform step grid to compute fleet mean and std dev."""
    returns_interp = []
    success_interp = []
    length_interp = []
    
    for df in seed_dfs.values():
        x = df['cum_steps'].values
        r = df['rolling_return'].bfill().fillna(0).values
        s = df['rolling_success'].bfill().fillna(0).values
        l = df['rolling_length'].bfill().fillna(200).values
        
        # Sort if necessary
        sort_idx = np.argsort(x)
        x_s, r_s, s_s, l_s = x[sort_idx], r[sort_idx], s[sort_idx], l[sort_idx]
        
        returns_interp.append(np.interp(eval_steps, x_s, r_s))
        success_interp.append(np.interp(eval_steps, x_s, s_s))
        length_interp.append(np.interp(eval_steps, x_s, l_s))
        
    r_mean = np.mean(returns_interp, axis=0)
    r_std = np.std(returns_interp, axis=0)
    s_mean = np.mean(success_interp, axis=0)
    s_std = np.std(success_interp, axis=0)
    l_mean = np.mean(length_interp, axis=0)
    l_std = np.std(length_interp, axis=0)
    
    return r_mean, r_std, s_mean, s_std, l_mean, l_std


# ==============================================================================
# 3. INDIVIDUAL ALGORITHM FIGURE GENERATOR
# ==============================================================================

def plot_individual_algorithm_figures(
    algo_key: str,
    seed_data: Dict[int, Dict[str, str]],
    output_dir: str,
    dpi: int = 300
) -> Dict[str, Any]:
    """Generates the 2 publication figures and statistical summary for a single algorithm."""
    algo_meta = ALGORITHM_CONFIGS.get(algo_key, {
        'name': algo_key.upper(),
        'full_name': algo_key.upper(),
        'color': '#333333'
    })
    algo_name = algo_meta['name']
    algo_full_name = algo_meta['full_name']
    primary_color = algo_meta['color']
    
    algo_out_dir = os.path.join(output_dir, "individual", algo_key)
    os.makedirs(algo_out_dir, exist_ok=True)
    
    # 1. Load episode data across all seeds
    seed_dfs = {}
    for seed, paths in sorted(seed_data.items()):
        df = load_episodes_dataframe(paths['episodes'])
        if df is not None:
            seed_dfs[seed] = df
            
    if not seed_dfs:
        print(f"Warning: No valid episode data found for {algo_name}")
        return {}
        
    # Multi-seed colors
    seed_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Step grid for interpolation
    max_steps_per_seed = [df['cum_steps'].max() for df in seed_dfs.values()]
    eval_max = min(max_steps_per_seed) if len(max_steps_per_seed) > 1 else max(max_steps_per_seed)
    eval_steps = np.linspace(5000, max(eval_max, 10000), 250)
    eval_k = eval_steps / 1000.0
    
    r_mean, r_std, s_mean, s_std, l_mean, l_std = interpolate_algorithm_fleet(seed_dfs, eval_steps)
    
    # --------------------------------------------------------------------------
    # FIGURE 1: 4-PANEL BENCHMARK PERFORMANCE CURVES
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    
    for idx, (seed, df) in enumerate(seed_dfs.items()):
        c = seed_palette[idx % len(seed_palette)]
        x_k = df['cum_steps'] / 1000.0
        label = f"Seed {seed}"
        axes1[0, 0].plot(x_k, df['rolling_return'], label=label, color=c, alpha=0.6, linewidth=1.4)
        axes1[0, 1].plot(x_k, df['rolling_success'], label=label, color=c, alpha=0.6, linewidth=1.4)
        axes1[1, 0].plot(x_k, df['rolling_length'], label=label, color=c, alpha=0.6, linewidth=1.4)
        
    # Panel A: Return
    axes1[0, 0].plot(eval_k, r_mean, color='black', linewidth=2.5, label='Fleet Mean')
    axes1[0, 0].fill_between(eval_k, r_mean - r_std, r_mean + r_std, color=primary_color, alpha=0.2, label='±1 Std Dev')
    axes1[0, 0].axhline(100.0, color='#27ae60', linestyle='--', linewidth=1.8, label='Goal Reward (+100)')
    axes1[0, 0].set_title('A. Episode Return Progression (Rolling 50 Eps)', fontsize=12, fontweight='bold')
    axes1[0, 0].set_xlabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
    axes1[0, 0].set_ylabel('Mean Episode Return', fontsize=10, fontweight='bold')
    axes1[0, 0].legend(loc='lower right', fontsize=8, frameon=True)
    axes1[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Panel B: Success Rate
    axes1[0, 1].plot(eval_k, s_mean, color='black', linewidth=2.5, label='Fleet Mean')
    axes1[0, 1].fill_between(eval_k, s_mean - s_std, s_mean + s_std, color=primary_color, alpha=0.2, label='±1 Std Dev')
    axes1[0, 1].axhline(80.0, color='#e67e22', linestyle=':', linewidth=1.5, label='80% Milestone')
    axes1[0, 1].axhline(90.0, color='#27ae60', linestyle='--', linewidth=1.5, label='90% Target')
    axes1[0, 1].set_title('B. Goal Success Rate % (Rolling 50 Eps)', fontsize=12, fontweight='bold')
    axes1[0, 1].set_xlabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
    axes1[0, 1].set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    axes1[0, 1].set_ylim(-5, 105)
    axes1[0, 1].legend(loc='lower right', fontsize=8, frameon=True)
    axes1[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # Panel C: Episode Length
    axes1[1, 0].plot(eval_k, l_mean, color='black', linewidth=2.5, label='Fleet Mean')
    axes1[1, 0].fill_between(eval_k, l_mean - l_std, l_mean + l_std, color=primary_color, alpha=0.2, label='±1 Std Dev')
    axes1[1, 0].set_title('C. Episode Length / Steps to Termination', fontsize=12, fontweight='bold')
    axes1[1, 0].set_xlabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
    axes1[1, 0].set_ylabel('Steps per Episode', fontsize=10, fontweight='bold')
    axes1[1, 0].legend(loc='upper right', fontsize=8, frameon=True)
    axes1[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Panel D: Outcome Bar Chart
    overall_sr = [(df['outcome'] == 'goal').mean() * 100.0 for df in seed_dfs.values()]
    recent_sr = [(df['outcome'].tail(100) == 'goal').mean() * 100.0 for df in seed_dfs.values()]
    collision_sr = [(df['outcome'] == 'collision').mean() * 100.0 for df in seed_dfs.values()]
    safety_sr = [(df['outcome'] == 'safety').mean() * 100.0 for df in seed_dfs.values()]
    timeout_sr = [(df['outcome'] == 'timeout').mean() * 100.0 for df in seed_dfs.values()]
    
    cats = ['Lifetime Goal %', 'Recent Goal %', 'Safety Stop %', 'Contact %', 'Timeout %']
    cat_means = [np.mean(overall_sr), np.mean(recent_sr), np.mean(safety_sr), np.mean(collision_sr), np.mean(timeout_sr)]
    cat_stds = [np.std(overall_sr), np.std(recent_sr), np.std(safety_sr), np.std(collision_sr), np.std(timeout_sr)]
    bar_colors = ['#2980b9', '#27ae60', '#8e44ad', '#c0392b', '#f39c12']
    
    bars = axes1[1, 1].bar(cats, cat_means, yerr=cat_stds, capsize=6, color=bar_colors, alpha=0.85, edgecolor='black', width=0.55)
    axes1[1, 1].set_title('D. Empirical Outcome Distributions (Fleet Mean ± Std)', fontsize=12, fontweight='bold')
    axes1[1, 1].set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    axes1[1, 1].set_ylim(0, 115)
    axes1[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    for bar in bars:
        h = bar.get_height()
        axes1[1, 1].text(bar.get_x() + bar.get_width()/2.0, h + 3.0, f'{h:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    max_step_fleet = max(df['cum_steps'].max() for df in seed_dfs.values())
    fig1.suptitle(f'{algo_full_name} Training & Policy Convergence Dashboard ({len(seed_dfs)} Seeds, ~{int(max_step_fleet/1000)}k Steps)', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    fig1_path = os.path.join(algo_out_dir, f"{algo_key}_benchmark_curves.png")
    mark_figure(fig1)
    fig1.savefig(fig1_path, dpi=dpi)
    plt.close(fig1)
    
    # --------------------------------------------------------------------------
    # FIGURE 2: 4-PANEL OPTIMIZATION & VALUE DIAGNOSTICS
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    has_updates = False
    
    for idx, (seed, paths) in enumerate(sorted(seed_data.items())):
        upd_path = paths.get('updates')
        if not upd_path or not os.path.exists(upd_path):
            continue
            
        try:
            # Read every 25th row for memory efficiency
            df_upd = pd.read_csv(upd_path)
            if len(df_upd) == 0:
                continue
            df_upd = df_upd.iloc[::25].copy()
            has_updates = True
            
            c = seed_palette[idx % len(seed_palette)]
            x_k = (df_upd['env_step'] if 'env_step' in df_upd.columns else df_upd['gradient_step'] * 4) / 1000.0
            s_label = f"Seed {seed}"
            
            # Loss (DQN loss, SAC critic_loss_mean / critic1_loss / actor_loss)
            loss_col = None
            for cand in ['loss', 'critic_loss_mean', 'critic1_loss', 'critic_loss', 'actor_loss']:
                if cand in df_upd.columns:
                    loss_col = cand
                    break
            if loss_col:
                smooth_loss = df_upd[loss_col].rolling(20, min_periods=2).mean()
                axes2[0, 0].plot(x_k, smooth_loss, label=s_label, color=c, alpha=0.7, linewidth=1.4)
                
            # TD error
            td_col = None
            for cand in ['td_error_abs_mean', 'td_error', 'abs_td_error']:
                if cand in df_upd.columns:
                    td_col = cand
                    break
            if td_col:
                smooth_td = df_upd[td_col].rolling(20, min_periods=2).mean()
                axes2[0, 1].plot(x_k, smooth_td, label=s_label, color=c, alpha=0.7, linewidth=1.4)
                
            # Q-values / Value estimate
            q_col = None
            for cand in ['q_taken_mean', 'q1_taken_mean', 'q1_mean', 'min_q_policy_mean', 'q_mean']:
                if cand in df_upd.columns:
                    q_col = cand
                    break
            if q_col:
                smooth_q = df_upd[q_col].rolling(20, min_periods=2).mean()
                axes2[1, 0].plot(x_k, smooth_q, label=s_label, color=c, alpha=0.7, linewidth=1.4)
                
            # Exploration / Alpha / Entropy
            if 'epsilon' in df_upd.columns and idx == 0:
                axes2[1, 1].plot(x_k, df_upd['epsilon'], color='#2c3e50', linewidth=2.2, label='Epsilon Schedule (ε)')
            elif 'alpha' in df_upd.columns:
                axes2[1, 1].plot(x_k, df_upd['alpha'], label=f"Alpha (s{seed})", color=c, alpha=0.7, linewidth=1.4)
            elif 'policy_entropy_mean' in df_upd.columns:
                smooth_ent = df_upd['policy_entropy_mean'].rolling(20, min_periods=2).mean()
                axes2[1, 1].plot(x_k, smooth_ent, label=f"Entropy (s{seed})", color=c, alpha=0.7, linewidth=1.4)
            elif 'entropy' in df_upd.columns:
                axes2[1, 1].plot(x_k, df_upd['entropy'], label=f"Entropy (s{seed})", color=c, alpha=0.7, linewidth=1.4)
        except Exception as e:
            print(f"Error parsing updates for {algo_name} seed {seed}: {e}")
            
    if has_updates:
        axes2[0, 0].set_title('A. Training Loss (Smoothed)', fontsize=12, fontweight='bold')
        axes2[0, 0].set_xlabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
        axes2[0, 0].set_ylabel('Loss', fontsize=10, fontweight='bold')
        if axes2[0, 0].get_legend_handles_labels()[0]:
            axes2[0, 0].legend(loc='upper right', fontsize=8)
        axes2[0, 0].grid(True, linestyle=':', alpha=0.6)
        
        axes2[0, 1].set_title('B. Mean Absolute TD Error |Q(s,a) - Target|', fontsize=12, fontweight='bold')
        axes2[0, 1].set_xlabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
        axes2[0, 1].set_ylabel('|TD Error|', fontsize=10, fontweight='bold')
        if axes2[0, 1].get_legend_handles_labels()[0]:
            axes2[0, 1].legend(loc='upper right', fontsize=8)
        axes2[0, 1].grid(True, linestyle=':', alpha=0.6)
        
        axes2[1, 0].set_title('C. Mean Action-Value Q(s, a) / Critic Estimate', fontsize=12, fontweight='bold')
        axes2[1, 0].set_xlabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
        axes2[1, 0].set_ylabel('Mean Value', fontsize=10, fontweight='bold')
        if axes2[1, 0].get_legend_handles_labels()[0]:
            axes2[1, 0].legend(loc='lower right', fontsize=8)
        axes2[1, 0].grid(True, linestyle=':', alpha=0.6)
        
        axes2[1, 1].set_title('D. Exploration / Temperature / Entropy', fontsize=12, fontweight='bold')
        axes2[1, 1].set_xlabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
        axes2[1, 1].set_ylabel('Parameter Value', fontsize=10, fontweight='bold')
        if axes2[1, 1].get_legend_handles_labels()[0]:
            axes2[1, 1].legend(loc='upper right', fontsize=8)
        axes2[1, 1].grid(True, linestyle=':', alpha=0.6)
        
        fig2.suptitle(f'{algo_full_name} Optimization & Value Function Diagnostics', fontsize=14, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        fig2_path = os.path.join(algo_out_dir, f"{algo_key}_training_diagnostics.png")
        mark_figure(fig2)
        fig2.savefig(fig2_path, dpi=dpi)
        plt.close(fig2)
    else:
        plt.close(fig2)
        fig2_path = None
        
    # --------------------------------------------------------------------------
    # METRICS SUMMARY (JSON & MARKDOWN)
    # --------------------------------------------------------------------------
    metrics = {
        'data_source': 'synthetic_demo' if DEMO_MODE else 'recorded_training_episodes',
        'algorithm': algo_name,
        'algorithm_full': algo_full_name,
        'num_seeds': len(seed_dfs),
        'seeds': list(seed_dfs.keys()),
        'total_transitions': int(sum(df['cum_steps'].max() for df in seed_dfs.values())),
        'total_episodes': int(sum(len(df) for df in seed_dfs.values())),
        'fleet_lifetime_goal_pct': float(np.mean(overall_sr)),
        'fleet_lifetime_goal_pct_std': float(np.std(overall_sr)),
        'fleet_recent_goal_pct': float(np.mean(recent_sr)),
        'fleet_recent_goal_pct_std': float(np.std(recent_sr)),
        'fleet_recent_return': float(np.mean([df['return'].tail(100).mean() for df in seed_dfs.values()])),
        'fleet_recent_return_std': float(np.std([df['return'].tail(100).mean() for df in seed_dfs.values()])),
        'seed_breakdown': {}
    }
    
    for seed, df in seed_dfs.items():
        metrics['seed_breakdown'][seed] = {
            'episodes': int(len(df)),
            'max_steps': int(df['cum_steps'].max()),
            'goals': int((df['outcome'] == 'goal').sum()),
            'collisions': int((df['outcome'] == 'collision').sum()),
            'safety_stops': int((df['outcome'] == 'safety').sum()),
            'timeouts': int((df['outcome'] == 'timeout').sum()),
            'recent_100_goal_pct': float((df['outcome'].tail(100) == 'goal').mean() * 100.0),
            'recent_100_return': float(df['return'].tail(100).mean()),
            'recent_100_return_std': float(df['return'].tail(100).std())
        }
        
    json_path = os.path.join(algo_out_dir, f"{algo_key}_metrics_summary.json")
    with open(json_path, 'w') as jf:
        json.dump(metrics, jf, indent=2, cls=NumpyEncoder)
        
    md_path = os.path.join(algo_out_dir, f"{algo_key}_summary.md")
    with open(md_path, 'w', encoding='utf-8') as mf:
        mf.write(f"# {algo_full_name} Benchmark Summary\n\n")
        mf.write(f"Data source: **{metrics['data_source']}**. Training-episode statistics, not held-out evaluation.\n\n")
        mf.write(f"- **Total Transitions**: {metrics['total_transitions']:,}\n")
        mf.write(f"- **Completed Episodes**: {metrics['total_episodes']:,}\n")
        mf.write(f"- **Fleet Mean Recent Goal SR**: {metrics['fleet_recent_goal_pct']:.1f}% ± {metrics['fleet_recent_goal_pct_std']:.1f}%\n")
        mf.write(f"- **Fleet Mean Recent Return**: {metrics['fleet_recent_return']:.2f} ± {metrics['fleet_recent_return_std']:.2f}\n\n")
        mf.write("### Per-Seed Breakdown\n\n")
        mf.write("| Seed | Episodes | Max Steps | Lifetime Goal % | Recent 100 Goal % | Recent 100 Return |\n")
        mf.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for seed, s_info in metrics['seed_breakdown'].items():
            g_pct = (s_info['goals'] / s_info['episodes']) * 100.0 if s_info['episodes'] > 0 else 0
            mf.write(f"| {seed} | {s_info['episodes']:,} | {s_info['max_steps']:,} | {g_pct:.1f}% | {s_info['recent_100_goal_pct']:.1f}% | {s_info['recent_100_return']:.2f} ± {s_info['recent_100_return_std']:.2f} |\n")
            
    print(f"Generated individual figures & metrics for {algo_name} -> {algo_out_dir}")
    return {
        'algo_key': algo_key,
        'metrics': metrics,
        'fig1_path': fig1_path,
        'fig2_path': fig2_path,
        'eval_steps': eval_steps,
        'r_mean': r_mean,
        'r_std': r_std,
        's_mean': s_mean,
        's_std': s_std,
        'l_mean': l_mean,
        'l_std': l_std
    }


# ==============================================================================
# 4. MULTI-ALGORITHM COMPARATIVE SUITE
# ==============================================================================

def plot_comparative_suite(
    algo_results: Dict[str, Dict[str, Any]],
    output_dir: str,
    dpi: int = 300
) -> None:
    """Generates cross-algorithm comparative figures and master comparative report."""
    comp_dir = os.path.join(output_dir, "comparative")
    os.makedirs(comp_dir, exist_ok=True)
    
    if len(algo_results) == 0:
        print("No algorithm data available for comparative plotting.")
        return
        
    print(f"\n[Comparative Suite] Generating cross-algorithm figures for {len(algo_results)} algorithms...")
    
    # --------------------------------------------------------------------------
    # FIGURE 1: 4-PANEL COMPARATIVE LEARNING CURVES
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12), dpi=dpi)
    
    for algo_key, res in algo_results.items():
        algo_cfg = ALGORITHM_CONFIGS.get(algo_key, {'name': algo_key.upper(), 'color': '#333333'})
        name = algo_cfg['name']
        c = algo_cfg['color']
        
        eval_k = res['eval_steps'] / 1000.0
        r_mean, r_std = res['r_mean'], res['r_std']
        s_mean, s_std = res['s_mean'], res['s_std']
        l_mean, l_std = res['l_mean'], res['l_std']
        
        # Panel A: Return
        axes1[0, 0].plot(eval_k, r_mean, label=name, color=c, linewidth=2.4)
        axes1[0, 0].fill_between(eval_k, r_mean - r_std, r_mean + r_std, color=c, alpha=0.15)
        
        # Panel B: Goal Success Rate %
        axes1[0, 1].plot(eval_k, s_mean, label=name, color=c, linewidth=2.4)
        axes1[0, 1].fill_between(eval_k, s_mean - s_std, s_mean + s_std, color=c, alpha=0.15)
        
        # Panel C: Episode Length
        axes1[1, 0].plot(eval_k, l_mean, label=name, color=c, linewidth=2.4)
        axes1[1, 0].fill_between(eval_k, l_mean - l_std, l_mean + l_std, color=c, alpha=0.15)

    # Format Panel A
    axes1[0, 0].axhline(100.0, color='#27ae60', linestyle='--', linewidth=1.5, label='Goal Reward (+100)')
    axes1[0, 0].set_title('A. Comparative Episodic Return (Fleet Mean ± 1 SD)', fontsize=13, fontweight='bold')
    axes1[0, 0].set_xlabel('Environment Steps (x1,000)', fontsize=11, fontweight='bold')
    axes1[0, 0].set_ylabel('Mean Return', fontsize=11, fontweight='bold')
    axes1[0, 0].legend(loc='lower right', fontsize=9, frameon=True)
    axes1[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Format Panel B
    axes1[0, 1].axhline(80.0, color='#e67e22', linestyle=':', linewidth=1.5, label='80% Milestone')
    axes1[0, 1].axhline(90.0, color='#27ae60', linestyle='--', linewidth=1.5, label='90% Target')
    axes1[0, 1].set_title('B. Comparative Goal Success Rate % (Rolling 50 Eps)', fontsize=13, fontweight='bold')
    axes1[0, 1].set_xlabel('Environment Steps (x1,000)', fontsize=11, fontweight='bold')
    axes1[0, 1].set_ylabel('Success Rate (%)', fontsize=11, fontweight='bold')
    axes1[0, 1].set_ylim(-5, 105)
    axes1[0, 1].legend(loc='lower right', fontsize=9, frameon=True)
    axes1[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # Format Panel C
    axes1[1, 0].set_title('C. Comparative Navigation Efficiency (Steps to Termination)', fontsize=13, fontweight='bold')
    axes1[1, 0].set_xlabel('Environment Steps (x1,000)', fontsize=11, fontweight='bold')
    axes1[1, 0].set_ylabel('Episode Steps', fontsize=11, fontweight='bold')
    axes1[1, 0].legend(loc='upper right', fontsize=9, frameon=True)
    axes1[1, 0].grid(True, linestyle=':', alpha=0.6)
    
    # Format Panel D: Grouped Outcome Bar Chart
    algo_names = [ALGORITHM_CONFIGS.get(k, {}).get('name', k) for k in algo_results.keys()]
    x_indices = np.arange(len(algo_names))
    width = 0.17
    
    recent_goals = [res['metrics']['fleet_recent_goal_pct'] for res in algo_results.values()]
    recent_goals_std = [res['metrics']['fleet_recent_goal_pct_std'] for res in algo_results.values()]
    
    lifetime_goals = [res['metrics']['fleet_lifetime_goal_pct'] for res in algo_results.values()]
    lifetime_goals_std = [res['metrics']['fleet_lifetime_goal_pct_std'] for res in algo_results.values()]
    
    collisions = []
    safety_stops = []
    timeouts = []
    for res in algo_results.values():
        total_eps = res['metrics']['total_episodes']
        col_count = sum(s['collisions'] for s in res['metrics']['seed_breakdown'].values())
        stop_count = sum(s['safety_stops'] for s in res['metrics']['seed_breakdown'].values())
        to_count = sum(s['timeouts'] for s in res['metrics']['seed_breakdown'].values())
        collisions.append((col_count / total_eps * 100.0) if total_eps > 0 else 0)
        safety_stops.append((stop_count / total_eps * 100.0) if total_eps > 0 else 0)
        timeouts.append((to_count / total_eps * 100.0) if total_eps > 0 else 0)
        
    b1 = axes1[1, 1].bar(x_indices - width*2, lifetime_goals, width, label='Lifetime Goal %', color='#2980b9', alpha=0.85)
    b2 = axes1[1, 1].bar(x_indices - width, recent_goals, width, yerr=recent_goals_std, capsize=4, label='Recent 100-Ep Goal %', color='#27ae60', alpha=0.85)
    b3 = axes1[1, 1].bar(x_indices + width, collisions, width, label='Physical Contact %', color='#c0392b', alpha=0.85)
    axes1[1, 1].bar(x_indices, safety_stops, width, label='Safety Stop %', color='#8e44ad', alpha=0.85)
    b4 = axes1[1, 1].bar(x_indices + width*2, timeouts, width, label='Timeout %', color='#f39c12', alpha=0.85)
    
    axes1[1, 1].set_xticks(x_indices)
    axes1[1, 1].set_xticklabels(algo_names, fontweight='bold', fontsize=10)
    axes1[1, 1].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    axes1[1, 1].set_ylim(0, 115)
    axes1[1, 1].set_title('D. Outcome Distributions Across Algorithms', fontsize=13, fontweight='bold')
    axes1[1, 1].legend(loc='upper right', fontsize=8, frameon=True)
    axes1[1, 1].grid(True, linestyle=':', alpha=0.6)
    
    fig1.suptitle('TurtleBot3 Phase-1 Multi-Algorithm Benchmark Comparison', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    comp_fig1_path = os.path.join(comp_dir, "comparative_learning_curves.png")
    mark_figure(fig1)
    fig1.savefig(comp_fig1_path, dpi=dpi)
    plt.close(fig1)
    
    # --------------------------------------------------------------------------
    # FIGURE 2: 4-PANEL SAMPLE EFFICIENCY & PERFORMANCE TRADEOFFS
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10), dpi=dpi)
    
    # Panel A: Steps to First 80% Success Rate
    steps_to_80 = []
    for algo_key, res in algo_results.items():
        s_curve = res['s_mean']
        steps = res['eval_steps']
        idx_80 = np.where(s_curve >= 80.0)[0]
        if len(idx_80) > 0:
            steps_to_80.append(steps[idx_80[0]] / 1000.0)
        else:
            steps_to_80.append(np.nan)
            
    algo_colors = [ALGORITHM_CONFIGS.get(k, {}).get('color', '#333333') for k in algo_results.keys()]
    
    valid_steps = [s if not np.isnan(s) else max(item['eval_steps'])/1000.0
                   for s, item in zip(steps_to_80, algo_results.values())]
    bars_eff = axes2[0, 0].bar(algo_names, valid_steps, color=algo_colors, alpha=0.85, edgecolor='black', width=0.55)
    axes2[0, 0].set_title('A. First 80% Crossing (Interpolated Training Curve)', fontsize=12, fontweight='bold')
    axes2[0, 0].set_ylabel('Environment Steps (x1,000)', fontsize=10, fontweight='bold')
    axes2[0, 0].grid(True, linestyle=':', alpha=0.6)
    for idx, bar in enumerate(bars_eff):
        val = steps_to_80[idx]
        txt = f"{val:.0f}k" if not np.isnan(val) else "Not reached"
        axes2[0, 0].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 3.0, txt, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    # Panel B: Recent 100-Episode Mean Return
    returns = [res['metrics']['fleet_recent_return'] for res in algo_results.values()]
    returns_std = [res['metrics']['fleet_recent_return_std'] for res in algo_results.values()]
    bars_ret = axes2[0, 1].bar(algo_names, returns, yerr=returns_std, capsize=6, color=algo_colors, alpha=0.85, edgecolor='black', width=0.55)
    axes2[0, 1].axhline(100.0, color='#27ae60', linestyle='--', linewidth=1.5, label='Goal Reward (+100)')
    axes2[0, 1].set_title('B. Recent Training Return (100 Episodes)', fontsize=12, fontweight='bold')
    axes2[0, 1].set_ylabel('Mean Return', fontsize=10, fontweight='bold')
    axes2[0, 1].legend(loc='lower right', fontsize=8)
    axes2[0, 1].grid(True, linestyle=':', alpha=0.6)
    for bar in bars_ret:
        h = bar.get_height()
        axes2[0, 1].text(bar.get_x() + bar.get_width()/2.0, h + 2.5, f"{h:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Panel C: Final Goal Success Rate %
    bars_sr = axes2[1, 0].bar(algo_names, recent_goals, yerr=recent_goals_std, capsize=6, color=algo_colors, alpha=0.85, edgecolor='black', width=0.55)
    axes2[1, 0].set_title('C. Recent Training Goal Rate (%)', fontsize=12, fontweight='bold')
    axes2[1, 0].set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold')
    axes2[1, 0].set_ylim(0, 115)
    axes2[1, 0].grid(True, linestyle=':', alpha=0.6)
    for bar in bars_sr:
        h = bar.get_height()
        axes2[1, 0].text(bar.get_x() + bar.get_width()/2.0, h + 3.0, f"{h:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    # Panel D: Safety vs Goal Pareto Frontier
    for idx, algo_key in enumerate(algo_results.keys()):
        c = algo_colors[idx]
        name = algo_names[idx]
        g = recent_goals[idx]
        col = collisions[idx]
        axes2[1, 1].scatter(col, g, color=c, s=180, edgecolor='black', zorder=5, label=name)
        axes2[1, 1].annotate(name, (col, g), textcoords="offset points", xytext=(8, 5), fontweight='bold', fontsize=9)
        
    axes2[1, 1].set_title('D. Recent Goal Rate vs Lifetime Contact Rate', fontsize=12, fontweight='bold')
    axes2[1, 1].set_xlabel('Physical Contact Rate (%)', fontsize=10, fontweight='bold')
    axes2[1, 1].set_ylabel('Goal Success Rate (%) [Higher is Better]', fontsize=10, fontweight='bold')
    axes2[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes2[1, 1].legend(loc='lower left', fontsize=8)
    
    fig2.suptitle('TurtleBot3 Training Statistics and First Threshold Crossing', fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    comp_fig2_path = os.path.join(comp_dir, "comparative_sample_efficiency.png")
    mark_figure(fig2)
    fig2.savefig(comp_fig2_path, dpi=dpi)
    plt.close(fig2)
    
    # --------------------------------------------------------------------------
    # MASTER COMPARISON CSV & MARKDOWN REPORT
    # --------------------------------------------------------------------------
    summary_rows = []
    for algo_key, res in algo_results.items():
        m = res['metrics']
        summary_rows.append({
            'Algorithm': m['algorithm_full'],
            'Key': algo_key,
            'Seeds': m['num_seeds'],
            'Total Steps': m['total_transitions'],
            'Total Episodes': m['total_episodes'],
            'Recent Goal SR %': f"{m['fleet_recent_goal_pct']:.1f} ± {m['fleet_recent_goal_pct_std']:.1f}",
            'Recent Return': f"{m['fleet_recent_return']:.2f} ± {m['fleet_recent_return_std']:.2f}",
            'Lifetime Goal %': f"{m['fleet_lifetime_goal_pct']:.1f} ± {m['fleet_lifetime_goal_pct_std']:.1f}",
            'Safety Stops': sum(s['safety_stops'] for s in m['seed_breakdown'].values()),
            'Physical Contacts': sum(s['collisions'] for s in m['seed_breakdown'].values()),
            'Data Source': m['data_source'],
        })
        
    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(comp_dir, "all_algorithms_metrics_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    
    # Format markdown table manually to avoid tabulate dependency
    cols = list(df_summary.columns)
    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    tbl_lines = [hdr, sep]
    for _, r in df_summary.iterrows():
        tbl_lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    tbl_str = "\n".join(tbl_lines)
    
    md_report_path = os.path.join(comp_dir, "all_algorithms_benchmark_report.md")
    with open(md_report_path, 'w', encoding='utf-8') as mf:
        mf.write("# 🏆 TurtleBot3 Phase-1 Multi-Algorithm Benchmark Comparison\n\n")
        mf.write(("SYNTHETIC DEMO - NOT EXPERIMENTAL RESULTS.\n\n" if DEMO_MODE else "Recorded training-episode statistics, not held-out evaluation.\n\n"))
        mf.write(tbl_str)
        mf.write("\n\n### Generated Comparative Visualizations:\n")
        mf.write(f"- [Learning Curves]({os.path.basename(comp_fig1_path)})\n")
        mf.write(f"- [Sample Efficiency & Pareto Tradeoffs]({os.path.basename(comp_fig2_path)})\n")
        
    print(f"Saved comparative figures and reports to -> {comp_dir}")


# ==============================================================================
# 5. STANDALONE INTERACTIVE HTML DASHBOARD
# ==============================================================================

def generate_interactive_html_dashboard(
    algo_results: Dict[str, Dict[str, Any]],
    output_dir: str
) -> str:
    """Creates a responsive, modern HTML gallery dashboard for local browser inspection."""
    html_path = os.path.join(output_dir, "index.html")
    
    algo_tabs_html = ""
    algo_sections_html = ""
    
    for idx, (algo_key, res) in enumerate(algo_results.items()):
        m = res['metrics']
        name = m['algorithm']
        full_name = m['algorithm_full']
        active_class = "active" if idx == 0 else ""
        
        fig1_rel = f"individual/{algo_key}/{algo_key}_benchmark_curves.png"
        fig2_rel = f"individual/{algo_key}/{algo_key}_training_diagnostics.png"
        
        algo_tabs_html += f"""
        <button class="tab-btn {active_class}" onclick="switchTab('{algo_key}')">{name}</button>
        """
        
        algo_sections_html += f"""
        <div id="tab-{algo_key}" class="tab-content {active_class}">
            <div class="header-card">
                <h2>{full_name} ({name})</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Recent Goal Success Rate</div>
                        <div class="metric-value">{m['fleet_recent_goal_pct']:.1f}% <span class="metric-std">±{m['fleet_recent_goal_pct_std']:.1f}%</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Recent Mean Return</div>
                        <div class="metric-value">{m['fleet_recent_return']:.2f} <span class="metric-std">±{m['fleet_recent_return_std']:.2f}</span></div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Transitions</div>
                        <div class="metric-value">{m['total_transitions']:,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Episodes</div>
                        <div class="metric-value">{m['total_episodes']:,} across {m['num_seeds']} seeds</div>
                    </div>
                </div>
            </div>
            
            <div class="figures-row">
                <div class="figure-container">
                    <h3>Policy Performance & Multi-Seed Convergence Curves</h3>
                    <img src="{fig1_rel}" alt="{name} Benchmark Curves" onclick="openModal(this.src)" />
                </div>
                <div class="figure-container">
                    <h3>Q-Learning & Optimization Diagnostics</h3>
                    <img src="{fig2_rel}" alt="{name} Diagnostics" onclick="openModal(this.src)" onerror="this.parentElement.style.display='none'" />
                </div>
            </div>
        </div>
        """
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TurtleBot3 Phase-1 Multi-Algorithm Benchmark Suite</title>
    <style>
        :root {{
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-card: #334155;
            --primary: #38bdf8;
            --text: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #4ade80;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            max-width: 1400px;
            margin: 0 auto 24px auto;
            border-bottom: 1px solid var(--surface-card);
            padding-bottom: 16px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            color: var(--primary);
        }}
        .header p {{
            margin: 0;
            color: var(--text-muted);
            font-size: 15px;
        }}
        .tabs-bar {{
            max-width: 1400px;
            margin: 0 auto 24px auto;
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            border-bottom: 2px solid var(--surface);
            padding-bottom: 8px;
        }}
        .tab-btn {{
            background: var(--surface);
            color: var(--text-muted);
            border: 1px solid var(--surface-card);
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s ease;
        }}
        .tab-btn:hover {{
            background: var(--surface-card);
            color: var(--text);
        }}
        .tab-btn.active {{
            background: var(--primary);
            color: #0f172a;
            border-color: var(--primary);
        }}
        .tab-content {{
            display: none;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .tab-content.active {{
            display: block;
            animation: fadeIn 0.25s ease;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(6px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .header-card {{
            background: var(--surface);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
            border: 1px solid var(--surface-card);
        }}
        .header-card h2 {{
            margin: 0 0 16px 0;
            font-size: 22px;
            color: var(--text);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }}
        .metric-card {{
            background: var(--surface-card);
            padding: 14px 18px;
            border-radius: 8px;
        }}
        .metric-label {{
            font-size: 12px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        .metric-value {{
            font-size: 22px;
            font-weight: 700;
            color: var(--accent-green);
        }}
        .metric-std {{
            font-size: 14px;
            color: var(--text-muted);
            font-weight: normal;
        }}
        .figures-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(620px, 1fr));
            gap: 24px;
            margin-bottom: 32px;
        }}
        .figure-container {{
            background: var(--surface);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid var(--surface-card);
        }}
        .figure-container h3 {{
            margin: 0 0 12px 0;
            font-size: 16px;
            color: var(--primary);
        }}
        .figure-container img {{
            width: 100%;
            border-radius: 8px;
            cursor: zoom-in;
            background: white;
            transition: transform 0.2s ease;
        }}
        .figure-container img:hover {{
            transform: scale(1.01);
        }}
        /* Fullscreen image modal */
        #modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(4px);
            align-items: center;
            justify-content: center;
        }}
        #modal img {{
            max-width: 95vw;
            max-height: 95vh;
            border-radius: 8px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            cursor: zoom-out;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 TurtleBot3 Phase-1 Multi-Algorithm Benchmark Suite</h1>
        <p>Virginia Tech Advanced Research Computing (ARC) | 60-Task Controlled Production Evaluation</p>
    </div>
    
    <div class="tabs-bar">
        <button class="tab-btn active" onclick="switchTab('comparative')">🏆 Cross-Algorithm Comparison</button>
        {algo_tabs_html}
    </div>
    
    <!-- Comparative Tab -->
    <div id="tab-comparative" class="tab-content active">
        <div class="header-card">
            <h2>Cross-Algorithm Performance & Sample Efficiency Overlays</h2>
            <p style="color: var(--text-muted); margin: 0;">Comparative learning curves and Pareto trade-off analyses aggregating 5-seed empirical distributions across all Phase-1 DRL models.</p>
        </div>
        <div class="figures-row">
            <div class="figure-container">
                <h3>Comparative Learning Curves (Return, Success Rate, Length, Outcomes)</h3>
                <img src="comparative/comparative_learning_curves.png" alt="Comparative Learning Curves" onclick="openModal(this.src)" />
            </div>
            <div class="figure-container">
                <h3>Training Threshold Crossings, Returns, and Contact Rates</h3>
                <img src="comparative/comparative_sample_efficiency.png" alt="Sample Efficiency & Pareto Frontier" onclick="openModal(this.src)" />
            </div>
        </div>
    </div>
    
    <!-- Individual Algorithm Tabs -->
    {algo_sections_html}
    
    <div id="modal" onclick="closeModal()">
        <img id="modal-img" src="" alt="Fullscreen view" />
    </div>
    
    <script>
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            const targetContent = document.getElementById('tab-' + tabId);
            if (targetContent) targetContent.classList.add('active');
            
            event.target.classList.add('active');
        }}
        function openModal(src) {{
            document.getElementById('modal-img').src = src;
            document.getElementById('modal').style.display = 'flex';
        }}
        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
        }}
        document.addEventListener('keydown', (e) => {{
            if (e.key === 'Escape') closeModal();
        }});
    </script>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as hf:
        hf.write(html_content)
        
    print(f"Interactive HTML dashboard generated -> {html_path}")
    return html_path


# ==============================================================================
# 6. DEMO / SYNTHETIC DATA GENERATOR
# ==============================================================================

def generate_demo_dataset(cache_dir: str) -> None:
    """
    Generates high-fidelity synthetic benchmark datasets for all 5 Phase-1 algorithms
    to enable complete verification of multi-algorithm comparative plotting suites.
    """
    print("\n[DEMO MODE] Generating synthetic multi-algorithm verification runs...")
    demo_root = os.path.join(cache_dir, "demo_dataset")
    os.makedirs(demo_root, exist_ok=True)
    
    algo_params = {
        'dqn':              {'base_ret': 96.0,  'ret_std': 5.0,  'sr_final': 82.0, 'sr_std': 4.5, 'eff_step': 140000, 'col_rate': 36.0},
        'doubledqn':        {'base_ret': 102.0, 'ret_std': 4.0,  'sr_final': 87.5, 'sr_std': 3.5, 'eff_step': 110000, 'col_rate': 29.0},
        'duelingdoubledqn':  {'base_ret': 108.0, 'ret_std': 3.5,  'sr_final': 91.0, 'sr_std': 2.8, 'eff_step': 85000,  'col_rate': 23.0},
        'rainbowdqn':       {'base_ret': 112.0, 'ret_std': 3.0,  'sr_final': 93.5, 'sr_std': 2.5, 'eff_step': 65000,  'col_rate': 19.0},
        'discretesac':      {'base_ret': 115.0, 'ret_std': 3.2,  'sr_final': 94.0, 'sr_std': 2.2, 'eff_step': 60000,  'col_rate': 18.0},
        'sdsac':            {'base_ret': 116.5, 'ret_std': 3.0,  'sr_final': 95.0, 'sr_std': 2.0, 'eff_step': 55000,  'col_rate': 16.0},
    }
    
    seeds = [101, 202, 303, 404, 505]
    
    for algo_key, p in algo_params.items():
        algo_dir = os.path.join(demo_root, algo_key)
        for s in seeds:
            s_dir = os.path.join(algo_dir, f"seed_{s}", "job_demo")
            os.makedirs(os.path.join(s_dir, "logs"), exist_ok=True)
            
            # Generate 1000 episodes (~250k steps)
            np.random.seed(s + hash(algo_key) % 10000)
            n_eps = 1000
            
            episodes = []
            cum_step = 0
            
            for ep in range(1, n_eps + 1):
                progress = min(1.0, ep / 700.0)
                # Success probability grows with training
                p_goal = min(p['sr_final'] / 100.0, 0.15 + (p['sr_final']/100.0 - 0.15) * (1.0 / (1.0 + np.exp(-10 * (progress - 0.3)))))
                p_timeout = 0.02
                p_safety = 1.0 - p_goal - p_timeout
                
                outcome = np.random.choice(['goal', 'safety', 'timeout'], p=[p_goal, p_safety, p_timeout])
                
                if outcome == 'goal':
                    ep_len = int(np.random.normal(160, 25))
                    ep_ret = np.random.normal(p['base_ret'], p['ret_std'])
                elif outcome == 'safety':
                    ep_len = int(np.random.normal(90, 40))
                    ep_ret = np.random.normal(-40.0, 15.0)
                else:
                    ep_len = 500
                    ep_ret = np.random.normal(-20.0, 10.0)
                    
                ep_len = max(20, min(500, ep_len))
                cum_step += ep_len
                
                episodes.append({
                    'episode': ep,
                    'total_steps': cum_step,
                    'end_env_step': cum_step,
                    'length': ep_len,
                    'return': ep_ret,
                    'reward': ep_ret,
                    'outcome': outcome,
                    'end_reason': outcome
                })
                
            df_ep = pd.DataFrame(episodes)
            df_ep.to_csv(os.path.join(s_dir, "episodes.csv"), index=False)
            
            # Identity manifest
            manifest = {
                'algorithm': ALGORITHM_CONFIGS[algo_key]['name'],
                'learning_seed': s,
                'experiment': 'demo-verification'
            }
            with open(os.path.join(s_dir, "run_identity.json"), 'w') as mf:
                json.dump(manifest, mf)
                
    print(f"Synthetic multi-algorithm dataset populated in -> {demo_root}")


# ==============================================================================
# 7. MAIN CLI CONTROLLER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TurtleBot3 Multi-Algorithm Benchmark Suite & Publication Figure Generator"
    )
    default_inputs = []
    default_output = str(Path(__file__).resolve().parent / 'generated_training')

    parser.add_argument(
        "--cluster",
        choices=["tinkercliffs", "owl", "both"],
        default=None,
        help="Target Virginia Tech ARC cluster to auto-configure paths (tinkercliffs, owl, or both)."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        default=default_inputs,
        help="Directories or zip archives to search for algorithm results."
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help="Destination directory for organized individual and comparative figures."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI resolution (default: 300 for publication grade)."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic multi-algorithm verification data to preview full comparative suite."
    )
    args = parser.parse_args()
    global DEMO_MODE
    DEMO_MODE = args.demo
    if args.demo and args.cluster:
        parser.error('--demo cannot be combined with a cluster input')
    if not args.demo and not args.input_dirs and not args.cluster:
        parser.error('Provide --input-dirs for recorded training data; use build_report.py for evaluation results')
    
    # Handle cluster shortcut
    if args.cluster == "tinkercliffs":
        args.input_dirs = ["/projects/rl/Turtlebot-RL/Ray/tb3_v1.1.0/results/tinkercliffs/controlled"]
        args.output_dir = "/projects/rl/Turtlebot-RL/Ray/tb3_v1.1.0/results/benchmark_figures_tinkercliffs"
    elif args.cluster == "owl":
        args.input_dirs = ["/projects/rl/Turtlebot-RL/Ray/tb3_v1.1.0/results/owl/controlled"]
        args.output_dir = "/projects/rl/Turtlebot-RL/Ray/tb3_v1.1.0/results/benchmark_figures_owl"
    elif args.cluster == "both":
        args.input_dirs = [
            "/projects/rl/Turtlebot-RL/Ray/tb3_v1.1.0/results/tinkercliffs/controlled",
            "/projects/rl/Turtlebot-RL/Ray/tb3_v1.1.0/results/owl/controlled"
        ]
        args.output_dir = "/projects/rl/Turtlebot-RL/Ray/tb3_v1.1.0/results/benchmark_figures_dual_cluster"
    
    print("=" * 80)
    print("=== TURTLEBOT3 MULTI-ALGORITHM BENCHMARK SUITE GENERATOR ===")
    print("=" * 80)
    
    if args.demo:
        args.output_dir = os.path.join(args.output_dir, 'synthetic_demo')
    protected = Path(__file__).resolve().parent
    output = Path(args.output_dir).resolve()
    if output in (protected, protected / 'figures', protected / 'data', protected.parent / 'results'):
        parser.error('Choose a generated output directory; tracked result files are protected')
    cache_dir = os.path.join(args.output_dir, ".unpack_cache")
    os.makedirs(args.output_dir, exist_ok=True)
    
    search_paths = list(args.input_dirs)
    
    if args.demo:
        generate_demo_dataset(cache_dir)
        search_paths = [os.path.join(cache_dir, "demo_dataset")]
        
    # Discover and extract archives
    extracted = unpack_archives_if_needed(search_paths, cache_dir)
    search_paths.extend(extracted)
    
    # Locate all algorithm runs
    print("\nScanning search paths for algorithm metrics...")
    found_runs = find_algorithm_runs(search_paths)
    
    if not found_runs:
        print("No algorithm runs found in the specified paths.")
        if not args.demo:
            print("Tip: Use --demo to populate synthetic benchmarking curves for verification.")
        sys.exit(1)
        
    print(f"\nDiscovered {len(found_runs)} algorithms:")
    for ak, s_dict in found_runs.items():
        print(f"  • {ALGORITHM_CONFIGS.get(ak, {}).get('name', ak)}: {len(s_dict)} seeds ({list(s_dict.keys())})")
        
    # Generate individual figures
    algo_results = {}
    for algo_key, seed_data in found_runs.items():
        print(f"\n[Processing Algorithm: {algo_key.upper()}]")
        res = plot_individual_algorithm_figures(algo_key, seed_data, args.output_dir, dpi=args.dpi)
        if res:
            algo_results[algo_key] = res
            
    # Generate comparative suite
    plot_comparative_suite(algo_results, args.output_dir, dpi=args.dpi)
    
    # Generate interactive HTML dashboard
    generate_interactive_html_dashboard(algo_results, args.output_dir)
    
    if args.demo:
        html_path = Path(args.output_dir) / 'index.html'
        html = html_path.read_text(encoding='utf-8')
        html = html.replace('<body>', '<body><p style="background:#fff0d0;color:#800000;padding:20px;font-weight:bold">SYNTHETIC DEMO - NOT EXPERIMENTAL RESULTS</p>', 1)
        html_path.write_text(html, encoding='utf-8')
    (Path(args.output_dir) / 'provenance.json').write_text(json.dumps({
        'data_source': 'synthetic_demo' if args.demo else 'recorded_training_episodes',
        'input_paths': search_paths,
        'interpretation': 'Training episode statistics, not the held-out evaluation',
        'outcomes': {'safety': 'proximity stop', 'collision': 'recorded physical contact'},
    }, indent=2), encoding='utf-8')

    # Master README
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as rf:
        rf.write("# 📊 TurtleBot3 Phase-1 Multi-Algorithm Benchmark Suite\n\n")
        rf.write(("SYNTHETIC DEMO - NOT EXPERIMENTAL RESULTS.\n\n" if args.demo else "Recorded training-episode statistics; not held-out policy evaluation.\n\n"))
        rf.write("## Directory Layout\n")
        rf.write("- `individual/`: Dedicated folders per algorithm containing multi-seed benchmark curves, Q-learning diagnostics, and per-seed JSON summaries.\n")
        rf.write("- `comparative/`: Cross-algorithm overlays for Return, Success Rate %, Sample Efficiency (steps to 80%), and Safety Pareto Frontiers.\n")
        rf.write("- `index.html`: Interactive responsive HTML report for viewing all plots in your browser.\n")
        
    print("\n" + "=" * 80)
    print("=== ALL BENCHMARK FIGURES & REPORTS SUCCESSFULLY GENERATED ===")
    print(f"Output Directory: {os.path.abspath(args.output_dir)}")
    print(f"Interactive Gallery: file:///{os.path.abspath(os.path.join(args.output_dir, 'index.html')).replace(os.sep, '/')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
