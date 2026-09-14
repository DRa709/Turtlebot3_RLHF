# DQN — algorithm description (TurtleBot_DQN_Random 1.1.7)

Roadmap section: §3.1. Reference: Mnih et al., *Nature* 2015; time-limit semantics per Pardo et al., ICML 2018.

## What it is

Deep Q-Network is model-free, off-policy and value-based. It learns one action-value function Q_θ(o, a) for the five discrete commands and acts by taking the argmax. Three mechanisms make Q-learning with a neural network stable: a replay buffer that breaks the correlation between consecutive transitions, a slowly updated target network that keeps the regression target fixed between synchronizations, and ε-greedy exploration preceded by a random warm-up.

## Update rule

For a sampled transition i = (o_i, a_i, r_i, o′_i, m_i^term):

y_i = r_i + γ (1 − m_i^term) max_{a′} Q_θ̄(o′_i, a′),  L(θ) = (1/|B|) Σ_i Huber(y_i − Q_θ(o_i, a_i)),

with no gradient through the target parameters θ̄ and the Huber loss quadratic below 1 and linear above. Only genuine terminations (goal, collision, safety stop) set m^term = 1. A time-limit truncation stores m^term = 0 together with the final pre-reset observation, so the target still bootstraps through it.

## The loop, with the frozen values of this study

1. Initialize θ; set θ̄ ← θ; empty replay buffer of capacity 100 000.
2. Warm-up: collect 5 000 transitions with uniform-random actions; no update is attempted before the replay reaches this threshold.
3. Act ε-greedy: ε decreases linearly from 1.0 to 0.05 over the first 100 000 transitions, counted from transition 0. The action that supplies replay item 5 000 uses ε = 0.9525095 (≈ 0.9525), and the first update occurs immediately after that item is stored. ε then remains at 0.05 after transition 100 000.
4. Store (o, a, r, o′, m^term).
5. Once the replay contains 5 000 items, sample 64 transitions uniformly after every stored transition and take one Adam step (learning rate 1e-4) on L, with gradient norm clipped at 10.
6. Every 1 000 gradient steps set θ̄ ← θ. Stop at exactly B_env = 500 000 transitions; γ = 0.99 throughout.

Network: MLP 41 → 256 → 256 → 5 with ReLU. The same trunk is used by Double DQN; Dueling and Rainbow change the head.

## Evaluation

Greedy, ε = 0, one policy channel (`policy_mode = greedy`). Every checkpoint is evaluated from its file by a separate `GreedyPolicy`; the learner's parameter digest and replay size are asserted unchanged across the evaluation block and the flag is recorded on every evaluation row.

## Role in the study

DQN is the base of the incremental value-based sequence DQN → Double DQN → Dueling Double DQN → Rainbow, each later method adding one mechanism. Its known limits are the ones those methods address: the max in the target overestimates values (Double), uniform replay treats every transition as equally useful (prioritized replay), and one-step targets propagate reward slowly (n-step returns).

The random initial-state distribution is an environment property, not an extra DQN update mechanism. DQN receives the realized starts through its observations and replay buffer.

## What is DQN-specific in the package

`turtlebot3_drl_nav/dqn.py` (network, replay, update, atomic checkpoints, greedy policy), `turtlebot3_drl_nav/dqn_agent_node.py`, `turtlebot3_drl_nav/learner_factory.py`, `config/phase1_dqn.yaml`, `arc/submit_dqn.sh`, `arc/dqn_array.sbatch`, `arc/dqn_eval_array.sbatch`, `arc/run_dqn_seed.sh`, `arc/run_dqn_eval.sh`, `tests/test_dqn.py`, `tests/test_orchestrator.py`, and the `ALGORITHM` marker. Everything else is the shared layer, verified by `SHARED_LAYER_MANIFEST.sha256`.

## Diagnostics recorded per update

Loss, mean |TD error|, Q of the taken actions, target mean, next-state max-Q, gradient norm, learning rate, the ε used for the action at that step, a target-synchronization flag, the batch's terminal fraction, batch size and replay size. ε is required on every DQN update row; Rainbow-only and Discrete-SAC-only fields stay empty (`recorder.UPDATE_APPLICABILITY`).

**In very simple words.** The robot keeps a scorecard for every button. It presses the button with the best score, but sometimes tries a random one just to check. It writes every trip in a diary and re-reads old pages to fix its scores.
