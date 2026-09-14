# Discrete SAC — algorithm description

Package: `TurtleBot_DiscreteSAC_Random` 1.0.2. References: Christodoulou,
*Soft Actor-Critic for Discrete Action Settings* (2019); temperature-stability
context from Zhou et al., *Revisiting Discrete Soft Actor-Critic* (2022).

## What it is

Discrete Soft Actor-Critic is model-free, off-policy, replay-based and
actor-critic. A categorical actor assigns a probability to each of the five
robot commands. Two critics independently score all five commands. Taking the
smaller critic value reduces optimistic value estimates; the entropy term keeps
the learned policy from becoming prematurely certain.

This package implements vanilla categorical Discrete SAC. It does not contain
Rainbow, automatic temperature tuning, an entropy-penalty critic, Q-clip, PER,
n-step returns or distributional critics.

## Update rule

For the next observation, all five actions are summed exactly:

$$
\bar V(o')=\sum_a\pi_\phi(a\mid o')
\left[\min_{k\in\{1,2\}}Q_{\bar\theta_k}(o',a)
-\alpha\log\pi_\phi(a\mid o')\right].
$$

The one-step target is

$$
y=r+\gamma(1-m^{\mathrm{term}})\bar V(o').
$$

Each critic minimizes mean squared error

$$
L_{Q_k}=|B|^{-1}\sum_i(y_i-Q_{\theta_k}(o_i,a_i))^2.
$$

The actor minimizes the exact categorical objective

$$
L_\pi=|B|^{-1}\sum_i\sum_a\pi_\phi(a\mid o_i)
\left[\alpha\log\pi_\phi(a\mid o_i)
-\min_kQ_{\theta_k}(o_i,a)\right].
$$

Targets and actor-side critic values are detached. Only genuine terminations
(goal, collision, safety stop) set $m^{\mathrm{term}}=1$. Time-limit and exact-
budget truncations store the final pre-reset $o'$ with $m^{\mathrm{term}}=0$, so
they bootstrap.

## Frozen study values

- Actor: MLP 41 → 256 → 256 → 5 logits, ReLU.
- Each of two critics: MLP 41 → 256 → 256 → 5 Q values, ReLU.
- Each network has 77,829 parameters; 233,487 parameters are trained.
- Uniform replay capacity 100,000; batch size 64.
- Uniform-random warm-up: 5,000 transitions, no update before transition 5,000.
- One actor step and one step for each critic per environment transition after warm-up.
- Adam actor and critic learning rates: $10^{-4}$.
- $\gamma=0.99$; global gradient-norm clip 10 for each network.
- Fixed $\alpha=0.2$; no alpha optimizer and no target entropy.
- Both target critics hard-copy every 1,000 gradient steps.
- Exact controlled budget: 500,000 environment transitions.
- Policy checkpoints every 25,000 transitions; full checkpoints every 100,000 and at the final budget.

Hard target synchronization is the study's controlled target-update rule; it is
intentional even though SAC is often presented with Polyak averaging.

## Acting and evaluation

During warm-up, actions are uniform random. Afterwards, training samples from
the actor's categorical distribution. Evaluation is frozen and has two
separate channels on matched initial states:

1. `stochastic`: seeded categorical sampling;
2. `deterministic`: argmax actor probability.

The stochastic action seed is derived from evaluation seed, checkpoint,
condition, scenario/episode, policy mode and decision sequence. Evaluation
does not modify the learner, optimizers or replay.

## Random initial-state role

The learner does not generate robot starts. Every episode, including episode
1, is transactionally reset to the declared random-start distribution by the
environment before the first observation. Requested and realized pose,
odometry, clearance, obstacle phase and role-specific seeds are recorded and
revalidated. See `RANDOM_INIT_SPEC.md`.

## Discrete-SAC-specific files

`turtlebot3_drl_nav/discretesac.py`,
`turtlebot3_drl_nav/discretesac_agent_node.py`,
`config/phase1_discretesac.yaml`, `arc/discretesac_array.sbatch`,
`arc/discretesac_eval_array.sbatch`, and `tests/test_discretesac*.py`.
No other algorithm implementation is present.

## Diagnostics per optimizer step

Actor loss; both critic losses and their mean; maximum absolute twin TD error;
taken-action Q values and critic gap; soft target and next soft value; minimum
policy Q; current/next entropy; maximum/minimum action probability; fixed alpha;
three gradient norms; both learning rates; terminal fraction; batch/replay size;
and target-sync flag.

**In very simple words.** The robot keeps two cautious scorecards and a set of
probabilities for its five buttons. It learns from stored trips, prefers useful
buttons, and receives a small bonus for not becoming too certain too early.
