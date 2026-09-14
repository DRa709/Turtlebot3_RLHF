# Rainbow DQN — algorithm description

Package: `TurtleBot_RainbowDQN_Random` 1.0.1. Roadmap section: §3.4.
Primary references: Hessel et al., *AAAI* 2018; van Hasselt et al., *AAAI*
2016; Wang et al., *ICML* 2016; Schaul et al., *ICLR* 2016; Bellemare et al.,
*ICML* 2017; Fortunato et al., *ICLR* 2018. Time-limit semantics follow Pardo
et al., *ICML* 2018.

## What it is

Rainbow DQN is model-free, off-policy and value-based. It combines the Double
and Dueling improvements with C51, prioritized replay, three-step returns and
NoisyNet exploration. There is no actor, policy-gradient loss, epsilon schedule
or on-policy rollout buffer.

## Network

All four linear stages are factorized Gaussian noisy layers:

```text
features:       41 -> 256 -> 256, ReLU after each layer
value logits:   256 -> 51
advantage:      256 -> 5 x 51
support:        51 equally spaced atoms from -200 to 200
```

For atom `j`, the action logits are

\[
\ell_j(o,a)=V_j(o)+A_j(o,a)-\frac{1}{5}\sum_b A_j(o,b).
\]

Softmax over the 51 atoms gives \(p_j(o,a;\theta)\), and
\(Q_\theta(o,a)=\sum_j z_jp_j(o,a;\theta)\). Mean-centering is over actions,
separately for every atom. The network has 310,372 trainable parameters,
including learned noisy means and scales.

Each noisy layer uses

\[
w=\mu_w+\sigma_w\odot\xi,
\]

with factorized Gaussian noise, \(\mu\sim U[-1/\sqrt p,1/\sqrt p]\), and
weight and bias scales initialized to \(\sigma_0/\sqrt p\), where `p` is the
fan-in and \(\sigma_0=0.5\). Fresh noise is drawn once per non-warm-up action
and once per update for each online/target network. Evaluation uses only the
mean parameters.

## Three-step distributional target

For a stored transition beginning at time `t`, let `k <= 3` stop at the first
episode boundary:

\[
G_t^{(k)}=\sum_{j=0}^{k-1}\gamma^j r_{t+j}.
\]

The accumulator stops for either termination or truncation, so a return never
crosses a reset. Only a genuine termination sets
\(m^{\mathrm{term}}=1\). A time-limit truncation retains the final pre-reset
successor observation, uses \(m^{\mathrm{term}}=0\), and therefore bootstraps.

The online network selects

\[
a^*=\arg\max_a\sum_jz_jp_j(o_{t+k},a;\theta),
\]

and the target network supplies \(p_j(o_{t+k},a^*;\bar\theta)\). Each target
atom is shifted to

\[
\hat z_j=\operatorname{clip}\!\left(
G_t^{(k)}+\gamma^k(1-m^{\mathrm{term}})z_j,
V_{\min},V_{\max}\right).
\]

The shifted distribution is projected onto the fixed support. Projection
indices are clamped to `[0, 50]`, exact-atom mass is handled explicitly, and
mass conservation is checked at runtime. If \(m_j\) is the projected target,
the unweighted per-sample categorical loss is

\[
\ell_i=-\sum_jm_{ij}\log p_j(o_i,a_i;\theta).
\]

The full target path, including online selection, target evaluation and C51
projection, is computed without gradients.

## Prioritized replay and loss

The replay buffer holds 100,000 n-step transitions. New transitions enter at
the current maximum raw priority. Sampling with replacement uses

\[
q_i=\frac{p_i^{0.6}}{\sum_l p_l^{0.6}},\qquad
w_i=\frac{(|D|q_i)^{-\beta}}{\max_l(|D|q_l)^{-\beta}}.
\]

The importance exponent increases linearly from 0.4 to 1.0 over 500,000
environment transitions. The optimized loss is

\[
L=\frac{1}{64}\sum_iw_i\ell_i,
\]

and the updated raw priority is \(p_i=\ell_i+10^{-6}\), using the unweighted
loss. Sampling and updates use logarithmic-time sum/min/max trees. If one item
is sampled more than once in a minibatch, its largest new priority is retained.

## Frozen training loop

1. Initialize online parameters and copy them to the target network.
2. Collect 5,000 transitions with uniform-random actions and no gradient step.
3. After warm-up, choose the expected-Q argmax under freshly sampled NoisyNet
   parameters; epsilon is inapplicable.
4. Push each raw transition through the episode-safe three-step accumulator.
5. Starting at environment transition 5,000, draw a prioritized batch of 64
   and take one Adam step after every transition.
6. Use learning rate `1e-4`, discount `0.99` and gradient-norm clip `10`.
7. Copy online parameters to the target every 1,000 gradient steps.
8. Stop at exactly 500,000 environment transitions.

Controlled runs are CPU-pinned to one PyTorch thread and use learning seeds
101, 202, 303, 404 and 505.

## Evaluation

Evaluation is noise-free and greedy. A policy-only checkpoint is loaded into a
separate evaluation object, learning is disabled, and replay is untouched. E1
uses new reproducible random starts; E2 uses the frozen 100-scenario set; E3
uses 20 fixed-anchor episodes. Evaluation transitions never consume the
training budget.

## Random initial states

Randomization belongs to the environment, not the learner. Episode 1 and every
later training episode receive a seeded robot pose drawn from the declared
clearance-admissible law. Reset, teleport, settle, fresh sensors, realized-pose
checks and obstacle-schedule acknowledgement complete before the first action.
The goal is fixed and the world contains static geometry plus two moving
obstacles. Full details are in `RANDOM_INIT_SPEC.md`.

## Rainbow-specific files

The learner and ROS adapter are `turtlebot3_drl_nav/rainbowdqn.py` and
`turtlebot3_drl_nav/rainbowdqn_agent_node.py`. Configuration and ARC entry
points are `config/phase1_rainbowdqn.yaml`,
`arc/rainbowdqn_array.sbatch`, and
`arc/rainbowdqn_eval_array.sbatch`. No other learner is present or imported.

## Diagnostics recorded per update

Every update records categorical loss, mean absolute scalar TD diagnostic,
selected-action Q, target value, next target value, gradient norm, target-sync
flag, replay size and terminal fraction; PER beta, weights and priorities;
NoisyNet sigma minimum/mean/maximum; configured and effective n-step length;
value-stream expectation and centered-advantage magnitude; selected
distribution entropy; and C51 low/high clipping fractions and maximum mass
error. `epsilon` is always empty, never zero.

**In simple words.** The robot predicts a whole range of possible scores for
each movement. Two networks divide choosing from judging, the value and
advantage streams split “how good is this place?” from “which move is better?”,
important memories are reread more often, three-step stories move information
faster, and learned noise supplies exploration without an epsilon coin flip.
