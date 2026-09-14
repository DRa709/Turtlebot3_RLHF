# SD-SAC — algorithm description

Package: `TurtleBot_SDSAC_Random` 1.0.2. Primary reference: Zhou et al.,
*Revisiting Discrete Soft Actor-Critic*, TMLR 2024 (arXiv:2209.10081v4).
Categorical SAC background: Christodoulou, 2019.

## What it is

Stable Discrete Soft Actor-Critic (SD-SAC) is model-free, off-policy,
replay-based and actor-critic. A categorical actor gives probabilities for the
five robot commands. Two critics score all five commands. SD-SAC addresses the
underestimation and instability identified for vanilla discrete SAC with three
linked mechanisms: double-average Q learning, Q-clip, and an entropy-change
penalty.

This is a controlled TurtleBot adaptation of SD-SAC. The three defining
mechanisms follow Zhou et al.; study choices such as the 41-256-256-5 network,
one-step replay target, fixed temperature and hard target cadence remain frozen
so the SD-SAC treatment is isolated. It is not a byte-for-byte reproduction of
the paper's Atari setup.

## Update rule

For every next observation, expectations are exact sums over all five actions.
The twin target critics are averaged:

$$
\bar Q_{\bar\theta}(o',a)=\tfrac12
\left(Q_{\bar\theta_1}(o',a)+Q_{\bar\theta_2}(o',a)\right),
$$

$$
\bar V(o')=\sum_a\pi_\phi(a\mid o')
\left[\bar Q_{\bar\theta}(o',a)-\alpha\log\pi_\phi(a\mid o')\right],
$$

$$
y=r+\gamma(1-m^{\mathrm{term}})\bar V(o').
$$

For critic $k$, let the clipped estimate around the current-state target critic
be

$$
Q^{\mathrm{clip}}_k=Q_{\bar\theta_k}(o,a)+
\operatorname{clip}\left(Q_{\theta_k}(o,a)-Q_{\bar\theta_k}(o,a),-c,c\right).
$$

The Q-clip loss takes the maximum for each sampled transition before the batch
mean:

$$
L_{Q_k}=|B|^{-1}\sum_i\max\left[
(Q_{\theta_k}(o_i,a_i)-y_i)^2,
(Q^{\mathrm{clip}}_{k,i}-y_i)^2\right].
$$

Each replay item stores $H_{\mathrm{old}}(o)$, the entropy of the behavior
policy when the transition was collected. For the explicit uniform-random
warm-up behavior, this is $\log 5$. Afterwards it is the exact categorical
entropy of the actor that selected the action. With

$$
H_\phi(o)=-\sum_a\pi_\phi(a\mid o)\log\pi_\phi(a\mid o),
$$

the actor minimizes

$$
L_\pi=|B|^{-1}\sum_i\sum_a\pi_\phi(a\mid o_i)
\left[\alpha\log\pi_\phi(a\mid o_i)-
\tfrac12(Q_{\theta_1}(o_i,a)+Q_{\theta_2}(o_i,a))\right]
+\tfrac12\beta|B|^{-1}\sum_i(H_{\mathrm{old},i}-H_\phi(o_i))^2.
$$

Targets and actor-side critic values are detached. Only genuine terminations
(goal, collision, safety stop) set $m^{\mathrm{term}}=1$. Time-limit and exact-
budget truncations store the final pre-reset observation with
$m^{\mathrm{term}}=0$, so they bootstrap.

## Frozen study values

- Actor: MLP 41 → 256 → 256 → 5 logits, ReLU.
- Each critic: MLP 41 → 256 → 256 → 5 Q values, ReLU.
- Uniform replay capacity 100,000; batch size 64.
- Uniform-random warm-up: 5,000 transitions; no earlier optimizer step.
- One actor step and one step per critic after every later transition.
- Adam actor and critic learning rates: $10^{-4}$.
- $\gamma=0.99$; gradient-norm clip 10 per network.
- Fixed $\alpha=0.2$; no temperature optimizer or target entropy.
- Entropy-penalty coefficient $\beta=0.5$.
- Q-clip range $c=0.5$.
- Both target critics hard-copy every 1,000 gradient steps.
- Controlled budget: exactly 500,000 environment transitions.
- Policy checkpoints every 25,000 transitions; full checkpoints every 100,000
  and at the final budget.

The paper uses a different Atari training recipe, including Polyak targets and
other task-specific values. Those choices are not imported here because this
package tests SD-SAC mechanisms in the frozen TurtleBot experimental setting.

## Acting and evaluation

Warm-up actions are uniform random. Afterwards, training samples from the
actor's categorical distribution. Evaluation is frozen and uses two separately
reported channels on matched starts: seeded stochastic categorical sampling
and deterministic argmax. Evaluation cannot mutate the learner, optimizers or
replay.

## Random initial-state role

The learner does not generate starts. The environment transactionally samples,
applies, settles, validates and records a seeded robot pose before the first
observation of every episode, including episode 1. See `RANDOM_INIT_SPEC.md`.
The fixed goal and simulation-time moving-obstacle law are part of the declared
world; obstacle phase is separately seeded and recorded.

## SD-SAC-specific files

`turtlebot3_drl_nav/sdsac.py`,
`turtlebot3_drl_nav/sdsac_agent_node.py`,
`config/phase1_sdsac.yaml`, `arc/sdsac_array.sbatch`,
`arc/sdsac_eval_array.sbatch`, and `tests/test_sdsac*.py`.
No other algorithm implementation is present or imported.

## Diagnostics per optimizer step

The package records actor total/base losses; entropy penalty and entropy gap;
old/current/next entropy; both Q-clip losses, clipped-Q means and activation
fractions; twin Q values and gap; averaged policy Q; soft targets; action
probability bounds; fixed $\alpha$, $\beta$ and $c$; three gradient norms;
learning rates; replay/batch sizes; terminal fraction; and target-sync flag.

**In very simple words.** The robot learns a probability for each of five
buttons and keeps two scorecards. It averages the scorecards, prevents either
from changing too abruptly, and penalizes sudden changes in how uncertain its
button choices are.
