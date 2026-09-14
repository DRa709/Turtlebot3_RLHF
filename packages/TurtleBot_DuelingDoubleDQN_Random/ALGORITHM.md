# Dueling Double DQN — algorithm description

Package: `TurtleBot_DuelingDoubleDQN_Random` 1.0.2. Roadmap section: §3.3.
Primary references: Wang et al., *Dueling Network Architectures for Deep
Reinforcement Learning*, ICML 2016; van Hasselt, Guez and Silver, *Deep
Reinforcement Learning with Double Q-learning*, AAAI 2016. Time-limit semantics
follow Pardo et al., ICML 2018.

## What it is

Dueling Double DQN is model-free, off-policy and value-based. It combines two
precisely defined mechanisms in one learner:

1. A dueling network represents state value and relative action advantages
   separately.
2. A Double-Q target uses the online network to select the next action and the
   target network to evaluate that action.

No prioritized replay, multi-step return, distributional target, noisy layer,
actor, policy-gradient objective or entropy regularization is implemented.

## Dueling architecture

For one observation `o`, the network computes

```text
features:   41 -> 256 -> 256, ReLU after each hidden layer
value head: 256 -> 1
advantage:  256 -> 5
```

The five Q-values are

\[
Q_\theta(o,a)=V_\theta(o)+A_\theta(o,a)
-\frac{1}{5}\sum_{b=1}^{5}A_\theta(o,b).
\]

Mean-centering removes the additive ambiguity between the value and advantage
streams. The network contains 78,086 trainable parameters.

## Update rule

For transition
\(i=(o_i,a_i,r_i,o'_i,m_i^{\mathrm{term}})\),

\[
a_i^*=\arg\max_{a'}Q_\theta(o'_i,a'),
\qquad
y_i=r_i+\gamma(1-m_i^{\mathrm{term}})
Q_{\bar\theta}(o'_i,a_i^*),
\]

and

\[
L(\theta)=\frac{1}{|B|}\sum_{i\in B}
\operatorname{Huber}\!\left(y_i-Q_\theta(o_i,a_i)\right).
\]

The complete target is computed without gradients. Genuine goal, collision and
safety-stop terminations set \(m^{\mathrm{term}}=1\). A time-limit truncation
stores the final pre-reset successor observation and
\(m^{\mathrm{term}}=0\), so it bootstraps.

## Declared training protocol

1. Initialize online parameters and copy them to the target network.
2. Use a uniform replay buffer of capacity 100,000.
3. Collect 5,000 uniform-random warm-up transitions with no gradient update.
4. Use epsilon-greedy exploration. Epsilon decreases linearly from 1.0 to 0.05
   over the first 100,000 environment transitions, including warm-up.
5. Starting at environment transition 5,000, sample 64 transitions uniformly
   and take one Adam step after every transition.
6. Use learning rate `1e-4`, discount `0.99`, Huber loss and gradient-norm clip
   `10`.
7. Copy online parameters to the target network every 1,000 gradient steps.
8. Stop at exactly 500,000 training transitions and write policy checkpoints
   every 25,000 transitions.

Controlled runs use learning seeds 101, 202, 303, 404 and 505 and are CPU-pinned
with one PyTorch thread.

## Random initial states

The learner does not generate start states. Episode 1 and every subsequent
training episode receive a reproducible robot position sampled uniformly over
the declared clearance-admissible region and an independent uniform yaw on
`[-pi, pi)`. Reset, relocation, settling, sensor freshness and realized-pose
validation complete before the first action. The goal and map remain fixed;
the world includes static geometry and two simulation-time moving obstacles.

## Evaluation

Evaluation is greedy (`epsilon=0`) using a frozen policy-only checkpoint.
Learning and replay mutation are disabled. E1 evaluates reproducible new draws;
E2 evaluates the frozen 100-scenario set; E3 evaluates 20 anchor episodes.

## Package isolation

This archive contains exactly one learner implementation:
`turtlebot3_drl_nav/duelingdoubledqn.py`. Its ROS adapter is
`turtlebot3_drl_nav/duelingdoubledqn_agent_node.py`. The configuration and ARC
entry points are `config/phase1_duelingdoubledqn.yaml`,
`arc/duelingdoubledqn_array.sbatch` and
`arc/duelingdoubledqn_eval_array.sbatch`. It does not import or require another
algorithm package. `tests/test_dueling_recorder_contract.py` additionally binds
the learner's value-stream and centered-advantage diagnostics to the canonical
update schema.

## Per-update diagnostics

Every update records loss, mean absolute TD error, selected-action Q-value,
target mean, target-network value at the online-selected next action, gradient
norm, learning rate, epsilon used by the corresponding action, target-sync
flag, terminal fraction, batch size, replay size, mean value-stream output and
mean absolute centered advantage.

**In simple words.** The robot estimates both how good the situation is and
which movement is better than the alternatives. It uses one network to choose
the next movement and a second, delayed network to judge that choice.
