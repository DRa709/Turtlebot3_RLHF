# Double DQN — algorithm description (TurtleBot_DoubleDQN_Random 1.0.1)

Roadmap section: §3.2. Reference: van Hasselt, Guez and Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016; time-limit semantics per Pardo et al., ICML 2018.

## What it is

Double Deep Q-Network is model-free, off-policy and value-based. It retains DQN's replay buffer, online network, target network and ε-greedy exploration, but changes how the next-state target is constructed. The online network selects the next action; the target network evaluates that selected action. This decoupling reduces the overestimation produced when the same noisy value estimates both select and evaluate a maximum.

## Update rule

For transition $i=(o_i,a_i,r_i,o'_i,m_i^{\mathrm{term}})$,

$$
a_i^*=\arg\max_{a'}Q_\theta(o'_i,a'),
\qquad
y_i=r_i+\gamma(1-m_i^{\mathrm{term}})Q_{\bar\theta}(o'_i,a_i^*),
$$

$$
L(\theta)=\frac{1}{|\mathcal B|}\sum_{i\in\mathcal B}
\operatorname{Huber}\!\left(y_i-Q_\theta(o_i,a_i)\right).
$$

The argmax is computed by the online network $Q_\theta$; its value is gathered from the target network $Q_{\bar\theta}$. The complete target is computed without a gradient. Only genuine terminations—goal, collision or safety stop—set $m_i^{\mathrm{term}}=1$. Time-limit truncations store $m_i^{\mathrm{term}}=0$ and the final pre-reset observation, so they bootstrap.

## Frozen study values

1. MLP: 41 → 256 → 256 → 5 with ReLU.
2. Replay capacity: 100,000 transitions; uniform sampling.
3. Warm-up: 5,000 uniform-random transitions and no earlier update.
4. Exploration: ε decreases linearly from 1.0 to 0.05 over 100,000 environment transitions, counted from transition 0, then stays at 0.05.
5. Batch size 64; one Adam update per stored transition from step 5,000 onward; learning rate $10^{-4}$; Huber loss; gradient norm clipped at 10.
6. Discount $\gamma=0.99$; hard target synchronization every 1,000 gradient steps.
7. Controlled budget: exactly 500,000 environment transitions, with policy checkpoints every 25,000 transitions and full checkpoints every 100,000 transitions plus the final checkpoint.

All frozen values other than the Double DQN target are the common value-based study settings. This package has no dueling head, prioritized replay, multi-step return or noisy network.

## Evaluation

Evaluation is greedy with ε = 0. A checkpoint is loaded into a separate frozen policy object; learning and replay remain untouched. E1 uses new random starts, E2 uses the frozen 100-scenario set, and E3 uses the separately reported anchor start.

## Role in the study

Double DQN is the second method in the incremental sequence DQN → Double DQN → Dueling Double DQN → Rainbow DQN. It isolates one mechanism: separating next-action selection from target evaluation. Any difference from DQN can therefore be interpreted as evidence about overestimation control, subject to the common random-start and execution contracts.

The random initial-state distribution belongs to the environment. Every training episode, including episode 1, supplies Double DQN with a new reproducible draw from $\nu_R$; the learner itself does not alter that distribution.

## Double-DQN-specific files

`turtlebot3_drl_nav/doubledqn.py`, `turtlebot3_drl_nav/doubledqn_agent_node.py`, `turtlebot3_drl_nav/learner_factory.py`, `config/phase1_doubledqn.yaml`, `arc/doubledqn_array.sbatch`, `arc/doubledqn_eval_array.sbatch`, `arc/run_doubledqn_seed.sh`, `arc/run_doubledqn_eval.sh`, `tests/test_doubledqn.py`, `tests/test_doubledqn_source.py`, `tests/test_orchestrator.py`, and `ALGORITHM`. `SHARED_LAYER_FILES.txt` defines the algorithm-independent scientific core; package-branded files remain Double-DQN-specific even when they implement a common contract.

## Diagnostics recorded per update

Loss, mean absolute TD error, Q-value of the sampled action, target mean, target-network value at the online-selected next action, gradient norm, learning rate, ε used for the corresponding action, target-synchronization flag, terminal fraction, batch size and replay size. Rainbow-only and Discrete-SAC-only fields remain empty.

**In very simple words.** One scorecard chooses the next button, while a slower scorecard judges that choice. Separating choosing from judging prevents one unusually optimistic score from controlling both decisions.
