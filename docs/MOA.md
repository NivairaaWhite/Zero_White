# Mixture-of-Agents routing (6.5)

`forge/moa.py`. The chat model gained Mixture-of-Experts feed-forward blocks in 6.4. This lifts the idea one level up, to whole agents, in three forms that share one router design. All are ordinary families of the Generalist: they enter the race and win only by validation skill.

| family | for | what it is |
|---|---|---|
| `moa` | classification, regression | K heterogeneous micro-agents trained **together** with a router: MLPs, linear heads, and perception experts (an autoencoder whose code feeds a small head). For regression it can also include a **frozen evolved program** (`program_formula`): logic as source code whose output the router can use but not change. |
| `moa_policy` | environments | the same router over K micro-policies inside the actor-critic, trained by the same policy gradient (`agents.PGAgent` gained three hook methods whose defaults are the old behaviour, so existing agents are unchanged). |
| `moa_blend` | the Autonomy layer | after a race, a router over the **finished winners of different families**. The gate is fitted on half the validation rows and the blend is scored on the other half, which it never saw; it is kept only if it beats the single winner there. |

**The router** is a linear softmax gate with noise during training and hard top-k at inference: only the chosen experts run on a row. A load-balancing term (the Switch-Transformer one: `K * sum_k f_k * P_k`) keeps it from sending everything to one expert. Usage and entropy are **measured and reported**, not assumed.

## Honest limits

* Routing helps when different regions of the input want different experts (the tests build such a task and compare with a single expert of the same total size). On a task one MLP already solves, the mixture costs more parameters for no gain and the race simply does not pick it.
* A linear gate partitions the input space by hyperplanes, no more.
* A frozen program expert can only be blended at the output (programs are not differentiable), and only for regression.
* The blend's gain is measured on half the validation set: a noisier estimate than the full-set skill the other candidates report. (In a run on a near-linear regression task it measured -0.04 against the winner and was correctly not used.)
