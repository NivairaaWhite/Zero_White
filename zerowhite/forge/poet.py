"""poet.py - open-ended co-evolution of ENVIRONMENTS and the agents that solve them (after POET).

The Generalist solves tasks it is given. POET-lite *invents* tasks. It keeps an archive of
(environment-parameters, agent) pairs and repeats:

  1. MUTATE      perturb the physics of a parent environment (gravity, pole length, push force, ...);
  2. MINIMAL CRITERION   the parent's agent must find the child environment neither trivial nor impossible
                 (skill inside [lo, hi]); otherwise the child is discarded: there is nothing to learn from it;
  3. NOVELTY     the child's parameters must differ enough from every environment already in the archive;
  4. OPTIMISE    a new agent is started FROM the parent's evolved programs and trained on the child;
  5. TRANSFER    every agent is tried on every environment; if a foreign agent beats the resident one
                 there, it takes over (stepping stones: a controller evolved for a heavy pole can be the
                 best starting point for a long one).

Agents here are evolved *programs*, so each environment ends up with a short readable controller.
Every agent is also saved in the zoo, so later `solve` calls can transfer from them.
"""

import numpy as np

from ..evolve import random_genome
from . import task as T
from .envs import ParamCartPole
from .families import _PROGRAM_SPACE, ProgramPolicyAgent
from .zoo import Zoo


class Pair:
    def __init__(self, params, task, agent, parent=None):
        self.params, self.task, self.agent, self.parent = params, task, agent, parent
        self.skill = -1.0


def _norm(params):
    b = ParamCartPole.BOUNDS
    return np.array([(params[k] - b[k][0]) / (b[k][1] - b[k][0]) for k in sorted(b)])


def mutate_params(params, rng, scale=0.35):
    b, child = ParamCartPole.BOUNDS, dict(params)
    for k in rng.choice(sorted(b), size=int(rng.integers(1, 3)), replace=False):
        child[k] = float(np.clip(child[k] * np.exp(rng.normal(0.0, scale)), *b[k]))
    return child


def _make(params, genome, seed):
    env = ParamCartPole(**params)
    task = T.EnvTask(env, name=env.name)
    return task, ProgramPolicyAgent(task, genome, seed)


def poet(workspace_root, rounds=6, seed=0, max_envs=5, units=0.6, crit=(0.15, 0.97), min_novelty=0.12, tries=15,
         verbose=True):
    rng = np.random.default_rng(seed)
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    genome = random_genome(_PROGRAM_SPACE, rng)                       # no default agent: even the first one is random
    p0 = dict(ParamCartPole.DEFAULTS)
    task0, a0 = _make(p0, genome, seed)
    a0.train(task0, units * 2, rng)
    pairs = [Pair(p0, task0, a0)]
    pairs[0].skill = a0.val_score(task0)[0]
    log(f"[poet] seed environment {task0.name}: skill {pairs[0].skill:.2f}")
    events = []
    for rd in range(1, rounds + 1):
        parent = pairs[int(rng.integers(len(pairs)))]
        found = None
        for _try in range(tries):                                       # POET draws many candidates, keeps the learnable ones
            cp = mutate_params(parent.params, rng, scale=0.35 + 0.08 * _try)
            cenv = ParamCartPole(**cp)
            ctask = T.EnvTask(cenv, name=cenv.name)
            s_par = ctask.skill(ctask.evaluate_fn(parent.agent._act_fn(), 10, 424242)[0])
            if not (crit[0] <= s_par <= crit[1]):
                continue
            if min(np.linalg.norm(_norm(cp) - _norm(p.params)) for p in pairs) < min_novelty:
                continue
            found = (cp, ctask, s_par)
            break
        if found is None:
            log(f"[poet {rd}] no child of {parent.task.name} passed the minimal criterion + novelty in {tries} tries")
            continue
        cp, ctask, s_par = found
        g2 = random_genome(_PROGRAM_SPACE, rng)
        child_agent = ProgramPolicyAgent(ctask, g2, seed + rd)
        child_agent.pe.restore(parent.agent.pe.state())                 # start from the parent's programs
        child_agent.train(ctask, units, rng)
        child = Pair(cp, ctask, child_agent, parent=parent.task.name)
        child.skill = child_agent.val_score(ctask)[0]
        pairs.append(child)
        log(f"[poet {rd}] NEW environment {ctask.name}: parent agent skill {s_par:.2f} -> child agent skill {child.skill:.2f}")
        events.append(dict(round=rd, kind="new_env", env=ctask.name, parent=parent.task.name, skill=child.skill))
        if len(pairs) > max_envs:
            pairs.pop(0)
        # transfer: does any other agent beat the resident one here?
        for p in pairs:
            p.agent.train(p.task, units * 0.5, rng)
            p.skill = p.agent.val_score(p.task)[0]
            for q in pairs:
                if q is p:
                    continue
                s = p.task.skill(p.task.evaluate_fn(q.agent._act_fn(), 20, 12345)[0])
                if s > p.skill + 0.03:
                    p.agent.pe.restore(q.agent.pe.state())
                    p.skill = p.agent.val_score(p.task)[0]
                    log(f"[poet {rd}]   transfer: agent from {q.task.name} now leads on {p.task.name} (skill {p.skill:.2f})")
                    events.append(dict(round=rd, kind="transfer", src=q.task.name, dst=p.task.name, skill=p.skill))
    zoo = Zoo(workspace_root)
    out = []
    for p in pairs:
        sig = p.task.signature()
        test = p.agent.test_metrics(p.task)
        zoo.add(p.task.name, p.agent, dict(sig, skill=float(p.skill), test=test, task=p.task.name,
                                           origin="poet", parent=p.parent), make_champion=True)
        out.append(dict(env=p.task.name, params=p.params, skill=float(p.skill), test=test, program=p.agent.describe()))
    return dict(environments=out, events=events)
