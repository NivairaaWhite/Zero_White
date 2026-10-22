"""
agents.py - AI agents Zero White builds, trains and evolves on its own engine.

Environments (pure numpy, no gym):
  gridworld   random start / goal / walls on an NxN grid, +1 at the goal and a
              small step cost. Reward is sparse, so exploration matters.
  cartpole    classic cart-pole balancing (same equations as the textbook
              version), +1 per step survived, 200 steps max.

PGAgent is an actor-critic: a policy MLP and a value MLP trained with policy
gradient + a learned baseline + an entropy bonus, optionally paid an INTRINSIC
bonus from a forward-dynamics curiosity model (curiosity.py, the
noreward-rl idea) for reaching transitions it cannot yet predict.

`evolve_agent` is the agent factory: it evolves the agent's genome (network
size AND learning rate / discount / entropy / curiosity weight), children
inherit trained weights when only training genes changed, and fitness is the
return on held-out evaluation episodes the agent never trained on.
"""

import collections
import copy
import json

import numpy as np

from . import fileio
from .curiosity import ForwardDynamicsCuriosity
from .engine import Tensor, mse_loss, policy_gradient_loss
from .evolve import Evolver, Gene
from .nn import Params, Adam, init_mlp, mlp, mlp_raw


# =========================================================================== environments
class GridWorld:
    """Reach the goal (+1). Observation = 3 stacked size x size planes (agent, goal, walls)."""
    n_actions = 4
    _MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, size=5, step_cost=0.02):
        self.size, self.step_cost = size, step_cost
        self.obs_dim = 3 * size * size
        self.max_steps = 4 * size
        self.name = f"gridworld{size}"

    def _reachable(self, walls, start, goal):
        seen, q = {start}, collections.deque([start])
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in self._MOVES:
                n = (r + dr, c + dc)
                if 0 <= n[0] < self.size and 0 <= n[1] < self.size and n not in walls and n not in seen:
                    seen.add(n)
                    q.append(n)
        return False

    def reset(self, rng):
        s = self.size
        cells = [(r, c) for r in range(s) for c in range(s)]
        while True:
            pick = [cells[i] for i in rng.choice(len(cells), size=2 + s, replace=False)]
            self.pos, self.goal, walls = pick[0], pick[1], set(pick[2:])
            if abs(self.pos[0] - self.goal[0]) + abs(self.pos[1] - self.goal[1]) >= 2 \
                    and self._reachable(walls, self.pos, self.goal):
                break
        self.walls, self.t = walls, 0
        return self._obs()

    def _obs(self):
        o = np.zeros((3, self.size, self.size))
        o[0][self.pos] = 1.0
        o[1][self.goal] = 1.0
        for w in self.walls:
            o[2][w] = 1.0
        return o.ravel()

    def step(self, a):
        dr, dc = self._MOVES[int(a)]
        n = (self.pos[0] + dr, self.pos[1] + dc)
        if 0 <= n[0] < self.size and 0 <= n[1] < self.size and n not in self.walls:
            self.pos = n
        self.t += 1
        if self.pos == self.goal:
            return self._obs(), 1.0, True, True
        return self._obs(), -self.step_cost, self.t >= self.max_steps, False


class CartPole:
    """Textbook cart-pole. Observation = (x, x_dot, theta, theta_dot)."""
    obs_dim, n_actions, max_steps, name = 4, 2, 200, "cartpole"

    def reset(self, rng):
        self.s = rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.s.copy()

    def step(self, a):
        x, xd, th, thd = self.s
        force = 10.0 if int(a) == 1 else -10.0
        cos, sin = np.cos(th), np.sin(th)
        tmp = (force + 0.05 * thd * thd * sin) / 1.1
        thacc = (9.8 * sin - cos * tmp) / (0.5 * (4.0 / 3.0 - 0.1 * cos * cos / 1.1))
        xacc = tmp - 0.05 * thacc * cos / 1.1
        tau = 0.02
        self.s = np.array([x + tau * xd, xd + tau * xacc, th + tau * thd, thd + tau * thacc])
        self.t += 1
        dead = abs(self.s[0]) > 2.4 or abs(self.s[2]) > 12 * np.pi / 180
        done = dead or self.t >= self.max_steps
        return self.s.copy(), 1.0, done, (self.t >= self.max_steps and not dead)


def make_env(name):
    if name.startswith("gridworld"):
        return GridWorld(int(name[len("gridworld"):] or 5))
    if name == "cartpole":
        return CartPole()
    try:
        from .forge.envs import make_extra_env
        return make_extra_env(name)
    except KeyError:
        raise KeyError(f"unknown environment '{name}' (gridworldN, cartpole, mountaincar, cartpole:g=..,len=..)") from None


# (updates, episodes per update) one evolution step trains each child for
ENV_BUDGET = {"gridworld5": (100, 16), "cartpole": (40, 8)}


# =========================================================================== agent
class PGAgent:
    kind = "pgagent"

    def __init__(self, obs_dim, n_actions, hidden=32, depth=1, lr=1e-2, gamma=0.97,
                 entropy=0.02, curiosity=0.0, seed=0):
        self.cfg = dict(obs_dim=obs_dim, n_actions=n_actions, hidden=hidden, depth=depth, lr=lr,
                        gamma=gamma, entropy=entropy, curiosity=curiosity)
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.p = Params()
        init_mlp(self.p, "pi", [obs_dim] + [hidden] * depth + [n_actions], rng, out_scale=0.1)
        init_mlp(self.p, "v", [obs_dim] + [hidden] * depth + [1], rng, out_scale=0.5)
        self.icm = ForwardDynamicsCuriosity(obs_dim, n_actions, hidden=max(16, hidden), seed=seed)
        self.opt, self.updates_trained = Adam(lr=lr, clip=5.0), 0
        self.bonus_scale = 1.0                      # running mean of the curiosity bonus, for normalisation

    # ---- acting ---------------------------------------------------------------------
    def policy(self, obs):
        z = mlp_raw(self.p.a, "pi", obs)
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def act(self, obs, rng, greedy=False):
        p = self.policy(obs[None])[0]
        return int(p.argmax()) if greedy else int(rng.choice(len(p), p=p))

    # ---- rollouts ---------------------------------------------------------------
    def _episode(self, env, rng, greedy=False):
        o = env.reset(rng)
        obs, acts, rews, nxt = [], [], [], []
        done, won = False, False
        while not done:
            a = self.act(o, rng, greedy)
            o2, r, done, won = env.step(a)
            obs.append(o); acts.append(a); rews.append(r); nxt.append(o2)
            o = o2
        return np.array(obs), np.array(acts), np.array(rews), np.array(nxt), bool(won)

    def evaluate(self, env, episodes=20, seed=12345, greedy=False):
        """Mean return and success rate of the policy AS TRAINED (stochastic) on a FIXED
        set of seeded episodes, so every genome is judged on the same ones. (Greedy
        play is available but can loop forever on gridworlds, so it is not the default.)"""
        rets, wins = [], 0
        for i in range(episodes):
            _, _, rews, _, won = self._episode(env, np.random.default_rng(seed + i), greedy)
            rets.append(rews.sum())
            wins += won
        return float(np.mean(rets)), wins / episodes

    # ---- learning ---------------------------------------------------------------
    def train(self, env, updates=40, episodes=8, rng=None):
        rng = rng or self.rng
        g, c = self.cfg["gamma"], self.cfg["curiosity"]
        last = {}
        for _ in range(updates):
            O, A, N, G, ext = [], [], [], [], []
            for _e in range(episodes):
                o, a, r, n, _ = self._episode(env, rng)
                ext.append(r.sum())
                if c > 0:
                    b = self.icm.bonus(o, a, n)
                    self.bonus_scale = 0.99 * self.bonus_scale + 0.01 * (float(b.mean()) + 1e-8)
                    r = r + c * b / self.bonus_scale * 0.05
                ret, run = np.zeros_like(r), 0.0
                for t in range(len(r) - 1, -1, -1):
                    run = r[t] + g * run
                    ret[t] = run
                O.append(o); A.append(a); N.append(n); G.append(ret)
            O, A, N, G = map(np.concatenate, (O, A, N, G))
            l = self.p.leaves()
            v = mlp(l, "v", Tensor(O)).reshape(-1)
            adv = G - v.data
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = policy_gradient_loss(mlp(l, "pi", Tensor(O)), A, adv, self.cfg["entropy"])
            loss = pg + mse_loss(v, G) * 0.5
            loss.backward()
            self.opt.step(self.p, l)
            if c > 0:
                self.icm.update(O, A, N)
            self.updates_trained += 1
            last = dict(loss=float(loss.data), train_return=float(np.mean(ext)))
        return last

    # ---- lifecycle ----------------------------------------------------------------
    def clone(self):
        c = copy.deepcopy(self)
        c.rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        return c

    def set_training(self, lr, gamma, entropy, curiosity):
        self.cfg.update(lr=lr, gamma=gamma, entropy=entropy, curiosity=curiosity)
        self.opt.lr = lr

    def genome(self):
        return {k: self.cfg[k] for k in ("hidden", "depth", "lr", "gamma", "entropy", "curiosity")}

    def save(self, path):
        arrays = {f"p.{k}": v for k, v in self.p.a.items()}
        arrays.update({f"c.{k}": v for k, v in self.icm.params.a.items()})
        meta = dict(cfg=self.cfg, updates=self.updates_trained, bonus_scale=self.bonus_scale)
        arrays["__cfg__"] = np.frombuffer(json.dumps(meta).encode("utf-8"), np.uint8)
        fileio.save_npz(path, arrays)

    @classmethod
    def load(cls, path):
        w = fileio.load_npz(path)
        meta = json.loads(bytes(w["__cfg__"]).decode("utf-8"))
        a = cls(**meta["cfg"])
        a.p = Params({k[2:]: w[k] for k in w if k.startswith("p.")})
        a.icm.params = Params({k[2:]: w[k] for k in w if k.startswith("c.")})
        a.updates_trained, a.bonus_scale = meta["updates"], meta["bonus_scale"]
        return a


AGENT_SPACE = {"hidden": Gene("int", 8, 96, shape=True), "depth": Gene("int", 1, 2, shape=True),
               "lr": Gene("log", 1e-3, 3e-2), "gamma": Gene("float", 0.9, 0.995),
               "entropy": Gene("log", 1e-3, 0.1), "curiosity": Gene("float", 0.0, 1.0)}


def evolve_agent(env_name, generations=3, population=4, updates=40, episodes=8, seed=0,
                 incumbent=None, verbose=True, eval_episodes=20):
    env = make_env(env_name)
    n = [0]

    def evaluate(genome, parent):
        if parent is not None:
            a = parent.clone()
            a.set_training(genome["lr"], genome["gamma"], genome["entropy"], genome["curiosity"])
        else:
            a = PGAgent(env.obs_dim, env.n_actions, seed=seed + n[0], **genome)
        n[0] += 1
        a.train(env, updates, episodes, rng=np.random.default_rng(seed * 1000 + n[0]))
        ret, succ = a.evaluate(env, eval_episodes)
        return ret, a, dict(eval_return=ret, success=succ, updates=a.updates_trained)

    ev = Evolver(AGENT_SPACE, evaluate, population=population, mu=2, seed=seed)
    if incumbent is not None:
        ret, succ = incumbent.evaluate(env, eval_episodes)
        ev.inject(incumbent.genome(), ret, incumbent, dict(eval_return=ret, success=succ,
                                                          updates=incumbent.updates_trained))
    # no incumbent: there is NO default agent. The first generation is a population of random genomes.
    ev.run(max(1, generations), verbose, label=f"agent/{env_name}")
    return ev
