"""forage/webenv.py - a web API as a reinforcement-learning ENVIRONMENT, so a policy-gradient agent or an evolved program can learn to use it by trial and error.

The learner's world is a short menu of ACTIONS written by a person (`ActionTemplate`: method, path, JSON body, headers, with `Ref` placeholders for values the server gave earlier and for the
person's secrets). It picks an INDEX. It never writes a URL, a header or a body: the environment sends the template through a `Session` -> `Actor` (act.py), which enforces the person's `ActPolicy`.
What it sees is a small vector computed from the session: the class of the last HTTP status, which named values (token, csrf, id, ...) it holds, whether it has a cookie, how far into the episode it is.
What it is paid is defined here, from the SERVER'S OWN CLAIMS: +`goal_reward` when `goal(session)` is true, `-step_cost` per request, `-error_cost` extra for a refused / failed / 4xx / 5xx request.

    env = WebEnv(session, ACTIONS, goal=lambda s: s.memory.get("qty") == 5, slots=("token", "csrf", "id"))
    task = EnvTask(env)                     # any agent family that works on environments
    mock_task(policy, ...)                  # the whole thing wired to the loopback practice server (mockapi.py)

Safety properties that are tests, not hopes: the learner cannot reach the policy or the secrets; the request budget of the ACTOR (a person's number) ends the run for good (`budget_exhausted`; the environment then answers
every step with `done` and sends nothing); a refused action costs the learner reward and the server nothing (it never arrives). If the server's responses are hostile (a `next` link to another host, a token with
control characters) the values are checked where they are used and the action is refused. The reward is only as honest as the server: for a real service write `goal` so that it checks something the server cannot
fake cheaply, or verify with a separate read.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .act import Actor, ActPolicy, Ref, Response, Session

STATUS_CLASSES = ("none", "2xx", "3xx", "auth", "client", "limit", "server", "refused")      # the last response, one-hot


def status_class(resp: Optional[Response]) -> int:
    if resp is None:
        return 0
    if resp.refused:
        return 7
    if not resp.sent:
        return 0
    s = resp.status
    if 200 <= s < 300:
        return 1
    if 300 <= s < 400:
        return 2
    if s in (401, 403):
        return 3
    if s == 429:
        return 5
    if 400 <= s < 500:
        return 4
    return 6 if s >= 500 else 0


@dataclass(frozen=True)
class ActionTemplate:
    """One item of the menu. `path` is a string, a Ref('slot') (a whole link an earlier response gave) or a list of segments; `json` / `headers` / `query` may contain Refs."""
    name: str
    method: str
    path: object
    json: object = None
    headers: dict = field(default_factory=dict)
    query: dict = field(default_factory=dict)


class WebEnv:
    """Environment protocol of the project: obs_dim, n_actions, reset(rng) -> obs, step(a) -> (obs, reward, done, success)."""

    def __init__(self, session: Session, actions: Sequence[ActionTemplate], goal: Callable[[Session], bool], slots: Sequence[str] = ("token", "csrf", "id"),
                 max_steps: int = 20, step_cost: float = 0.02, error_cost: float = 0.05, goal_reward: float = 1.0, reset_hook: Optional[Callable[[], None]] = None,
                 name: str = "web", min_steps: Optional[int] = None):
        if not actions:
            raise ValueError("a WebEnv needs at least one action")
        self.session, self.actions, self.goal, self.slots = session, list(actions), goal, tuple(slots)
        self.max_steps, self.step_cost, self.error_cost, self.goal_reward = int(max_steps), float(step_cost), float(error_cost), float(goal_reward)
        self.reset_hook, self.name = reset_hook, name
        self.n_actions = len(self.actions)
        self.obs_dim = len(STATUS_CLASSES) + len(self.slots) + 2
        self.max_return = self.goal_reward - self.step_cost * (min_steps if min_steps is not None else 1)
        self.t = 0
        self.last: Optional[Response] = None
        self.exhausted = False                                    # the actor's budget (a person's number) is used up: nothing more is sent, ever
        self.history = []                                         # (action name, status class name) of the current episode, for people to read

    # ---- observation
    def _obs(self) -> np.ndarray:
        o = np.zeros(self.obs_dim)
        o[status_class(self.last)] = 1.0
        k = len(STATUS_CLASSES)
        for i, s in enumerate(self.slots):
            o[k + i] = 1.0 if s in self.session.memory else 0.0
        o[k + len(self.slots)] = 1.0 if self.session.cookies else 0.0
        o[k + len(self.slots) + 1] = self.t / max(1, self.max_steps)
        return o

    # ---- protocol
    def _budget_left(self) -> bool:
        a = self.session.actor
        return a.n_requests < a.policy.max_requests

    def reset(self, rng=None):
        self.session.reset()
        if self.reset_hook is not None:
            self.reset_hook()
        self.t, self.last, self.history = 0, None, []
        self.exhausted = not self._budget_left()
        return self._obs()

    def step(self, a):
        if self.exhausted or not self._budget_left():
            self.exhausted = True
            return self._obs(), 0.0, True, False
        act = self.actions[int(a) % self.n_actions]
        resp = self.session.request(act.method, act.path, json=act.json, headers=act.headers or None, query=act.query or None)
        self.last = resp
        self.t += 1
        self.history.append((act.name, STATUS_CLASSES[status_class(resp)]))
        r = -self.step_cost
        cls = status_class(resp)
        if cls in (3, 4, 5, 6, 7) or (cls == 0 and not resp.sent):
            r -= self.error_cost
        won = False
        try:
            won = bool(self.goal(self.session))
        except Exception:                                         # noqa: BLE001 - a goal function that cannot decide is 'not yet'
            won = False
        if won:
            r += self.goal_reward
        done = won or self.t >= self.max_steps
        return self._obs(), r, done, won

    def action_names(self) -> list:
        return [a.name for a in self.actions]


# =========================================================================== the practice task on the loopback mock server
def mock_actions() -> list:
    token = Ref("slot", "token", "Bearer {}")
    auth = {"Authorization": token}
    csrf = {"Authorization": token, "X-CSRF-Token": Ref("slot", "csrf")}
    return [
        ActionTemplate("health", "GET", "/health"),
        ActionTemplate("login", "POST", "/login", json={"user": "alice", "password": Ref("secret", "password")}),
        ActionTemplate("login_guess", "POST", "/login", json={"user": "alice", "password": "hunter2"}),
        ActionTemplate("me", "GET", "/me", headers=auth),
        ActionTemplate("get_csrf", "GET", "/csrf", headers=auth),
        ActionTemplate("create_item", "POST", "/items", json={"name": "widget", "qty": 3}, headers=csrf),
        ActionTemplate("create_no_csrf", "POST", "/items", json={"name": "widget", "qty": 3}, headers=auth),
        ActionTemplate("list_items", "GET", "/items", headers=auth),
        ActionTemplate("list_next", "GET", Ref("slot", "next"), headers=auth),
        ActionTemplate("set_qty_5", "PUT", ["items", Ref("slot", "id")], json={"qty": 5}, headers=csrf),
        ActionTemplate("delete_item", "DELETE", ["items", Ref("slot", "id")], headers=csrf),
        ActionTemplate("follow_redirect_away", "GET", "/redirect-away"),
    ]


def mock_goal(session: Session) -> bool:
    """The item the learner created has quantity 5, as the server last reported it."""
    m = session.memory
    return m.get("name") == "widget" and m.get("qty") == 5


def mock_env(server_url: str, state_reset: Callable[[], None], policy: Optional[ActPolicy] = None, password: str = "correct horse battery", ledger_path: Optional[str] = None,
             max_steps: int = 20, dry_run: bool = False) -> WebEnv:
    """The practice task: log in with the person's password (a secret the learner can only name), fetch the CSRF value, create a 'widget', set its quantity to 5.
    The shortest solution is 4 requests (login, get_csrf, create_item, set_qty_5)."""
    pol = policy or ActPolicy(write_hosts=("127.0.0.1",), max_requests=20000, max_requests_per_episode=max_steps)
    actor = Actor(server_url, pol, ledger_path=ledger_path, secrets={"password": password}, dry_run=dry_run)
    return WebEnv(Session(actor), mock_actions(), mock_goal, slots=("token", "csrf", "id", "next"), max_steps=max_steps, reset_hook=state_reset, name="web:mock", min_steps=4)
