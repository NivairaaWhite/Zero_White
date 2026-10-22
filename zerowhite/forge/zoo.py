"""zoo.py - persistent store of every agent the Generalist has built.

  <ws>/forge/zoo/<task>/<agent_id>/   the agent's own files + spec.json (family, genome, skill, parentage...)
  <ws>/forge/zoo/<task>/best.json     which agent is the current champion for that task
  <ws>/forge/prior.json               the learned prior (prior.py)

The champion is only ever replaced by something with a strictly higher validation skill, so a task's
best agent never gets worse. A few runners-up are kept for transfer; the rest are deleted.
"""

import os
import shutil
import time

from .. import fileio

KEEP = 3


class Zoo:
    def __init__(self, ws_root):
        self.root = os.path.join(os.path.abspath(ws_root), "forge", "zoo")
        os.makedirs(self.root, exist_ok=True)

    def task_dir(self, task_name):
        return os.path.join(self.root, _safe(task_name))

    def tasks(self):
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d)))

    def best(self, task_name):
        p = os.path.join(self.task_dir(task_name), "best.json")
        if not os.path.exists(p):
            return None
        try:
            rec = fileio.read_json(p)
        except ValueError:
            return None
        return rec if os.path.isdir(os.path.join(self.task_dir(task_name), rec["id"])) else None

    def spec(self, task_name, agent_id):
        return fileio.read_json(os.path.join(self.task_dir(task_name), agent_id, "spec.json"))

    def entries(self, task_name):
        d = self.task_dir(task_name)
        out = []
        for a in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            sp = os.path.join(d, a, "spec.json")
            if os.path.exists(sp):
                try:
                    out.append(dict(fileio.read_json(sp), id=a))
                except ValueError:
                    pass
        return out

    def add(self, task_name, agent, spec, make_champion):
        """Save `agent` under a new id. Returns the id."""
        d = self.task_dir(task_name)
        aid = f"{agent.family}-{int(time.time() * 1000) % 10 ** 10:010d}"
        ad = os.path.join(d, aid)
        os.makedirs(ad, exist_ok=True)
        agent.save(ad)
        fileio.write_json(os.path.join(ad, "spec.json"), dict(spec, family=agent.family, genome=agent.genome,
                                                            description=agent.describe(), created=time.time()),
                          indent=2)
        if make_champion:
            fileio.write_json(os.path.join(d, "best.json"), dict(id=aid, **{k: spec[k] for k in ("skill",) if k in spec}),
                              indent=2)
        self._prune(task_name)
        return aid

    def _prune(self, task_name):
        entries = sorted(self.entries(task_name), key=lambda e: -e.get("skill", -9))
        champ = (self.best(task_name) or {}).get("id")
        keep = {e["id"] for e in entries[:KEEP]} | ({champ} if champ else set())
        for e in entries:
            if e["id"] not in keep:
                shutil.rmtree(os.path.join(self.task_dir(task_name), e["id"]), ignore_errors=True)


def _safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))[:80]
