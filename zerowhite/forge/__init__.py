"""forge - the general model: give it a task, it builds (and keeps improving) the agents for it.

  task.py        what a "task" is: supervised data, an environment, text, or image generation
  envs.py        extra pure-numpy environments, some parametrised (for open-ended co-evolution)
  families.py    the kinds of agent it can build; there is NO default agent in any of them
  prior.py       a learned generative model over (family, genome): what worked on similar tasks before
  zoo.py         persistent store of every agent built, with lineage and a readable description
  generalist.py  profile -> transfer from the zoo -> propose from the prior -> race -> keep the winner
"""
