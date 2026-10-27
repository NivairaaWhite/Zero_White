"""EVOLVABLE HOOK: which pool member the next child is mutated from.

GENESIS version: uniformly at random. Zero White may rewrite this file.

Arguments: fitnesses (1-D float array, sorted best first), rng (numpy Generator)
Returns:   an integer index into `fitnesses`. The caller clamps it into range.
"""


def select_parent(fitnesses, rng):
    return int(rng.integers(len(fitnesses)))
