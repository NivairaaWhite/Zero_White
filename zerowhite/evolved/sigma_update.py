"""EVOLVABLE HOOK: how the search step size (sigma) changes after each generation.

GENESIS version: do nothing (constant step size). Zero White may rewrite this file
(see `python -m zerowhite selfmod hooks`); every accepted rewrite is stored in the
history store and can be rolled back. Allowed: arithmetic, comparisons, `x if c else y`,
np./math. calls from the safety allow-list. No loops, imports or attribute tricks.

Arguments: sigma (float), improved (bool: did the best child beat its parent?), generation (int)
Returns:   the next sigma (float). The caller clips it to a safe range.
"""


def sigma_update(sigma, improved, generation):
    return sigma
