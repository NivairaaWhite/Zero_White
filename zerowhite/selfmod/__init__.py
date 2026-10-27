"""Self-modification machinery. Importing this package is cheap on purpose: the heavy parts
(patcher, search) are imported explicitly so that `evolve.py` can depend on `selfmod.safety`
without creating an import cycle."""
