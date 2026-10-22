"""zerowhite.omnivore - devour anything without breaking.

    devour(paths, workspace)       identify by content, open archives safely, repair damage, build tasks, set aside the rest with a reason
    load_tasks(workspace)          the tasks that were built, ready for `Generalist.solve`
    Library                        the ledger / quarantine record / task cache on disk
    Limits                         every cap (derived from the machine by `substrate`)

Nothing here ever raises on what a file contains, moves or deletes a source file, or runs your code unless you pass allow_code=True.
See docs/OMNIVORE.md.
"""
from .devour import Digest, Entry, devour, load_tasks, default_limits  # noqa: F401
from .library import Library  # noqa: F401
from .limits import Limits  # noqa: F401
