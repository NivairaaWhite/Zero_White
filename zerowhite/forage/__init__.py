"""forage - an explicit, allow-listed way for Zero White to read a few web pages into its inbox. Off unless you name the hosts. See fetch.py."""
from .fetch import Forager, Policy, html_to_text  # noqa: F401
