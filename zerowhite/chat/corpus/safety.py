"""safety.py - content filters applied before anything is trained on.

Two different things live here, on purpose:

1. A NON-NEGOTIABLE rule: a record in which sexual content occurs together with a minor or with a family member is always
   dropped. There is no configuration switch for it. (Public roleplay corpora such as PIPPA contain exactly this kind of
   sibling/parent roleplay, and this project's "big sister" use case is a wholesome companion.)
2. A configurable general filter (`filter.safety.drop_sexual`) for sexually explicit text, plus PII redaction.

The sexual-content detector is a lexicon + word-boundary matcher. It is deliberately conservative about *false positives*
(breast cancer, sex education, "sexual harassment policy", a kiss on the cheek all pass) which means it is also imperfect on
the other side: it WILL miss euphemistic or misspelled text. It reduces risk; it does not remove the need to look at your data.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple

# ---- lexicons: edit here (and only here) --------------------------------------------------------------------------
# STRONG: a single occurrence marks the record as sexually explicit.
_STRONG = [
    r"porn\w*", r"hentai", r"nsfw", r"erotic\w*", r"blow\s?jobs?", r"hand\s?jobs?", r"cum\s?shots?", r"dildos?",
    r"masturbat\w*", r"orgasm\w*", r"(?:have|had|having|has)\s+sex", r"sex\s+scenes?", r"sexual\s+intercourse",
    r"fuck(?:ing|ed|s)?\s+(?:me|you|her|him|them|my|your|hard|harder)", r"cock\s+(?:in|inside)", r"suck(?:ing|ed)?\s+(?:my|his|your)\s+(?:cock|dick)",
    r"(?:his|your|my)\s+(?:cock|dick)\b", r"(?:her|your|my)\s+(?:pussy|cunt)\b", r"ejaculat\w*", r"creampie", r"anal\s+sex",
    r"sex\s+slave",
]
# MILD: three different mild terms together also mark a record explicit; two mild terms + a minor/family term trigger rule 1.
_MILD = [
    r"penis", r"vagina\w*", r"nipples?", r"breasts?", r"clit\w*", r"moan(?:s|ed|ing)?", r"arous(?:ed|al|ing)", r"naked",
    r"nude", r"undress(?:ed|es|ing)?", r"lust(?:ful|y)?", r"horny", r"seduc(?:e|ed|es|tive|ing)", r"erection", r"panties",
    r"lingerie", r"strip(?:s|ped|ping)?\s+(?:off|naked|down)", r"sensual(?:ly)?", r"thrust(?:s|ed|ing)?", r"throbbing",
    r"wet\s+(?:with|between)", r"grop(?:e|ed|ing)", r"fondl\w*", r"caress(?:ed|es|ing)?\s+(?:her|his|your)\s+(?:body|breasts?|thighs?)",
]
_MINOR = [
    r"child(?:ren)?", r"kids?", r"minors?", r"under\s?-?age[d]?", r"pre\s?-?teens?", r"teen(?:s|ager|agers|age)?", r"school\s?(?:girl|boy)s?",
    r"loli\w*", r"shota\w*", r"little\s+(?:girl|boy)s?", r"young\s+(?:girl|boy)s?", r"toddlers?", r"infants?", r"juvenile",
    r"(?:[0-9]|1[0-7])\s?-?\s?(?:years?|yrs?)\s?-?\s?old", r"(?:[0-9]|1[0-7])\s?(?:y/?o|yo)\b", r"aged?\s+(?:[0-9]|1[0-7])\b",
    r"\b(?:[0-9]|1[0-7])\s?(?:yr|yrs)\b",
]
_FAMILY = [
    r"sisters?", r"brothers?", r"sibling\w*", r"mothers?", r"mom", r"mommy", r"mum", r"fathers?", r"dad", r"daddy",
    r"step\s?-?(?:sister|brother|mother|father|mom|dad|son|daughter|parent|sibling)s?", r"daughters?", r"sons?", r"aunt\w*",
    r"uncle\w*", r"cousins?", r"niece", r"nephew", r"grand(?:ma|pa|mother|father)\w*", r"onee\s?-?(?:chan|san)", r"onii\s?-?(?:chan|san)",
    r"imouto", r"oniisan", r"oneesan", r"nii\s?-?(?:san|chan)",
]


def _compile(words: Iterable[str]):
    return re.compile(r"(?<![A-Za-z0-9])(?:" + "|".join(words) + r")(?![A-Za-z0-9])", re.IGNORECASE)


_RE_STRONG, _RE_MILD, _RE_MINOR, _RE_FAMILY = _compile(_STRONG), _compile(_MILD), _compile(_MINOR), _compile(_FAMILY)
_EXCLUDE = re.compile(r"cum\s+laude|breast\s+(?:cancer|feeding|milk|stroke|pump)|sex(?:ual)?\s+(?:education|harassment|abuse prevention|"
                      r"discrimination|assault\s+prevention)|erotic\s+(?:poetry|literature)\s+of\s+ancient", re.IGNORECASE)


def _distinct(rx, text: str) -> int:
    return len({m.group(0).lower() for m in rx.finditer(text)})


def sexual_level(text: str) -> Tuple[int, int]:
    """(# distinct strong terms, # distinct mild terms) after removing known benign phrases."""
    t = _EXCLUDE.sub(" ", text)
    return _distinct(_RE_STRONG, t), _distinct(_RE_MILD, t)


def is_sexual(text: str) -> bool:
    s, m = sexual_level(text)
    return s >= 1 or m >= 3


def has_minor_or_family(text: str) -> bool:
    return bool(_RE_MINOR.search(text) or _RE_FAMILY.search(text))


def hard_violation(text: str) -> bool:
    """The always-on rule: sexual content together with a minor or a family member."""
    s, m = sexual_level(text)
    if not (s >= 1 or m >= 2):
        return False
    return has_minor_or_family(text)


# ---- PII ------------------------------------------------------------------------------------------------------------
_RE_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_RE_IP = re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])")
_RE_PHONE = re.compile(r"(?<![\w/])(?<!\d[\s.-])(?:\+\d[\d\s().-]{8,16}\d|\(?\d{3,4}\)?[\s.-]\d{3,4}[\s.-]\d{3,4})(?![\w/])(?![\s.-]\d)")
_RE_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_RE_DATE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _phone_sub(m):
    s = m.group(0)
    digits = re.sub(r"\D", "", s)
    if not (10 <= len(digits) <= 15) or _RE_DATE.match(s.strip()):
        return s
    return "<phone>"


def _card_sub(m):
    digits = re.sub(r"\D", "", m.group(0))
    return "<card>" if 13 <= len(digits) <= 19 and luhn_ok(digits) else m.group(0)


def redact(text: str, kinds: Iterable[str]) -> str:
    kinds = set(kinds)
    if "card" in kinds:
        text = _RE_CARD.sub(_card_sub, text)
    if "email" in kinds:
        text = _RE_EMAIL.sub("<email>", text)
    if "ip" in kinds:
        text = _RE_IP.sub("<ip>", text)
    if "phone" in kinds:
        text = _RE_PHONE.sub(_phone_sub, text)
    return text


class SafetyFilter:
    """Built from the effective config of one file/folder."""

    def __init__(self, cfg):
        s = cfg.filter.safety
        self.drop_sexual = bool(s.drop_sexual)
        self.redact_kinds = [k for k in s.redact]
        self.terms = [t for t in s.drop_terms]
        self.term_re = _compile([re.escape(t) for t in self.terms]) if self.terms else None
        try:
            self.extra_re = [re.compile(x, re.IGNORECASE) for x in s.drop_regex]
        except re.error as e:
            raise ValueError(f"filter.safety.drop_regex: invalid regular expression ({e})") from e

    def verdict(self, text: str) -> str:
        """'' when the text is acceptable, else the drop reason."""
        if hard_violation(text):
            return "safety_minor_family"
        if self.drop_sexual and is_sexual(text):
            return "safety_sexual"
        if self.term_re and self.term_re.search(text):
            return "safety_term"
        for rx in self.extra_re:
            if rx.search(text):
                return "safety_regex"
        return ""

    def redact_text(self, text: str) -> str:
        return redact(text, self.redact_kinds) if self.redact_kinds else text
