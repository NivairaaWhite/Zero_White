"""g2p.py - English text to phonemes (ARPAbet symbols), by spelling rules.

A small rule-based letter-to-sound system: a table of irregular common words, then greedy grapheme matching with a handful of
context rules (silent final e, soft c/g, vowel length from the following consonants, -tion, -ing ...). It is imperfect - English
spelling is - but good enough for the built-in robotic voice and for estimating speaking rate from transcripts. Other languages:
use an engine (espeak / SAPI / say / piper) that knows them.
"""
from __future__ import annotations

import re
from typing import List, Tuple

VOWELS = {"IY", "IH", "EH", "AE", "AA", "AO", "UH", "UW", "AH", "ER", "AX", "EY", "AY", "OY", "AW", "OW"}
LETTER_V = set("aeiouy")

# irregular / very frequent words; "0" after a vowel marks it unstressed
LEXICON = {
    "the": "DH AX", "a": "AX", "an": "AX N", "of": "AX V", "to": "T UW", "and": "AE N D", "in": "IH N", "is": "IH Z", "it": "IH T",
    "you": "Y UW", "that": "DH AE T", "he": "HH IY", "she": "SH IY", "was": "W AH Z", "for": "F AO R", "on": "AA N", "are": "AA R",
    "as": "AE Z", "with": "W IH DH", "his": "HH IH Z", "they": "DH EY", "i": "AY", "at": "AE T", "be": "B IY", "this": "DH IH S",
    "have": "HH AE V", "from": "F R AH M", "or": "AO R", "one": "W AH N", "had": "HH AE D", "by": "B AY", "word": "W ER D",
    "but": "B AH T", "not": "N AA T", "what": "W AH T", "all": "AO L", "were": "W ER", "we": "W IY", "when": "W EH N", "your": "Y AO R",
    "can": "K AE N", "said": "S EH D", "there": "DH EH R", "use": "Y UW Z", "each": "IY CH", "which": "W IH CH", "do": "D UW",
    "how": "HH AW", "their": "DH EH R", "if": "IH F", "will": "W IH L", "up": "AH P", "other": "AH DH ER", "about": "AX B AW T",
    "out": "AW T", "many": "M EH N IY", "then": "DH EH N", "them": "DH EH M", "these": "DH IY Z", "so": "S OW", "some": "S AH M",
    "her": "HH ER", "would": "W UH D", "make": "M EY K", "like": "L AY K", "him": "HH IH M", "into": "IH N T UW", "time": "T AY M",
    "has": "HH AE Z", "look": "L UH K", "two": "T UW", "more": "M AO R", "write": "R AY T", "go": "G OW", "see": "S IY",
    "number": "N AH M B ER", "no": "N OW", "way": "W EY", "could": "K UH D", "people": "P IY P AX L", "my": "M AY", "than": "DH AE N",
    "first": "F ER S T", "water": "W AO T ER", "been": "B IH N", "call": "K AO L", "who": "HH UW", "oil": "OY L", "its": "IH T S",
    "now": "N AW", "find": "F AY N D", "long": "L AO NG", "down": "D AW N", "day": "D EY", "did": "D IH D", "get": "G EH T",
    "come": "K AH M", "made": "M EY D", "may": "M EY", "part": "P AA R T", "over": "OW V ER", "new": "N UW", "sound": "S AW N D",
    "take": "T EY K", "only": "OW N L IY", "little": "L IH T AX L", "work": "W ER K", "know": "N OW", "place": "P L EY S",
    "year": "Y IY R", "live": "L IH V", "me": "M IY", "back": "B AE K", "give": "G IH V", "most": "M OW S T", "very": "V EH R IY",
    "after": "AE F T ER", "thing": "TH IH NG", "our": "AW ER", "just": "JH AH S T", "name": "N EY M", "good": "G UH D",
    "sentence": "S EH N T AX N S", "man": "M AE N", "think": "TH IH NG K", "say": "S EY", "great": "G R EY T", "where": "W EH R",
    "help": "HH EH L P", "through": "TH R UW", "much": "M AH CH", "before": "B IH F AO R", "line": "L AY N", "right": "R AY T",
    "too": "T UW", "mean": "M IY N", "old": "OW L D", "any": "EH N IY", "same": "S EY M", "tell": "T EH L", "boy": "B OY",
    "follow": "F AA L OW", "came": "K EY M", "want": "W AA N T", "show": "SH OW", "also": "AO L S OW", "around": "AX R AW N D",
    "form": "F AO R M", "three": "TH R IY", "small": "S M AO L", "set": "S EH T", "put": "P UH T", "end": "EH N D", "does": "D AH Z",
    "another": "AX N AH DH ER", "well": "W EH L", "large": "L AA R JH", "must": "M AH S T", "big": "B IH G", "even": "IY V AX N",
    "such": "S AH CH", "because": "B IH K AO Z", "turn": "T ER N", "here": "HH IY R", "why": "W AY", "ask": "AE S K", "went": "W EH N T",
    "men": "M EH N", "read": "R IY D", "need": "N IY D", "land": "L AE N D", "different": "D IH F ER AX N T", "home": "HH OW M",
    "us": "AH S", "move": "M UW V", "try": "T R AY", "kind": "K AY N D", "hand": "HH AE N D", "picture": "P IH K CH ER",
    "again": "AX G EH N", "change": "CH EY N JH", "off": "AO F", "play": "P L EY", "spell": "S P EH L", "air": "EH R", "away": "AX W EY",
    "animal": "AE N AX M AX L", "house": "HH AW S", "point": "P OY N T", "page": "P EY JH", "letter": "L EH T ER", "mother": "M AH DH ER",
    "answer": "AE N S ER", "found": "F AW N D", "study": "S T AH D IY", "still": "S T IH L", "learn": "L ER N", "should": "SH UH D",
    "world": "W ER L D", "high": "HH AY", "every": "EH V R IY", "near": "N IY R", "add": "AE D", "food": "F UW D", "between": "B IH T W IY N",
    "own": "OW N", "below": "B IH L OW", "country": "K AH N T R IY", "plant": "P L AE N T", "last": "L AE S T", "school": "S K UW L",
    "father": "F AA DH ER", "keep": "K IY P", "tree": "T R IY", "never": "N EH V ER", "start": "S T AA R T", "city": "S IH T IY",
    "earth": "ER TH", "eye": "AY", "light": "L AY T", "head": "HH EH D", "under": "AH N D ER", "story": "S T AO R IY", "saw": "S AO",
    "left": "L EH F T", "don't": "D OW N T", "few": "F Y UW", "while": "W AY L", "along": "AX L AO NG", "might": "M AY T", "close": "K L OW S",
    "something": "S AH M TH IH NG", "seem": "S IY M", "next": "N EH K S T", "hard": "HH AA R D", "open": "OW P AX N", "example": "IH G Z AE M P AX L",
    "begin": "B IH G IH N", "life": "L AY F", "always": "AO L W EY Z", "those": "DH OW Z", "both": "B OW TH", "paper": "P EY P ER",
    "together": "T AX G EH DH ER", "got": "G AA T", "group": "G R UW P", "often": "AO F AX N", "run": "R AH N", "important": "IH M P AO R T AX N T",
    "until": "AH N T IH L", "children": "CH IH L D R AX N", "side": "S AY D", "feet": "F IY T", "car": "K AA R", "mile": "M AY L",
    "night": "N AY T", "walk": "W AO K", "white": "W AY T", "sea": "S IY", "began": "B IH G AE N", "grow": "G R OW", "took": "T UH K",
    "river": "R IH V ER", "four": "F AO R", "carry": "K AE R IY", "state": "S T EY T", "once": "W AH N S", "book": "B UH K",
    "hear": "HH IY R", "stop": "S T AA P", "without": "W IH TH AW T", "second": "S EH K AX N D", "later": "L EY T ER", "miss": "M IH S",
    "idea": "AY D IY AX", "enough": "IH N AH F", "eat": "IY T", "face": "F EY S", "watch": "W AA CH", "far": "F AA R", "indian": "IH N D IY AX N",
    "really": "R IY L IY", "almost": "AO L M OW S T", "let": "L EH T", "above": "AX B AH V", "girl": "G ER L", "sometimes": "S AH M T AY M Z",
    "mountain": "M AW N T AX N", "cut": "K AH T", "young": "Y AH NG", "talk": "T AO K", "soon": "S UW N", "list": "L IH S T", "song": "S AO NG",
    "being": "B IY IH NG", "leave": "L IY V", "family": "F AE M AX L IY", "body": "B AA D IY", "music": "M Y UW Z IH K", "color": "K AH L ER",
    "stand": "S T AE N D", "sun": "S AH N", "questions": "K W EH S CH AX N Z", "fish": "F IH SH", "area": "EH R IY AX", "mark": "M AA R K",
    "dog": "D AO G", "horse": "HH AO R S", "birds": "B ER D Z", "problem": "P R AA B L AX M", "complete": "K AX M P L IY T", "room": "R UW M",
    "knew": "N UW", "since": "S IH N S", "ever": "EH V ER", "piece": "P IY S", "told": "T OW L D", "usually": "Y UW ZH UW AX L IY",
    "friends": "F R EH N D Z", "easy": "IY Z IY", "heard": "HH ER D", "order": "AO R D ER", "red": "R EH D", "door": "D AO R", "sure": "SH UH R",
    "become": "B IH K AH M", "top": "T AA P", "ship": "SH IH P", "across": "AX K R AO S", "today": "T AX D EY", "during": "D UH R IH NG",
    "short": "SH AO R T", "better": "B EH T ER", "best": "B EH S T", "however": "HH AW EH V ER", "low": "L OW", "hours": "AW ER Z",
    "black": "B L AE K", "products": "P R AA D AH K T S", "happened": "HH AE P AX N D", "whole": "HH OW L", "measure": "M EH ZH ER",
    "remember": "R IH M EH M B ER", "early": "ER L IY", "waves": "W EY V Z", "reached": "R IY CH T", "listen": "L IH S AX N", "wind": "W IH N D",
    "rock": "R AA K", "space": "S P EY S", "covered": "K AH V ER D", "fast": "F AE S T", "several": "S EH V R AX L", "hold": "HH OW L D",
    "himself": "HH IH M S EH L F", "toward": "T AO R D", "five": "F AY V", "step": "S T EH P", "morning": "M AO R N IH NG", "passed": "P AE S T",
    "vowel": "V AW AX L", "true": "T R UW", "hundred": "HH AH N D R AX D", "against": "AX G EH N S T", "pattern": "P AE T ER N",
    "numeral": "N UW M ER AX L", "table": "T EY B AX L", "north": "N AO R TH", "slowly": "S L OW L IY", "money": "M AH N IY", "map": "M AE P",
    "farm": "F AA R M", "pulled": "P UH L D", "draw": "D R AO", "voice": "V OY S", "seen": "S IY N", "cold": "K OW L D", "cried": "K R AY D",
    "plan": "P L AE N", "notice": "N OW T AX S", "south": "S AW TH", "sing": "S IH NG", "war": "W AO R", "ground": "G R AW N D", "fall": "F AO L",
    "king": "K IH NG", "town": "T AW N", "i'll": "AY L", "unit": "Y UW N AX T", "figure": "F IH G Y ER", "certain": "S ER T AX N", "field": "F IY L D",
    "travel": "T R AE V AX L", "wood": "W UH D", "fire": "F AY ER", "upon": "AX P AA N", "done": "D AH N", "english": "IH NG G L IH SH",
    "road": "R OW D", "half": "HH AE F", "ten": "T EH N", "fly": "F L AY", "gave": "G EY V", "box": "B AA K S", "finally": "F AY N AX L IY",
    "wait": "W EY T", "correct": "K ER EH K T", "oh": "OW", "quickly": "K W IH K L IY", "person": "P ER S AX N", "became": "B IH K EY M",
    "shown": "SH OW N", "minutes": "M IH N AX T S", "strong": "S T R AO NG", "verb": "V ER B", "stars": "S T AA R Z", "front": "F R AH N T",
    "feel": "F IY L", "fact": "F AE K T", "inches": "IH N CH AX Z", "street": "S T R IY T", "decided": "D IH S AY D AX D", "contain": "K AX N T EY N",
    "course": "K AO R S", "surface": "S ER F AX S", "produce": "P R AX D UW S", "building": "B IH L D IH NG", "ocean": "OW SH AX N",
    "class": "K L AE S", "note": "N OW T", "nothing": "N AH TH IH NG", "rest": "R EH S T", "carefully": "K EH R F AX L IY", "scientists": "S AY AX N T AX S T S",
    "inside": "IH N S AY D", "wheels": "W IY L Z", "stay": "S T EY", "green": "G R IY N", "known": "N OW N", "island": "AY L AX N D",
    "week": "W IY K", "less": "L EH S", "machine": "M AX SH IY N", "base": "B EY S", "ago": "AX G OW", "stood": "S T UH D", "plane": "P L EY N",
    "system": "S IH S T AX M", "behind": "B IH HH AY N D", "ran": "R AE N", "round": "R AW N D", "boat": "B OW T", "game": "G EY M",
    "force": "F AO R S", "brought": "B R AO T", "understand": "AH N D ER S T AE N D", "warm": "W AO R M", "common": "K AA M AX N",
    "bring": "B R IH NG", "explain": "IH K S P L EY N", "dry": "D R AY", "though": "DH OW", "language": "L AE NG G W AX JH", "shape": "SH EY P",
    "deep": "D IY P", "thousands": "TH AW Z AX N D Z", "yes": "Y EH S", "clear": "K L IY R", "equation": "IH K W EY ZH AX N", "yet": "Y EH T",
    "government": "G AH V ER N M AX N T", "filled": "F IH L D", "heat": "HH IY T", "full": "F UH L", "hot": "HH AA T", "check": "CH EH K",
    "object": "AA B JH EH K T", "am": "AE M", "rule": "R UW L", "among": "AX M AH NG", "noun": "N AW N", "power": "P AW ER", "cannot": "K AE N AA T",
    "able": "EY B AX L", "six": "S IH K S", "size": "S AY Z", "dark": "D AA R K", "ball": "B AO L", "material": "M AX T IH R IY AX L",
    "special": "S P EH SH AX L", "heavy": "HH EH V IY", "fine": "F AY N", "pair": "P EH R", "circle": "S ER K AX L", "include": "IH N K L UW D",
    "built": "B IH L T", "hello": "HH AX L OW", "thank": "TH AE NG K", "thanks": "TH AE NG K S", "please": "P L IY Z", "sorry": "S AA R IY",
    "computer": "K AX M P Y UW T ER", "zero": "Z IY R OW", "assistant": "AX S IH S T AX N T", "python": "P AY TH AA N", "code": "K OW D",
    "question": "K W EH S CH AX N", "yourself": "Y AO R S EH L F", "myself": "M AY S EH L F", "learning": "L ER N IH NG", "model": "M AA D AX L",
    "tips": "T IH P S", "three": "TH R IY", "twelve": "T W EH L V", "twenty": "T W EH N T IY", "eight": "EY T", "nine": "N AY N",
    "seven": "S EH V AX N", "eleven": "IH L EH V AX N", "thirty": "TH ER T IY", "forty": "F AO R T IY", "fifty": "F IH F T IY", "zero's": "Z IY R OW Z",
}
FUNCTION_WORDS = {"the", "a", "an", "of", "to", "and", "in", "is", "it", "for", "on", "at", "as", "be", "by", "or", "was", "are", "were",
                  "he", "she", "we", "you", "i", "me", "my", "his", "her", "its", "our", "your", "their", "that", "this", "with", "from", "but",
                  "not", "have", "has", "had", "do", "does", "so", "if", "than", "then", "them", "they", "will", "would", "can", "could", "should"}

DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
SPELL = {c: p for c, p in zip("abcdefghijklmnopqrstuvwxyz", [
    "EY", "B IY", "S IY", "D IY", "IY", "EH F", "JH IY", "EY CH", "AY", "JH EY", "K EY", "EH L", "EH M", "EH N", "OW", "P IY", "K Y UW",
    "AA R", "EH S", "T IY", "Y UW", "V IY", "D AH B AX L Y UW", "EH K S", "W AY", "Z IY"])}


def number_to_words(n: int) -> str:
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 10:
        return DIGITS[str(n)]
    if n < 20:
        return TEENS[n - 10]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + DIGITS[str(n % 10)])
    if n < 1000:
        return DIGITS[str(n // 100)] + " hundred" + ("" if n % 100 == 0 else " " + number_to_words(n % 100))
    if n < 1_000_000:
        return number_to_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + number_to_words(n % 1000))
    if n < 1_000_000_000:
        return number_to_words(n // 1_000_000) + " million" + ("" if n % 1_000_000 == 0 else " " + number_to_words(n % 1_000_000))
    return " ".join(DIGITS[c] for c in str(n))


# ---------------------------------------------------------------------------------------------------------------------
# letter-to-sound
# ---------------------------------------------------------------------------------------------------------------------
_CONS_PH = {"b": "B", "d": "D", "f": "F", "h": "HH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N", "p": "P", "r": "R", "t": "T",
            "v": "V", "w": "W", "z": "Z", "g": "G", "s": "S", "c": "K", "q": "K", "x": "K S", "y": "Y"}
_DIGRAPHS = [  # (pattern, phonemes) consonant and vowel-team graphemes, longest first
    ("tch", "CH"), ("dge", "JH"), ("sch", "S K"), ("igh", "AY"), ("eigh", "EY"), ("ough", "AO"), ("augh", "AO"),
    ("tion", "SH AX N"), ("sion", "ZH AX N"), ("cian", "SH AX N"), ("ture", "CH ER"),
    ("ch", "CH"), ("sh", "SH"), ("th", "TH"), ("ph", "F"), ("wh", "W"), ("ck", "K"), ("ng", "NG"), ("gh", ""), ("kn", "N"), ("wr", "R"),
    ("qu", "K W"), ("gn", "N"), ("mb", "M"),
    ("ee", "IY"), ("ea", "IY"), ("oo", "UW"), ("ou", "AW"), ("ow", "OW"), ("oi", "OY"), ("oy", "OY"), ("ai", "EY"), ("ay", "EY"),
    ("oa", "OW"), ("au", "AO"), ("aw", "AO"), ("ew", "UW"), ("eu", "UW"), ("ie", "IY"), ("ei", "EY"), ("ue", "UW"), ("ui", "UW"),
]


def _is_vowel(w: str, i: int) -> bool:
    return 0 <= i < len(w) and w[i] in LETTER_V


def _digraph(w: str, i: int, pat: str, ph: str) -> str:
    """Context adjustments of a grapheme's default pronunciation."""
    n = len(w)
    end = i + len(pat)
    after = w[end:]
    if pat == "ow":
        return "OW" if end >= n else "AW"
    if pat == "ie":
        return ("AY" if n <= 3 else "IY") if end >= n else "IY"
    if pat == "ea":
        return "EH" if after[:1] == "d" or after[:2] == "th" else "IY"
    if pat == "oo":
        return "UH" if after[:1] == "k" else "UW"
    if pat == "mb":
        return "M" if end >= n else "M B"
    return ph


def _word_phones(w: str) -> List[str]:
    if w in LEXICON:
        return LEXICON[w].split()
    if len(w) == 1 and w in SPELL:
        return SPELL[w].split()
    out: List[str] = []
    i, n = 0, len(w)
    while i < n:
        rest = w[i:]
        c = w[i]
        if i + 1 < n and w[i + 1] == c and c not in LETTER_V and c not in "cg":          # double consonant -> one sound
            i += 1
            continue
        hit = None
        for pat, ph in _DIGRAPHS:
            if rest.startswith(pat) and not (pat == "ture" and i == 0):
                hit = (pat, _digraph(w, i, pat, ph))
                break
        if hit:
            if hit[1]:
                out.extend(hit[1].split())
            i += len(hit[0])
            continue
        prev_v = _is_vowel(w, i - 1)
        nxt = w[i + 1] if i + 1 < n else ""
        if c in LETTER_V:
            ph = _vowel(w, i, prev_v)
            out.extend(ph)
            if ph == ["ER"] and nxt == "r":
                i += 1                                                      # er / ir / ur: the vowel and the r are one sound
        elif c == "c":
            out.append("S" if nxt in ("e", "i", "y") else "K")
        elif c == "g":
            soft = nxt in ("e", "i", "y") and rest[:3] not in ("get", "gil", "gif", "giv", "gig")
            out.append("JH" if soft else "G")
        elif c == "s":
            between = prev_v and (i == n - 1 or nxt in LETTER_V)
            out.append("Z" if between and not (i == n - 1 and w[i - 1:i] == "u") else "S")
        elif c == "x":
            out.extend(["K", "S"] if i > 0 else ["Z"])
        elif c == "h" and i > 0 and w[i - 1] in "ctsgw":
            pass
        elif c == "y":
            out.extend(_vowel(w, i, prev_v) if i > 0 else ["Y"])
        elif c in _CONS_PH:
            out.extend(_CONS_PH[c].split())
        i += 1
    return out


def _vowel(w: str, i: int, prev_v: bool) -> List[str]:
    c = w[i]
    n = len(w)
    nxt = w[i + 1] if i + 1 < n else ""
    nxt2 = w[i + 2] if i + 2 < n else ""
    last = i == n - 1
    has_earlier_vowel = any(ch in LETTER_V for ch in w[:i])
    if c == "e" and last and has_earlier_vowel and n > 2:                    # silent final e
        return []
    if c == "e" and i == n - 2 and nxt in ("s", "d") and n > 3 and has_earlier_vowel and w[i - 1] not in "tdsxzc":
        return []                                                           # walked, likes
    magic = bool(nxt) and nxt not in LETTER_V and nxt2 == "e" and (i + 3 >= n or w[i + 3:] in ("s", "d", "r", "ly", "ment", "ful", "less", "ness"))
    r_next = nxt == "r"
    if c == "a":
        if r_next:
            return ["EH"] if nxt2 == "e" else ["AA"]
        if nxt == "l" and (nxt2 in ("l", "k") or i + 2 >= n):
            return ["AO"]
        if magic:
            return ["EY"]
        return ["AX"] if (i > 0 and last) else ["AE"]
    if c == "e":
        if r_next and i > 0:
            return ["ER"]
        return ["IY"] if magic or (last and n <= 2) else ["EH"]
    if c == "i":
        if r_next:
            return ["ER"]
        if magic or (nxt == "n" and nxt2 == "d" and i + 3 >= n):
            return ["AY"]
        return ["IH"]
    if c == "o":
        if r_next:
            return ["AO"]
        if magic or (last and n <= 3) or (nxt == "l" and nxt2 == "d"):
            return ["OW"]
        return ["AA"]
    if c == "u":
        if r_next:
            return ["ER"]
        if magic or (last and i > 0 and n <= 3):
            return ["UW"]
        return ["AH"]
    if c == "y":
        if last:
            return ["IY"] if n > 3 else ["AY"]
        return ["IH"]
    return []


def word_to_phones(word: str) -> List[str]:
    w = re.sub(r"[^a-z']", "", word.lower())
    if not w:
        return []
    w = w.replace("'s", "s") if w not in LEXICON and w.endswith("'s") else w
    if "'" in w and w not in LEXICON:
        w = w.replace("'", "")
    ph = _word_phones(w)
    voiceless = {"P", "T", "K", "F", "TH", "CH", "SH", "S"}
    if len(ph) > 1 and len(w) > 2:
        if w.endswith("s") and not w.endswith(("ss", "us", "is", "ous")) and ph[-1] in ("S", "Z") and ph[-2] != "AX":
            ph[-1] = "S" if ph[-2] in voiceless else "Z"
        elif w.endswith("ed") and ph[-1] == "D" and ph[-2] in voiceless:
            ph[-1] = "T"
    # stress: first vowel of a polysyllabic content word is primary; function words and later vowels are reduced via the AX/lexicon table
    return ph


def is_function_word(word: str) -> bool:
    return word.lower() in FUNCTION_WORDS


def expand_text(text: str) -> List[Tuple[str, str]]:
    """Normalise text into [(token, kind)] where kind is 'w' (word) or a punctuation pause class: ',' '.' '?' '!'."""
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"(\d+)\.(\d+)", lambda m: m.group(1) + " point " + " ".join(DIGITS[c] for c in m.group(2)), text)
    text = re.sub(r"\d+", lambda m: " " + number_to_words(int(m.group(0))) + " " if len(m.group(0)) < 10 else " " + " ".join(DIGITS[c] for c in m.group(0)) + " ", text)
    text = text.replace("&", " and ").replace("%", " percent ").replace("+", " plus ").replace("=", " equals ")
    toks: List[Tuple[str, str]] = []
    for m in re.finditer(r"[A-Za-z']+|[,;:\-—]+|[.]+|[?]+|[!]+|\n+", text):
        s = m.group(0)
        if s[0].isalpha() or s[0] == "'":
            if s.strip("'"):
                toks.append((s, "w"))
        elif s[0] in ",;:-—":
            toks.append((",", ","))
        elif s[0] == ".":
            toks.append((".", "."))
        elif s[0] == "?":
            toks.append(("?", "?"))
        elif s[0] == "!":
            toks.append(("!", "!"))
        elif s[0] == "\n":
            toks.append((".", "."))
    return toks


def text_to_phones(text: str) -> int:
    """Number of phonemes in `text` (used to estimate speaking rate from a transcript)."""
    return sum(len(word_to_phones(t)) for t, k in expand_text(text) if k == "w")
