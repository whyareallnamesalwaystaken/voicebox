"""
English text normalization for TTS.

TTS engines are trained on plain prose. Hand them raw numbers, dates, file
paths, code or bare symbols and they guess, and the guesses come out as
garbled non-speech. Measured on Chatterbox Turbo before this module existed:

    "audio.py#L150 ... trim_tts_output()"  ->  "audio dot p i four one five
                                               for trim dot docs dot poot output"
    "11/02/2026"                           ->  "1102 -26"
    "2:1"                                  ->  "2 -1"

``normalize_for_tts`` rewrites those into the words a person would actually
say ("audio dot pie, line one hundred fifty", "November second, twenty
twenty-six", "two to one") before the text reaches any engine.

Rules are deliberately conservative: anything that already reads as normal
prose passes through untouched. Paralinguistic tags like ``[laugh]`` are
never modified. Only English is normalized; other languages pass through
as-is because the spoken forms here are English.

Set ``VOICEBOX_TEXT_NORMALIZE=0`` in the server environment to disable.
"""

from __future__ import annotations

import os
import re

# ---------------------------------------------------------------------------
# Number speller
# ---------------------------------------------------------------------------

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [
    (10**12, "trillion"),
    (10**9, "billion"),
    (10**6, "million"),
    (1000, "thousand"),
]
_ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_MONTHS = [
    "", "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]


def _under_thousand(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(f"{_ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        tens = _TENS[n // 10]
        parts.append(f"{tens}-{_ONES[n % 10]}" if n % 10 else tens)
    elif n or not parts:
        parts.append(_ONES[n])
    return " ".join(parts)


def number_to_words(n: int) -> str:
    """Cardinal words for an integer: 1250 -> 'one thousand two hundred fifty'."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n < 1000:
        return _under_thousand(n)
    if n >= 10**15:
        return digits_to_words(str(n))
    parts = []
    for value, name in _SCALES:
        if n >= value:
            parts.append(f"{_under_thousand(n // value)} {name}")
            n %= value
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def ordinal_to_words(n: int) -> str:
    """Ordinal words: 21 -> 'twenty-first'."""
    words = number_to_words(n)
    head, sep, last = words.rpartition("-") if "-" in words.split(" ")[-1] else words.rpartition(" ")
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head}{sep}{last}"


def digits_to_words(s: str) -> str:
    """Digit by digit: '0042' -> 'zero zero four two'."""
    return " ".join(_ONES[int(c)] for c in s if c.isdigit())


def year_to_words(n: int) -> str:
    """Year style: 1999 -> 'nineteen ninety-nine', 2026 -> 'twenty twenty-six'."""
    if 2000 <= n <= 2009:
        return number_to_words(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return f"{number_to_words(hi)} hundred"
    if lo < 10:
        return f"{number_to_words(hi)} oh {_ONES[lo]}"
    return f"{number_to_words(hi)} {number_to_words(lo)}"


def _int_words(raw: str) -> str:
    """Integer token (may contain thousands commas) to words, choosing digit-
    by-digit for things that are clearly codes rather than quantities."""
    plain = raw.replace(",", "")
    if not plain:
        return raw
    if "," not in raw and (len(plain) >= 7 or (len(plain) > 1 and plain[0] == "0")):
        return digits_to_words(plain)
    return number_to_words(int(plain))


def _decimal_words(raw: str) -> str:
    """'3.14' -> 'three point one four', '1,250.5' -> '... point five'."""
    whole, _, frac = raw.partition(".")
    head = _int_words(whole) if whole else "zero"
    return f"{head} point {digits_to_words(frac)}" if frac else head


def _number_words(raw: str) -> str:
    return _decimal_words(raw) if "." in raw else _int_words(raw)


# ---------------------------------------------------------------------------
# Identifiers, files and paths
# ---------------------------------------------------------------------------

# How extensions are said out loud. Anything not listed is spelled as letters
# when short (<=3 chars), otherwise read as a word.
_EXT_SPOKEN = {
    "py": "pie", "json": "jason", "txt": "text", "wav": "wave", "yaml": "yammel",
    "yml": "yammel", "toml": "tommel", "md": "M D", "js": "J S", "ts": "T S",
    "tsx": "T S X", "jsx": "J S X", "mp3": "M P three", "mp4": "M P four",
    "html": "H T M L", "css": "C S S", "exe": "E X E", "sh": "S H",
    "png": "P N G", "jpg": "J peg", "jpeg": "J peg", "pdf": "P D F", "csv": "C S V",
    "rs": "R S", "go": "go", "log": "log", "lock": "lock", "zip": "zip",
}
_FILE_EXTS = "|".join(sorted(_EXT_SPOKEN, key=len, reverse=True)) + "|ini|cfg|bat|ps1|db|env|xml|svg|gif|webp|bin|dll|cpp|java|kt|swift|rb|php"

_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_DIGIT_RUN_RE = re.compile(r"\d+")


def _speak_ext(ext: str) -> str:
    low = ext.lower()
    if low in _EXT_SPOKEN:
        return _EXT_SPOKEN[low]
    if len(low) <= 3:
        return " ".join(low.upper())
    return low


# Lowercase letter runs with no vowel (y counts) are initialisms when they
# show up in code or as bare tokens: "tts" -> "T T S", "npm" -> "N P M".
# Engines otherwise try to pronounce them ("tts" came out as "teeds").
_VOWELLESS_RE = re.compile(r"(?<![\w'])([b-df-hj-np-tv-xz]{2,5})(?![\w'])")
_VOWELLESS_KEEP = frozenset({"hmm", "hmmm", "shh", "shhh", "psst", "pst", "nth", "brr", "grr", "mm", "mmm", "tsk", "pfft", "zzz"})


# Common dev abbreviations that are said as a word, not spelled out.
_VOWELLESS_SPOKEN = {
    "src": "source", "cfg": "config", "pkg": "package", "msg": "message",
    "ctx": "context", "cmd": "command", "mgr": "manager", "svc": "service",
}


def _spell_vowelless(word: str) -> str:
    low = word.lower()
    if low in _VOWELLESS_KEEP:
        return word
    if low in _VOWELLESS_SPOKEN:
        return _VOWELLESS_SPOKEN[low]
    return " ".join(word.upper())


def _speak_word_part(part: str) -> str:
    """One identifier piece: split camelCase, spell digit runs and
    vowel-less initialisms."""
    pieces = _CAMEL_RE.sub(" ", part).split()
    out = []
    for p in pieces:
        p = _DIGIT_RUN_RE.sub(lambda m: f" {_int_words(m.group())} ", p)
        p = _VOWELLESS_RE.sub(lambda m: _spell_vowelless(m.group(1)), p)
        out.append(" ".join(p.split()))
    return " ".join(out)


def _speak_identifier(token: str, is_file: bool = False) -> str:
    """'trim_tts_output()' -> 'trim tts output', 'audio.py' -> 'audio dot pie'."""
    token = token.strip("()[]{}<>'\"`")
    if not token:
        return ""
    stem, ext = token, ""
    if is_file and "." in token:
        stem, _, ext = token.rpartition(".")
    segs = [s for s in re.split(r"\.", stem) if s]
    spoken = []
    for seg in segs:
        words = [_speak_word_part(w) for w in re.split(r"[_\-]+", seg) if w]
        spoken.append(" ".join(w for w in words if w))
    text = " dot ".join(s for s in spoken if s)
    if ext:
        text = f"{text} dot {_speak_ext(ext)}" if text else f"dot {_speak_ext(ext)}"
    return text


# Paths deeper than this are spoken as their last two parts only. Reading a
# full absolute path aloud is a long string of folder names nobody can follow
# by ear; the folder and file are what the listener actually needs.
_PATH_MAX_SPOKEN_PARTS = 3


def _speak_path(path: str) -> str:
    """'src/app.json' -> 'src, app dot jason';
    'C:/Users/me/proj/app.json' -> 'proj, app dot jason'."""
    path = path.rstrip(".,;:!?)")
    drive = ""
    m = re.match(r"^([A-Za-z]):[\\/]", path)
    if m:
        drive = f"{m.group(1).upper()} drive"
        path = path[3:]
    home = path.startswith("~")
    segs = [s for s in re.split(r"[\\/]+", path.lstrip("~")) if s and s not in (".",)]
    spoken = []
    if drive:
        spoken.append(drive)
    if home:
        spoken.append("home")
    for i, seg in enumerate(segs):
        if seg == "..":
            spoken.append("up one level")
            continue
        anchor = ""
        if "#" in seg:
            seg, _, anchor = seg.partition("#")
        is_file = bool(re.search(rf"\.(?:{_FILE_EXTS})$", seg, re.IGNORECASE))
        s = _speak_identifier(seg, is_file=is_file)
        if s:
            spoken.append(s)
        if anchor:
            spoken.append(_speak_anchor(anchor))
    if len(spoken) > _PATH_MAX_SPOKEN_PARTS:
        tail = spoken[-3:] if spoken[-1].startswith("line") else spoken[-2:]
        spoken = tail
    return ", ".join(spoken)


def _speak_anchor(anchor: str) -> str:
    m = re.fullmatch(r"L(\d+)(?:-L?(\d+))?", anchor)
    if m:
        if m.group(2):
            return f"lines {number_to_words(int(m.group(1)))} to {number_to_words(int(m.group(2)))}"
        return f"line {number_to_words(int(m.group(1)))}"
    return _speak_identifier(anchor)


# ---------------------------------------------------------------------------
# Rewrite rules (order matters: most specific first)
# ---------------------------------------------------------------------------

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_SUFFIX = {"k": "thousand", "K": "thousand", "m": "million", "M": "million",
           "b": "billion", "B": "billion", "bn": "billion", "T": "trillion"}
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"),
             "€": ("euro", "euros", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence")}
_ABBREV = [
    (r"\be\.g\.", "for example"),
    (r"\bi\.e\.", "that is"),
    (r"\betc\.", "et cetera"),
    (r"\bvs\.?(?=\s)", "versus"),
    (r"\bapprox\.", "approximately"),
]


def _currency(m: re.Match) -> str:
    sym, amount, suffix = m.group("sym"), m.group("amt"), m.group("suf")
    one, many, sub_one, sub_many = _CURRENCY[sym]
    if suffix:
        return f"{_number_words(amount)} {_SUFFIX[suffix]} {many}"
    whole, _, frac = amount.replace(",", "").partition(".")
    whole_n = int(whole) if whole else 0
    words = f"{number_to_words(whole_n)} {one if whole_n == 1 else many}"
    if frac:
        cents = int((frac + "0")[:2])
        if cents:
            sub = f"{number_to_words(cents)} {sub_one if cents == 1 else sub_many}"
            words = sub if whole_n == 0 else f"{words} and {sub}"
    return words


def _date_mdy(m: re.Match) -> str:
    mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    year = int(y) if len(y) == 4 else 2000 + int(y)
    return f"{_MONTHS[mo]} {ordinal_to_words(d)}, {year_to_words(year)}"


def _date_iso(m: re.Match) -> str:
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    return f"{_MONTHS[mo]} {ordinal_to_words(d)}, {year_to_words(y)}"


def _time_or_ratio(m: re.Match) -> str:
    a, b, ampm = m.group(1), m.group(2), m.group(3)
    if len(b) == 2 and int(a) <= 24 and int(b) < 60:
        h, mins = int(a), int(b)
        if mins == 0:
            spoken = f"{number_to_words(h)} o'clock" if not ampm else number_to_words(h)
        elif mins < 10:
            spoken = f"{number_to_words(h)} oh {_ONES[mins]}"
        else:
            spoken = f"{number_to_words(h)} {number_to_words(mins)}"
        if ampm:
            spoken += " " + " ".join(ampm.replace(".", "").lower())
        return spoken
    if ampm:
        return m.group(0)
    return f"{number_to_words(int(a))} to {number_to_words(int(b))}"


def _year_or_number(m: re.Match) -> str:
    raw = m.group(0)
    if "," not in raw and "." not in raw and len(raw) == 4 and 1100 <= int(raw) <= 2099:
        return year_to_words(int(raw))
    return _number_words(raw)


_PARA_TAG_RE = re.compile(r"(\[[^\]]*\])")
_URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)
# A path: drive-letter, ~/, ./, ../, /abs, or any token with 2+ separators,
# or a single separator next to a known file extension. Plain "and/or" and
# fractions are excluded (handled elsewhere).
_PATH_RE = re.compile(
    r"(?<![\w@])(?:[A-Za-z]:[\\/][^\s,;\"'<>|]*"
    r"|~?\.{0,2}[\\/][\w.\-~]+(?:[\\/][\w.\-~]*)*(?:#[\w\-]+)?"
    r"|[\w.\-]+(?:[\\/][\w.\-]+){2,}[\\/]?(?:#[\w\-]+)?"
    rf"|[\w.\-]+[\\/][\w.\-]+\.(?:{_FILE_EXTS})(?:#[\w\-]+)?)",
)
_FILE_RE = re.compile(rf"(?<![\w/\\])[\w\-]+(?:\.[\w\-]+)*\.(?:{_FILE_EXTS})(?:#[\w\-]+)?\b", re.IGNORECASE)
_SCOPED_PKG_RE = re.compile(r"(?<!\w)@([\w\-]+)/([\w.\-]+)")
_FUNC_CALL_RE = re.compile(r"\b([A-Za-z_][\w.]*)\(\)")
_SNAKE_RE = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")
_FLAG_RE = re.compile(r"(?<![\w-])(--?)([A-Za-z][\w\-]*)")


def _rewrite_prose(text: str) -> str:
    for pat, rep in _ABBREV:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    # Dates and versions before paths: "11/02/2026" and "v2.0.1" look like
    # paths/files to the rules below.
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b", _date_mdy, text)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", _date_iso, text)
    text = re.sub(r"\b[vV](\d+(?:\.\d+)+)\b", lambda m: "version " + _version_words(m.group(1)), text)
    text = re.sub(r"(?<![\w.])(\d+(?:\.\d+){2,})\b", lambda m: _version_words(m.group(1)), text)
    text = re.sub(r"\band/or\b", "and or", text, flags=re.IGNORECASE)

    # Links and paths next: they contain digits and symbols the number and
    # symbol rules below would otherwise chew up.
    text = _URL_RE.sub(lambda m: _speak_path(re.sub(r"^(?:https?://)?(?:www\.)?", "", m.group(0), flags=re.I)), text)
    text = _SCOPED_PKG_RE.sub(lambda m: f"at {_speak_identifier(m.group(1))}, {_speak_identifier(m.group(2))}", text)
    text = _PATH_RE.sub(lambda m: _speak_path(m.group(0)) + _trailing_punct(m.group(0)), text)
    text = _FILE_RE.sub(lambda m: _speak_file_token(m.group(0)), text)
    text = _FUNC_CALL_RE.sub(lambda m: _speak_identifier(m.group(1)), text)
    text = _SNAKE_RE.sub(lambda m: _speak_identifier(m.group(0)), text)

    # Code-ish operators and anchors.
    text = re.sub(r"\s*&&\s*", " and then ", text)
    text = re.sub(r"\s*\|\|\s*", " or else ", text)
    text = re.sub(r"#L(\d+)\b", lambda m: f" line {number_to_words(int(m.group(1)))}", text)
    text = re.sub(r"(?<![\w&])#(\d+)\b", lambda m: f"number {number_to_words(int(m.group(1)))}", text)
    text = _FLAG_RE.sub(lambda m: ("dash dash " if m.group(1) == "--" else "dash ") + _speak_identifier(m.group(2)), text)

    # Money, percents, dates, times: before plain numbers.
    text = re.sub(
        rf"(?P<sym>[$€£])\s?(?P<amt>{_NUM})(?P<suf>bn|[kKmMbBT])?\b",
        _currency, text,
    )
    text = re.sub(rf"({_NUM})\s?%", lambda m: f"{_number_words(m.group(1))} percent", text)
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s?°\s?([CF])\b",
        lambda m: f"{_number_words(m.group(1))} degrees " + ("Celsius" if m.group(2) == "C" else "Fahrenheit"),
        text,
    )
    text = re.sub(r"\b(\d{1,2}):(\d{1,2})(?:\s?([AaPp]\.?[Mm]\.?))?(?!\d)", _time_or_ratio, text)
    text = re.sub(r"\b(\d+)/(\d+)\b", _fraction, text)
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal_to_words(int(m.group(1))), text, flags=re.I)
    text = re.sub(rf"\b({_NUM})(bn|[kKmMbBT])\b", lambda m: f"{_number_words(m.group(1))} {_SUFFIX[m.group(2)]}", text)
    text = re.sub(r"\b(\d+)\s?(?:-|–|to)\s?(\d+)\b(?![-/.]\d)", _range, text)
    text = re.sub(r"(?<![\w.])-(\d)", r"minus \1", text)
    text = re.sub(rf"\b(?:{_NUM})\b", _year_or_number, text)

    # Bare initialisms ("npm"), after the time rule has consumed "am"/"pm".
    text = _VOWELLESS_RE.sub(lambda m: _spell_vowelless(m.group(1)), text)

    # Remaining bare symbols.
    for pat, rep in (
        (r"\s*(?:->|=>|→)\s*", " to "),
        (r"(?<=\s)&(?=\s)|\s&\s?", " and "),
        (r"\s?\+\s?(?=\w)", " plus "),
        (r"\s=\s|(?<=\w)=(?=\w)", " equals "),
        (r"°", " degrees "),
        (r"(?<=\s)~(?=\s?\w)", "about "),
        (r"(?<=\s)@(?=\w)", "at "),
        (r"\s>=\s", " at least "),
        (r"\s<=\s", " at most "),
        (r"\s>\s", " greater than "),
        (r"\s<\s", " less than "),
        (r"(?<=\w)/(?=\w)", " or "),
        (r"[\\|^{}<>_~`#]", " "),
    ):
        text = re.sub(pat, rep, text)

    return re.sub(r"[ \t]{2,}", " ", text)


def _version_words(raw: str) -> str:
    """'2.0.1' -> 'two point oh point one'."""
    return " point ".join("oh" if p == "0" else _int_words(p) for p in raw.split("."))


def _fraction(m: re.Match) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    if a == b:
        return f"{number_to_words(a)} {number_to_words(b)}"  # 50/50
    return f"{number_to_words(a)} over {number_to_words(b)}"


def _trailing_punct(token: str) -> str:
    m = re.search(r"[.,;:!?)]+$", token)
    return m.group(0) if m else ""


def _speak_file_token(token: str) -> str:
    anchor = ""
    if "#" in token:
        token, _, anchor = token.partition("#")
    spoken = _speak_identifier(token, is_file=True)
    return f"{spoken}, {_speak_anchor(anchor)}" if anchor else spoken


def _range(m: re.Match) -> str:
    a, b = int(m.group(1)), int(m.group(2))
    return f"{number_to_words(a)} to {number_to_words(b)}"


def normalization_enabled() -> bool:
    return os.environ.get("VOICEBOX_TEXT_NORMALIZE", "1").strip().lower() not in ("0", "false", "no", "off")


def normalize_for_tts(text: str, language: str = "en") -> str:
    """Rewrite numbers, symbols, paths and code into speakable English.

    Non-English text and paralinguistic ``[tags]`` are returned untouched.
    Never raises: on any internal error the original text is returned so a
    normalizer bug can only cost pronunciation, never a generation.
    """
    if not text or not (language or "en").lower().startswith("en"):
        return text
    try:
        parts = _PARA_TAG_RE.split(text)
        out = [p if _PARA_TAG_RE.fullmatch(p) else _rewrite_prose(p) for p in parts]
        joined = "".join(out)
        return re.sub(r"[ \t]{2,}", " ", joined).strip()
    except Exception:  # pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).exception("normalize_for_tts failed; using raw text")
        return text
