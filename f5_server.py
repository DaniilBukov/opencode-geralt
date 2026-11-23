"""Persistent F5-TTS Russian worker.

Loads the model once and serves POST /tts with JSON {"text": "..."} -> audio/wav.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent


def _path_from_env(name: str, *defaults: Path) -> Path:
    """Explicit env override first, then the first default that exists."""
    value = os.environ.get(name, "").strip()
    if value:
        return Path(value).expanduser().resolve()
    for candidate in defaults:
        if candidate.exists():
            return candidate
    return defaults[0]


MODEL_DIR = PLUGIN_DIR / "models" / "f5"
CKPT = _path_from_env(
    "F5_CKPT",
    MODEL_DIR / "model_last_inference.safetensors",
    PLUGIN_DIR / "engines" / "f5" / "ckpt" / "model_last_inference.safetensors",
)
VOCAB = _path_from_env(
    "F5_VOCAB",
    MODEL_DIR / "vocab.txt",
    PLUGIN_DIR / "engines" / "f5" / "ckpt" / "vocab.txt",
)
REF = _path_from_env(
    "F5_REF",
    PLUGIN_DIR / "assets" / "ref" / "0x0011bb5c.wav",
    PLUGIN_DIR / "engines" / "witcher3-tools" / "geralt-wav" / "0x0011bb5c.wav",
)
REF_TEXT = os.environ.get("F5_REF_TEXT", "").strip() or (
    "А может, я предпочту путь ведьмака, чтобы лежать по ночам на траве "
    "и смотреть на звезды."
)
DEVICE = os.environ.get("F5_DEVICE", "").strip()
PORT = int(os.environ.get("F5_PORT", "8766"))
SPEED = float(os.environ.get("F5_SPEED", "1.0"))
NFE = int(os.environ.get("F5_NFE", "32"))


def resolve_device() -> str:
    if DEVICE:
        return DEVICE
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"

sys.path.insert(0, str(PLUGIN_DIR))

_tts = None
_ref_text = None
_pronounce = None


def load_pronounce() -> dict:
    import re

    path = PLUGIN_DIR / "f5_pronounce.json"
    if not path.exists():
        return {}
    raw = __import__("json").loads(path.read_text(encoding="utf-8"))

    def word_pattern(word: str):
        body = r"\+?".join(re.escape(ch) for ch in word)
        return re.compile(rf"(?<![А-Яа-яЁёA-Za-z]){body}(?![А-Яа-яЁёA-Za-z])", re.IGNORECASE)

    return {word_pattern(key): value for key, value in raw.items()}


def apply_pronounce(text: str) -> str:
    if not _pronounce:
        return text
    result = text
    for pattern, value in _pronounce.items():
        result = pattern.sub(value, result)
    return result


# ---------------------------------------------------------------- latin -> cyrillic
# Curated Russian pronunciation for terms that show up in answers. Keys are lowercase,
# '_', '-' and '.' are stripped before lookup.
LATIN_WORDS = {
    "opencode": "оупенкоуд",
    "code": "код",
    "cache": "кэш",
    "caches": "кэши",
    "server": "сервер",
    "client": "клиент",
    "plugin": "плагин",
    "plugins": "плагины",
    "config": "конфиг",
    "configs": "конфиги",
    "file": "файл",
    "files": "файлы",
    "folder": "фолдер",
    "path": "путь",
    "root": "рут",
    "user": "юзер",
    "admin": "админ",
    "default": "дефолт",
    "log": "лог",
    "logs": "логи",
    "test": "тест",
    "tests": "тесты",
    "batch": "бэтч",
    "script": "скрипт",
    "scripts": "скрипты",
    "session": "сессия",
    "stream": "стрим",
    "build": "билд",
    "deploy": "деплой",
    "release": "релиз",
    "feature": "фича",
    "bug": "баг",
    "fix": "фикс",
    "commit": "коммит",
    "branch": "бранч",
    "merge": "мёрдж",
    "push": "пуш",
    "pull": "пул",
    "request": "реквест",
    "response": "респонс",
    "error": "эррор",
    "timeout": "таймаут",
    "thread": "тред",
    "queue": "кью",
    "worker": "воркер",
    "daemon": "демон",
    "backup": "бэкап",
    "bootstrap": "бутстрап",
    "update": "апдейт",
    "upgrade": "апгрейд",
    "install": "инсталл",
    "download": "даунлоуд",
    "upload": "аплоуд",
    "browser": "браузер",
    "window": "окно",
    "button": "кнопка",
    "click": "клик",
    "mode": "режим",
    "speed": "спид",
    "voice": "войс",
    "audio": "аудио",
    "video": "видео",
    "image": "имидж",
    "text": "текст",
    "model": "модель",
    "models": "модели",
    "sample": "сэмпл",
    "dataset": "датасет",
    "training": "тренинг",
    "checkpoint": "чекпойнт",
    "epoch": "эпоха",
    "inference": "инференс",
    "latency": "латентность",
    "memory": "память",
    "disk": "диск",
    "size": "размер",
    "prompt": "промпт",
    "prompts": "промпты",
    "token": "токен",
    "tokens": "токены",
    "python": "пайтон",
    "torch": "торч",
    "pytorch": "пайторч",
    "cuda": "куда",
    "nvidia": "энвидиа",
    "whisper": "уиспер",
    "windows": "виндоус",
    "linux": "линукс",
    "docker": "докер",
    "github": "гитхаб",
    "git": "гит",
    "node": "ноуд",
    "javascript": "джаваскрипт",
    "typescript": "тайпскрипт",
    "json": "джейсон",
    "yaml": "ямл",
    "toml": "томл",
    "bash": "бэш",
    "powershell": "пауэршелл",
    "telegram": "телеграм",
    "discord": "дискорд",
    "google": "гугл",
    "youtube": "ютуб",
    "chrome": "хром",
    "firefox": "файрфокс",
    "vscode": "виэс код",
    "nginx": "энджинкс",
    "ruaccent": "руакцент",
    "f5": "эф пять",
    "tts": "ти ти эс",
    "api": "эй пи ай",
    "gpu": "джи пи ю",
    "cpu": "си пи ю",
    "vram": "вирам",
    "ram": "рам",
    "ssd": "эс эс ди",
    "hdd": "эйч ди ди",
    "usb": "ю эс би",
    "ide": "ай ди",
    "os": "о эс",
    "sdk": "эс ди кей",
    "cli": "си эл ай",
    "gui": "гуи",
    "llm": "эл эл эм",
    "ai": "эй ай",
    "ml": "эм эл",
    "url": "ю ар эл",
    "http": "эйч ти ти пи",
    "https": "эйч ти ти пи эс",
    "ssh": "эс эс эйч",
    "pdf": "пэ дэ эф",
    "csv": "си эс ви",
    "svg": "эс ви джи",
    "png": "пэ эн джи",
    "ui": "ю ай",
    "ux": "ю экс",
    "svc": "эс ви си",
    "nfe": "эн эф и",
    "cfg": "си эф джи",
    "strength": "стренс",
    "step": "стэп",
    "steps": "стэпы",
    "seed": "сид",
    "chunk": "чанк",
    "buffer": "буфер",
    "port": "порт",
    "host": "хост",
    "iphone": "айфон",
    "chatgpt": "чат джи пи ти",
    "openai": "оупен эй ай",
    "claude": "клауд",
    "gemini": "джемини",
    "deepseek": "дипсик",
}

LETTER_NAMES = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи",
    "h": "эйч", "i": "ай", "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн",
    "o": "оу", "p": "пи", "q": "кью", "r": "ар", "s": "эс", "t": "ти", "u": "ю",
    "v": "ви", "w": "дабл ю", "x": "экс", "y": "уай", "z": "зед",
}

_DIGRAPHS = [
    ("tion", "шн"), ("sion", "жн"), ("ough", "оу"), ("augh", "о"),
    ("eigh", "эй"), ("igh", "ай"), ("tch", "ч"), ("dge", "дж"),
    ("ch", "ч"), ("sh", "ш"), ("th", "с"), ("ph", "ф"), ("wh", "в"),
    ("ck", "к"), ("kn", "н"), ("wr", "р"), ("qu", "кв"), ("gh", "г"),
    ("oo", "у"), ("ee", "и"), ("ea", "и"), ("oa", "о"), ("ai", "эй"),
    ("ay", "эй"), ("oi", "ой"), ("oy", "ой"), ("ou", "ау"), ("ow", "оу"),
    ("au", "о"), ("aw", "о"), ("ew", "ю"), ("ie", "и"), ("ei", "эй"),
    ("ey", "эй"), ("ue", "ю"),
]

_SINGLE = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г",
    "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н",
    "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у",
    "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
}


def _translit_word(word: str) -> str:
    """Rough English -> Russian reading for words missing from LATIN_WORDS."""
    w = word.lower()
    out: list[str] = []
    i = 0
    while i < len(w):
        rest = w[i:]
        for src, dst in _DIGRAPHS:
            if rest.startswith(src):
                out.append(dst)
                i += len(src)
                break
        else:
            ch = w[i]
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if ch == "c":
                out.append("с" if nxt in "eiy" else "к")
            elif ch == "g":
                out.append("дж" if nxt in "eiy" else "г")
            elif ch == "e" and i == len(w) - 1 and len(w) > 2 and w[i - 1] not in "aeiouy":
                pass  # silent final -e: "code", "cache"
            else:
                out.append(_SINGLE.get(ch, ch))
            i += 1
    return "".join(out)


def _translate_part(part: str) -> str:
    from num2words import num2words

    if not part:
        return ""
    if part.isdigit():
        return num2words(int(part), lang="ru")
    key = part.lower()
    if key in LATIN_WORDS:
        return LATIN_WORDS[key]
    if len(part) <= 5 and part.isupper():
        return " ".join(LETTER_NAMES.get(ch, ch) for ch in key)
    return _translit_word(key)


def transliterate_latin(text: str) -> str:
    """Read Latin words (and file names like f5_server.py) in Russian letters."""
    import re

    def repl(match: re.Match) -> str:
        token = match.group(0)
        key = token.lower().replace("_", "").replace("-", "").replace(".", "")
        if key in LATIN_WORDS and not any(c in token for c in "._-"):
            return LATIN_WORDS[key]
        parts = re.split(r"[._\-]+|(?<=[a-z0-9])(?=[A-Z])", token)
        translated = [_translate_part(p) for p in parts if p]
        return " ".join(p for p in translated if p)

    return re.sub(r"[A-Za-z][A-Za-z0-9._\-]*[A-Za-z0-9]|[A-Za-z]", repl, text)


def load_model() -> None:
    global _tts, _ref_text, _pronounce
    import soundfile  # noqa: F401
    from f5_tts.api import F5TTS
    from ruaccent import RUAccent

    accent = RUAccent()
    accent.load(omograph_model_size="turbo3", use_dictionary=True)
    _ref_text = accent.process_all(REF_TEXT)
    _pronounce = load_pronounce()

    device = resolve_device()
    print(f"[f5] loading model on {device}", file=sys.stderr, flush=True)
    _tts = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=str(CKPT),
        vocab_file=str(VOCAB),
        device=device,
    )
    _tts._accent = accent  # keep for requests
    print("[f5] ready", file=sys.stderr, flush=True)


def synth(text: str, speed: float, nfe: int) -> bytes:
    import io
    import re

    import numpy as np
    import soundfile as sf

    # Normalise punctuation F5 has no idea about.
    text = text.replace("\r", "")
    text = normalize_numbers(text)
    text = transliterate_latin(text)
    text = re.sub(r"\.{2,}", "…", text)
    text = text.replace("(", " — ").replace(")", " — ")
    text = text.replace("[", " — ").replace("]", " — ")
    text = re.sub(r"[ \t]+", " ", text)

    # Split into segments: paragraphs -> sentences -> clauses.
    segments: list[tuple[str, float]] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in re.findall(r"[^.!?…]+[.!?…]*", paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            for clause in re.split(r"(?<=[;:—–])\s+", sentence):
                clause = clause.strip()
                if not clause:
                    continue
                pause = pause_after(clause)
                # F5 truncates the last syllable when a clause ends with ; : — –,
                # so replace those with a full stop and keep the pause separately.
                clean = clause.lstrip("—–").strip()
                if clean.endswith((";", ":", "—", "–", ",")):
                    clean = clean.rstrip(";:—–,").strip()
                    if clean and not clean.endswith((".", "!", "?", "…")):
                        clean += "."
                if clean:
                    segments.append((clean, pause))
        # Paragraph break: slightly longer than a sentence.
        if segments:
            last_text, last_pause = segments[-1]
            segments[-1] = (last_text, max(last_pause, 0.45))

    if not segments:
        segments = [(text, 0.0)]

    # Group segments into blocks of ~200 characters: one F5 call per block keeps
    # per-call overhead low, while pauses are still inserted between blocks.
    blocks: list[tuple[str, float]] = []
    buffer: list[str] = []
    size = 0
    last_pause = 0.38
    for part, pause in segments:
        if buffer and size + len(part) > 200:
            blocks.append((" ".join(buffer), last_pause))
            buffer = []
            size = 0
        buffer.append(part)
        size += len(part) + 1
        last_pause = pause
    if buffer:
        blocks.append((" ".join(buffer), 0.0))

    sr_out = 24000
    pieces = []
    for index, (part, pause) in enumerate(blocks):
        accented = _tts._accent.process_all(part)
        accented = apply_pronounce(accented)
        wav, sr, _ = _tts.infer(
            ref_file=str(REF),
            ref_text=_ref_text,
            gen_text=accented,
            seed=1234,
            speed=speed,
            nfe_step=nfe,
        )
        sr_out = sr
        audio = np.asarray(wav, dtype=np.float32)
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 0:
            loud = np.where(np.abs(audio) > peak * 0.02)[0]
            if loud.size:
                margin = int(sr * 0.03)
                audio = audio[max(0, loud[0] - margin) : min(len(audio), loud[-1] + margin)]
        pieces.append(audio)
        if index != len(blocks) - 1:
            pieces.append(np.zeros(int(sr_out * max(0.0, pause)), dtype=np.float32))

    merged = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    buf = io.BytesIO()
    sf.write(buf, merged, sr_out, format="WAV", subtype="PCM_16")
    return buf.getvalue()


_morph = None


def normalize_numbers(text: str) -> str:
    """Expand digits and abbreviations into Russian words before accenting."""
    import re

    from num2words import num2words
    import pymorphy3

    global _morph
    if _morph is None:
        _morph = pymorphy3.MorphAnalyzer()
    morph = _morph

    def card(n: int) -> str:
        return num2words(n, lang="ru")

    def plural(n: int, forms: tuple[str, str, str]) -> str:
        if n % 10 == 1 and n % 100 != 11:
            return forms[0]
        if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            return forms[1]
        return forms[2]

    def inflect_phrase(phrase: str, grammemes: set[str]) -> str:
        result = []
        for word in phrase.split():
            parsed = morph.parse(word)[0]
            form = parsed.inflect(grammemes)
            result.append(form.word if form else word)
        return " ".join(result)

    def ordinal_in_case(year: int, follow: str) -> str:
        words = num2words(year, lang="ru", to="ordinal").split()
        follow = follow.lower()
        if follow.startswith("году") or follow.startswith("годах"):
            grammemes = {"loct"}
        elif follow.startswith("года"):
            grammemes = {"gent"}
        elif follow.startswith("годом"):
            grammemes = {"ablt"}
        else:
            grammemes = {"nomn"}
        parsed_list = morph.parse(words[-1])
        parsed = next(
            (p for p in parsed_list if "masc" in p.tag and "sing" in p.tag),
            parsed_list[0],
        )
        form = parsed.inflect(grammemes)
        words[-1] = form.word if form else words[-1]
        return " ".join(words)

    units = {
        "мс": ("миллисекунда", "миллисекунды", "миллисекунд"),
        "сек": ("секунда", "секунды", "секунд"),
        "мин": ("минута", "минуты", "минут"),
        "ч": ("час", "часа", "часов"),
        "тб": ("терабайт", "терабайта", "терабайт"),
        "гб": ("гигабайт", "гигабайта", "гигабайт"),
        "мб": ("мегабайт", "мегабайта", "мегабайт"),
        "кб": ("килобайт", "килобайта", "килобайт"),
        "ггц": ("гигагерц", "гигагерца", "гигагерц"),
        "кгц": ("килогерц", "килогерца", "килогерц"),
        "мгц": ("мегагерц", "мегагерца", "мегагерц"),
        "гц": ("герц", "герца", "герц"),
        "квт": ("киловатт", "киловатта", "киловатт"),
        "вт": ("ватт", "ватта", "ватт"),
        "км": ("километр", "километра", "километров"),
        "см": ("сантиметр", "сантиметра", "сантиметров"),
        "мм": ("миллиметр", "миллиметра", "миллиметров"),
        "кг": ("килограмм", "килограмма", "килограммов"),
        "мг": ("миллиграмм", "миллиграмма", "миллиграммов"),
        "мл": ("миллилитр", "миллилитра", "миллилитров"),
        "руб": ("рубль", "рубля", "рублей"),
        "г": ("грамм", "грамма", "граммов"),
        "л": ("литр", "литра", "литров"),
        "м": ("метр", "метра", "метров"),
        "с": ("секунда", "секунды", "секунд"),
    }
    unit_keys = sorted(units, key=len, reverse=True)
    unit_alt = "|".join(re.escape(k) for k in unit_keys)

    # Years: "в 2026 году" -> "в две тысячи двадцать шестом году"
    text = re.sub(
        r"\b(1[89]\d{2}|20\d{2})\b(\s+)(год[а-я]*|г\.)",
        lambda m: ordinal_in_case(int(m.group(1)), m.group(3)) + m.group(2) + m.group(3),
        text,
        flags=re.IGNORECASE,
    )

    # Thousands written with a space: "1 500" -> "1500".
    text = re.sub(
        r"\b\d{1,3}(?:[ \u00a0\u202f]\d{3})+\b",
        lambda m: re.sub(r"\D", "", m.group(0)),
        text,
    )

    # Ranges: "10-15 мин" -> "с десяти до пятнадцати минут".
    text = re.sub(
        r"\b(\d+)\s*[-–—]\s*(\d+)\s*(" + unit_alt + r")?\.?(?![А-Яа-яЁёA-Za-z])",
        lambda m: (
            f"с {inflect_phrase(card(int(m.group(1))), {'gent'})}"
            f" до {inflect_phrase(card(int(m.group(2))), {'gent'})}"
            + (f" {units[m.group(3).lower()][2]}" if m.group(3) else "")
        ),
        text,
        flags=re.IGNORECASE,
    )

    unit_re = re.compile(
        r"(\d+(?:[.,]\d+)?)\s*(" + unit_alt + r")(\.)?(?![А-Яа-яЁёA-Za-z])",
        re.IGNORECASE,
    )

    def unit_sub(match: re.Match) -> str:
        raw = match.group(1).replace(",", ".")
        forms = units[match.group(2).lower()]
        if "." in raw:
            phrase = f"{num2words(float(raw), lang='ru')} {forms[1]}"
        else:
            n = int(raw)
            phrase = f"{card(n)} {plural(n, forms)}"
        dot = match.group(3) or ""
        if dot:
            # Keep the full stop only when it really ends the sentence,
            # not when it was an abbreviation dot in the middle ("10 мин. на дорогу").
            following = match.string[match.end() :].lstrip()[:1]
            if following and following.islower():
                dot = ""
        return phrase + dot

    text = unit_re.sub(unit_sub, text)

    # Fixed phrases with units.
    text = re.sub(
        r"\b(\d+)\s*(?:fps|фпс)\b",
        lambda m: f"{card(int(m.group(1)))} кадров в секунду",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d+)\s*dpi\b",
        lambda m: f"{card(int(m.group(1)))} точек на дюйм",
        text,
        flags=re.IGNORECASE,
    )

    # Currency: "5 $" -> "пять долларов"
    currency = {"$": ("доллар", "доллара", "долларов"), "€": ("евро", "евро", "евро"), "₽": ("рубль", "рубля", "рублей")}
    text = re.sub(
        r"(\d+(?:[.,]\d+)?)\s*([$€₽])",
        lambda m: (
            f"{card(int(m.group(1)))} {plural(int(m.group(1)), currency[m.group(2)])}"
            if "," not in m.group(1) and "." not in m.group(1)
            else f"{num2words(float(m.group(1).replace(',', '.')), lang='ru')} {currency[m.group(2)][1]}"
        ),
        text,
    )

    # Time: "12:30" -> "двенадцать тридцать"
    text = re.sub(
        r"\b(\d{1,2}):(\d{2})\b",
        lambda m: f"{card(int(m.group(1)))} {card(int(m.group(2)))}",
        text,
    )

    # Percentages with agreement.
    def percent(match: re.Match) -> str:
        raw = match.group(1).replace(",", ".")
        if "." in raw:
            return f"{num2words(float(raw), lang='ru')} процентов"
        n = int(raw)
        if n % 10 == 1 and n % 100 != 11:
            word = "процент"
        elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            word = "процента"
        else:
            word = "процентов"
        return f"{card(n)} {word}"

    text = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", percent, text)

    # Multi-part versions: "1.2.3" -> "один точка два точка три"
    text = re.sub(
        r"\b\d+(?:\.\d+){2,}\b",
        lambda m: " точка ".join(card(int(x)) for x in m.group(0).split(".")),
        text,
    )

    # Decimals: "3,14" -> "три запятая четырнадцать", "1.0" -> "один точка ноль"
    text = re.sub(
        r"\b(\d+)([.,])(\d+)\b",
        lambda m: f"{card(int(m.group(1)))} {'запятая' if m.group(2) == ',' else 'точка'} {card(int(m.group(3)))}",
        text,
    )

    # Remaining integers.
    text = re.sub(r"\b\d+\b", lambda m: card(int(m.group(0))), text)

    # Standalone abbreviations.
    abbreviations = {
        "т.д.": "так далее",
        "т.п.": "тому подобное",
        "т.е.": "то есть",
        "т.к.": "так как",
        "напр.": "например",
        "рис.": "рисунок",
        "стр.": "страница",
        "табл.": "таблица",
        "др.": "другое",
        "пр.": "прочее",
        "см.": "смотри",
    }
    for abbr, full in abbreviations.items():
        text = re.sub(re.escape(abbr), full, text, flags=re.IGNORECASE)
    text = text.replace("№", "номер")
    return text


def pause_after(clause: str) -> float:
    if clause.endswith("…"):
        return 0.55
    if clause.endswith(("!", "?", ".")):
        return 0.38
    if clause.endswith((";", ":")):
        return 0.26
    if clause.endswith(("—", "–")):
        return 0.22
    return 0.3


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send(200, b"ok", "text/plain")
        elif self.path == "/reload":
            global _pronounce
            _pronounce = load_pronounce()
            self._send(200, b"reloaded", "text/plain")
        elif self.path == "/shutdown":
            self._send(200, b"bye", "text/plain")
            import threading
            import time as _time

            def _exit() -> None:
                _time.sleep(0.2)
                os._exit(0)

            threading.Thread(target=_exit, daemon=True).start()
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/tts":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = str(payload.get("text", "")).strip()
            if not text:
                self._send(400, b"empty text", "text/plain")
                return
            speed = float(payload.get("speed", SPEED))
            nfe = int(payload.get("nfe", NFE))
            audio = synth(text, speed, nfe)
            self._send(200, audio, "audio/wav")
        except Exception as error:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            self._send(500, f"{type(error).__name__}: {error}".encode("utf-8"), "text/plain")


def main() -> None:
    load_model()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[f5] listening on {PORT}", file=sys.stderr, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
