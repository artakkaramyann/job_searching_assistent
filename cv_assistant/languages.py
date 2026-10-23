"""Языки и явно названные уровни. «Максимальный» не превращается в C2."""

import re

NAMES = {
    "Армянский": r"армянск\w*|armenian",
    "Русский": r"русск\w*|russian",
    "Английский": r"английск\w*|english",
    "Немецкий": r"немецк\w*|german",
    "Французский": r"французс?к\w*|french",
    "Испанский": r"испанск\w*|spanish",
    "Китайский": r"китайск\w*|chinese",
    "Грузинский": r"грузинск\w*|georgian",
    "Итальянский": r"итальянск\w*|italian",
    "Японский": r"японск\w*|japanese",
    "Арабский": r"арабск\w*|arabic",
    "Португальский": r"португальск\w*|portuguese",
}
LEVEL = re.compile(r"как\s+носитель|на\s+уровне\s+носител\w*|\b[abcавс][12]\b|родно\w*|native|свободн\w*|fluent|максимальн\w*|базов\w*|начальн\w*|средн\w*|хорошо", re.I)
NEGATION = re.compile(r"\b(?:не|нет|not|don't|cannot)\b", re.I)
CLAUSE = re.compile(r"[,;.!?:]|\b(?:но|but)\b", re.I)


def positive_level_matches(text: str) -> list[re.Match]:
    matches = []
    for match in LEVEL.finditer(text):
        # «Не родной, уровень B1»: отрицание относится только к первой части.
        prefix = CLAUSE.split(text[:match.start()])[-1]
        if not NEGATION.search(prefix):
            matches.append(match)
    return matches


def level_text(text: str) -> str | None:
    matches = positive_level_matches(text)
    if not matches:
        return None
    value = matches[0].group().lower()
    if "носител" in value:
        return "как носитель (самооценка)"
    if re.fullmatch(r"[abcавс][12]", value):
        return value.upper().translate(str.maketrans("АВС", "ABC"))
    for prefix, label in [("родн", "родной"), ("native", "родной"), ("свобод", "свободно"),
                          ("fluent", "свободно"), ("максим", "максимальный (самооценка)"),
                          ("базов", "базовый"), ("начальн", "начальный"), ("средн", "средний")]:
        if value.startswith(prefix):
            return label
    return value


def extract_languages(turns: list[dict], proposed: list[dict] = ()) -> list[dict]:
    result = {}
    pending_names, pending_level = [], None
    for turn in turns:
        answer = turn["answer"]
        matches = sorted((m.start(), m.end(), name) for name, pattern in NAMES.items()
                         for m in re.finditer(r"\b(?:" + pattern + r")\b", answer, re.I))
        # Только явное «все/оба ...» распространяет общий уровень на весь список.
        shared = re.search(r"\b(?:все|всё|всех|всеми|оба|обе|all)\b[^.!?]*$", answer, re.I)
        shared_level = (level_text(shared.group()) if shared and not NEGATION.search(answer)
                        and len(positive_level_matches(answer)) == 1 else None)
        # Короткое подтверждение относится только к явно заданному уточнению.
        if not matches and pending_names and turn["topic"] == "languages" and turn.get("key") == "language_levels":
            if (re.match(r"^(?:да\b|верно\b|все\b|на всех\b)", answer.strip(), re.I)
                    and not NEGATION.search(answer) and not re.search(r"\b(?:но|кроме|только)\b", answer, re.I)):
                for name in pending_names:
                    result[name] = {"name": name, "level": level_text(answer) or pending_level,
                                    "evidence": result[name]["evidence"] + "\n" + answer}
                pending_names, pending_level = [], None
        ambiguous = (len(matches) > 1 and not shared_level and not NEGATION.search(answer)
                     and len(positive_level_matches(answer)) == 1
                     and re.search(r"знаю|говорю|разговариваю|владею", answer, re.I))
        if ambiguous:
            pending_names = [name for _, _, name in matches]
            pending_level = level_text(answer)
        for i, (start, end, name) in enumerate(matches):
            stop = matches[i+1][0] if i+1 < len(matches) else len(answer)
            suffix = answer[end:stop]
            before = answer[matches[i-1][1] if i else 0:start]
            prefix = CLAUSE.split(before)[-1] if i == 0 or CLAUSE.search(before) else ""
            # «Русский, родной армянский»: слово перед следующим языком не относится к предыдущему.
            if i+1 < len(matches):
                suffix = CLAUSE.split(suffix)[0]
            segment = prefix + " " + answer[start:end] + suffix
            level = None if ambiguous else (shared_level or level_text(suffix) or level_text(prefix))
            if len(matches) == 1 and level is None:
                level = level_text(answer)
            corrected = bool(NEGATION.search(segment))
            if name not in result or level is not None or ambiguous or corrected:
                result[name] = {"name": name, "level": level, "evidence": answer}
                if ambiguous:
                    result[name]["suggested_level"] = pending_level
    # Редкие языки из LLM: имя и уровень должны буквально присутствовать в цитате.
    # Для распространённых языков привязку уровней выше проверяем независимо от LLM.
    for item in proposed:
        name, evidence, level = item["name"], item["evidence"], item["level"]
        # Модель может вернуть «армянский» или Armenian вместо «Армянский».
        if any(re.fullmatch(pattern, name.strip(), re.I) for pattern in NAMES.values()):
            continue
        if any(existing.casefold() == name.casefold() for existing in result):
            continue
        if not any(evidence and evidence.casefold() in t["answer"].casefold() for t in turns):
            continue
        if name.casefold() not in evidence.casefold():
            continue
        if level and level.casefold() not in evidence.casefold():
            level = None
        result[name] = {"name": name, "level": level, "evidence": evidence}
    return list(result.values())
