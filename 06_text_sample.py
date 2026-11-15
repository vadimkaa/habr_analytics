"""Шаг 06. Текстовые метрики из зафиксированного кэша HTML-страниц.

По умолчанию сеть не используется и новая выборка не генерируется.
Исходная выборка: 1200+400 записей из квартилей VPD всех возрастов; 1595 ID.
Манифест фиксирует реальные записи кэша, а не обещает побайтовую повторяемость
запросов к изменяющемуся сайту. Кэш относится к ноябрю 2026, не к дате публикации.
"""

import gzip
import json
import re

import numpy as np
from bs4 import BeautifulSoup
from razdel import sentenize
from habrpub.analysis import DATA, save

WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z]+(?:[-’\'][А-Яа-яЁёA-Za-z]+)*")
RU_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
EN_RE = re.compile(r"[A-Za-z]+(?:[-’\'][A-Za-z]+)*")
VOWELS_RU = re.compile("[аеёиоуыэюя]")
VOWELS_ALL = re.compile("[аеёиоуыэюяaeiouy]")


def prepare_prose(raw):
    """DOM-очистка. Удалённые доли считаются последовательно без двойного учёта."""
    soup = BeautifulSoup(raw, "html.parser")
    total = max(len(soup.get_text(" ", strip=True)), 1)
    counters = {}
    for selector, name in [
        ("script,style", "service"),
        ("pre", "code_char"),
        ("table", "table"),
        (
            "math,.katex,.MathJax,.mathjax,tex,span.tex,div.tex,img.formula,img[data-tex]",
            "formula",
        ),
        ("code", "inline_code"),
    ]:
        nodes = soup.select(selector)
        chars = 0
        n = 0
        for node in nodes:
            if node.parent is None:
                continue
            chars += len(node.get_text(" ", strip=True))
            n += 1
            node.decompose()
        counters[name + "_share"] = chars / total
        counters[name + "_n"] = n
    # Границы блочных элементов сохраняем; inline-теги не создают предложения.
    for node in soup.find_all(
        ["p", "div", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6", "br"]
    ):
        node.insert_before("\n")
        node.insert_after("\n")
    prose = soup.get_text(" ", strip=False)
    prose = "\n".join(
        re.sub(r"[^\S\n]+", " ", p).strip() for p in prose.splitlines() if p.strip()
    )
    return prose, counters


def mattr(seq, window=500):
    """MATTR: средний TTR по скользящим окнам из `window` слов.

    Обычный TTR падает с ростом текста и в основном измеряет длину; MATTR устойчив к длине. Для текста короче окна MATTR равен
    обычному TTR всего текста."""
    if not seq:
        return None
    if len(seq) <= window:
        return len(set(seq)) / len(seq)
    from collections import Counter

    counts = Counter(seq[:window])
    ttrs = [len(counts) / window]
    for i in range(window, len(seq)):
        old = seq[i - window]
        counts[old] -= 1
        if counts[old] == 0:
            del counts[old]
        counts[seq[i]] += 1
        ttrs.append(len(counts) / window)
    return float(np.mean(ttrs))


def compute_metrics(prose, meta=None):
    words = WORD_RE.findall(prose)
    ru = [w for w in words if RU_RE.fullmatch(w)]
    en = [w for w in words if EN_RE.fullmatch(w)]
    sentences = [
        s.text
        for p in prose.splitlines()
        for s in sentenize(p)
        if WORD_RE.search(s.text)
    ]
    result = {
        **(meta or {}),
        "words": len(words),
        "ru_words": len(ru),
        "en_words": len(en),
        "sentences": len(sentences),
    }
    if len(words) < 30 or not sentences:
        return result
    asl = len(words) / len(sentences)
    asw = sum(len(VOWELS_ALL.findall(w.lower())) for w in words) / len(words)
    asl_ru = len(ru) / len(sentences) if ru else None
    asw_ru = (
        sum(len(VOWELS_RU.findall(w.lower())) for w in ru) / len(ru) if ru else None
    )
    result.update(
        asl=asl,
        asw_all=asw,
        asl_ru=asl_ru,
        asw_ru=asw_ru,
        ttr_ru4=len(set(w.lower() for w in ru if len(w) >= 4))
        / max(sum(len(w) >= 4 for w in ru), 1),
        mattr_ru4=mattr([w.lower() for w in ru if len(w) >= 4]),
        long_word_share=sum(len(w) >= 10 for w in ru) / len(ru) if ru else None,
        en_word_share=len(en) / len(words),
    )
    # ИУФО — имя поля сохранено для сопоставления с исходными метриками.
    for suffix, a, b in [("", asl, asw), ("_ru", asl_ru, asw_ru)]:
        if a is None:
            continue
        result["flesch_en" + suffix] = 206.835 - 1.015 * a - 84.6 * b
        result["flesch_web" + suffix] = 206.835 - 1.3 * a - 60.1 * b
        result["oborneva" + suffix] = 206.835 - 1.52 * a - 65.14 * b
        result["fk_mod" + suffix] = 0.36 * a + 5.76 * b - 11.97
    return result


def main():
    path = DATA / "lex_texts_raw.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(
            "Нужен исходный lex_texts_raw.jsonl.gz; этот шаг не скачивает тексты."
        )
    rows = [json.loads(line) for line in gzip.open(path, "rt", encoding="utf-8")]
    manifest = [{"id": str(r["id"]), "corpus": r["corpus"]} for r in rows]
    (DATA / "text_sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    out = []
    for r in rows:
        p, c = prepare_prose(r["textHtml"])
        meta = {
            k: r.get(k)
            for k in ["id", "corpus", "readingTime", "complexity", "isCorporative"]
        }
        meta["id"] = str(meta["id"])
        out.append(compute_metrics(p, {**meta, **c}))
    with open(DATA / "lex_metrics.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    save(
        {
            "n_cache_records": len(rows),
            "n_unique_ids": len({r["id"] for r in out}),
            "n_valid_records": sum("oborneva" in r for r in out),
            "extractor": "BeautifulSoup html.parser; razdel; alphabetic words including one-letter words",
            "limitations": "syllables approximated by vowels; formulas without formula class or data-tex attribute are missed; current rather than publication text",
        },
        "06_text_results.json",
    )
    print("06", len(rows), "records; unique", len({r["id"] for r in out}))


if __name__ == "__main__":
    main()
