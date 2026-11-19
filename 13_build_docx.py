"""Сборка article.docx из article.md — канонический путь фиксации правды.

Источник правды текста статьи: article_template.md + JSON-результаты
-> 10_build_article.py -> article.md. Word-документ — производный файл:
его правки руками не сохраняются при следующем пересчёте, править нужно
шаблон и пересобирать. Запуск: python 13_build_docx.py [article.md] [article.docx]

Зависимость python-docx нужна только здесь, не для анализа:
    python -m pip install -r requirements-docx.txt
"""

import re
import shutil
import sys
import zipfile
from pathlib import Path
import hashlib

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
except ImportError:
    raise SystemExit(
        "Нужен python-docx: python -m pip install -r requirements-docx.txt "
        "(для анализа не требуется, только для сборки docx)"
    )

# Шаблон python-docx несёт имена восточноазиатских шрифтов в теме, таблице
# шрифтов и языковых настройках. Тексту статьи они не нужны, поэтому после
# сохранения они убираются; видимый текст не меняется.
EAST_ASIAN = re.compile(
    "[\u2e80-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef\u3000-\u303f]"
)


def _clean_part(name, xml):
    if name == "word/theme/theme1.xml":
        return re.sub(
            r'<a:font script="[^"]*" typeface="[^"]*"/>',
            lambda m: "" if EAST_ASIAN.search(m.group()) else m.group(),
            xml,
        )
    if name == "word/fontTable.xml":
        return re.sub(
            r'<w:font w:name="[^"]*">.*?</w:font>',
            lambda m: "" if EAST_ASIAN.search(m.group()) else m.group(),
            xml,
            flags=re.S,
        )
    if name == "word/settings.xml":
        return xml.replace('w:eastAsia="ja-JP"', 'w:eastAsia="ru-RU"')
    return xml


def drop_east_asian_fonts(path):
    """Переписывает docx без восточноазиатских шрифтов шаблона."""
    tmp = Path(path).with_suffix(".tmp.docx")
    with (
        zipfile.ZipFile(path) as zin,
        zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout,
    ):
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".xml"):
                data = _clean_part(item.filename, data.decode("utf-8")).encode("utf-8")
            zout.writestr(item, data)
    shutil.move(tmp, path)


ROOT = Path(__file__).resolve().parent
BODY_FONT = "DejaVu Sans"
CODE_FONT = "DejaVu Sans Mono"
ACCENT = RGBColor(0x4F, 0x81, 0xBD)
IMG_WIDTH = Cm(16.3)


def setup_styles(doc):
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = Pt(10.5)
    st.paragraph_format.space_after = Pt(5)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    for name, size in [("Heading 1", 16), ("Heading 2", 12.5), ("Heading 3", 11)]:
        s = doc.styles[name]
        s.font.name = BODY_FONT
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    t = doc.styles["Title"]
    t.font.name = BODY_FONT
    t.font.size = Pt(25)
    t.font.color.rgb = RGBColor(0, 0, 0)
    t.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    cap = doc.styles["Caption"]
    cap.font.name = BODY_FONT
    cap.font.size = Pt(9)
    cap.font.bold = True
    cap.font.color.rgb = ACCENT
    cap.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)


SPOILER_FILL = "EEF2F7"
SPOILER_EDGE = "4F81BD"


def mark_spoiler(par, edge=True):
    """Серая заливка и синяя черта слева: в docx спойлер раскрыт, но виден."""
    ppr = par._p.get_or_add_pPr()
    shd = ppr.makeelement(
        qn("w:shd"),
        {qn("w:val"): "clear", qn("w:color"): "auto", qn("w:fill"): SPOILER_FILL},
    )
    if edge:
        bdr = ppr.makeelement(qn("w:pBdr"), {})
        left = bdr.makeelement(
            qn("w:left"),
            {
                qn("w:val"): "single",
                qn("w:sz"): "24",
                qn("w:space"): "6",
                qn("w:color"): SPOILER_EDGE,
            },
        )
        bdr.append(left)
        ppr.append(bdr)
    ppr.append(shd)
    return par


def set_table_borders(tbl):
    tblPr = tbl._tbl.tblPr
    borders = tblPr.makeelement(qn("w:tblBorders"), {})
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = borders.makeelement(qn("w:" + edge), {})
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "4")
        el.set(qn("w:color"), "BFBFBF")
        borders.append(el)
    tblPr.append(borders)


def set_col_widths(tbl):
    """Ширины колонок по объёму текста: длинные колонки шире, сумма = 16,9 см."""
    n_cols = len(tbl.columns)
    weights = []
    for j in range(n_cols):
        texts = [row.cells[j].text for row in tbl.rows]
        weights.append(max(3, min(60, max(len(t) for t in texts))) ** 0.75)
    total = sum(weights)
    usable = Cm(16.9)
    for j, w in enumerate(weights):
        width = int(usable * w / total)
        for row in tbl.rows:
            row.cells[j].width = width


INLINE = re.compile(r"(\*\*.+?\*\*|\*[^*]+?\*|`[^`]+?`)")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


def anchor_name(url):
    return "h_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:24]


def add_hyperlink(par, text, url):
    """Кликабельная гиперссылка вместо сырой markdown-записи [текст](url)."""
    if url.startswith("#"):
        hl = par._p.makeelement(qn("w:hyperlink"), {qn("w:anchor"): anchor_name(url)})
    else:
        r_id = par.part.relate_to(
            url,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
            is_external=True,
        )
        hl = par._p.makeelement(qn("w:hyperlink"), {qn("r:id"): r_id})
    r = par._p.makeelement(qn("w:r"), {})
    rPr = par._p.makeelement(qn("w:rPr"), {})
    color = par._p.makeelement(qn("w:color"), {qn("w:val"): "4F81BD"})
    underline = par._p.makeelement(qn("w:u"), {qn("w:val"): "single"})
    rPr.append(color)
    rPr.append(underline)
    r.append(rPr)
    t = par._p.makeelement(qn("w:t"), {})
    t.text = text
    r.append(t)
    hl.append(r)
    par._p.append(hl)


def _add_styled_runs(par, text, base_size=None):
    """Курсив/полужирный/код внутри строки markdown."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            r = par.add_run(piece[2:-2])
            r.bold = True
        elif piece.startswith("`") and piece.endswith("`"):
            r = par.add_run(piece[1:-1])
            r.font.name = CODE_FONT
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            r.italic = True
        else:
            r = par.add_run(piece)
        if base_size and r.font.name != CODE_FONT:
            r.font.size = base_size
            r.font.name = BODY_FONT


def add_runs(par, text, base_size=None):
    """Стилизованный текст с гиперссылками: [текст](url) внутри абзаца."""
    pos = 0
    for m in LINK.finditer(text):
        if m.start() > pos:
            _add_styled_runs(par, text[pos : m.start()], base_size)
        add_hyperlink(par, m.group(1), m.group(2))
        pos = m.end()
    if pos < len(text):
        _add_styled_runs(par, text[pos:], base_size)
    return par


def fix_table(tbl):
    """Повторение строки заголовка и запрет разрыва строки при переносе
    страницы (таблицы рвались внутри строки и без заголовка
    на новой странице)."""
    for i, row in enumerate(tbl.rows):
        trPr = row._tr.get_or_add_trPr()
        if i == 0:
            th = trPr.makeelement(qn("w:tblHeader"), {qn("w:val"): "true"})
            trPr.append(th)
        cs = trPr.makeelement(qn("w:cantSplit"), {})
        trPr.append(cs)


def add_image(doc, alt, path, base):
    full = base / path
    if not full.exists():
        raise SystemExit(f"нет рисунка: {full}")
    # Заглавная гравюра портретная: при полной ширине 16,3 см её высота
    # превысила бы страницу, поэтому она уже и по центру.
    width = IMG_WIDTH
    doc.add_picture(str(full), width=width)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    # Подписи к рисункам идут в article.md отдельной курсивной строкой
    # («*Рисунок N. …*»): на Хабре alt-текст под картинкой не показывается.
    # Поэтому alt здесь в подпись не превращается.


def add_table(doc, rows):
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    body = []
    for line in rows[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        cells += [""] * (len(header) - len(cells))
        body.append(cells[: len(header)])
    tbl = doc.add_table(rows=1 + len(body), cols=len(header))
    tbl.style = "Table Grid"
    set_table_borders(tbl)
    size = Pt(9)
    for j, text in enumerate(header):
        cell = tbl.rows[0].cells[j]
        cell.text = ""
        p = cell.paragraphs[0]
        add_runs(p, text, base_size=size)
        for r in p.runs:
            r.bold = True
    for i, row in enumerate(body, start=1):
        for j, text in enumerate(row):
            cell = tbl.rows[i].cells[j]
            cell.text = ""
            add_runs(cell.paragraphs[0], text, base_size=size)
    set_col_widths(tbl)
    fix_table(tbl)
    # Последняя строка остаётся рядом с подписью таблицы.
    for cell in tbl.rows[-1].cells:
        for par in cell.paragraphs:
            par.paragraph_format.keep_with_next = True


def add_heading(doc, text, style, index):
    par = doc.add_paragraph(text, style=style)
    slug = "#" + re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
    bookmark = par._p.makeelement(qn("w:bookmarkStart"), {qn("w:id"): str(index), qn("w:name"): anchor_name(slug)})
    par._p.insert(1, bookmark)
    par._p.append(par._p.makeelement(qn("w:bookmarkEnd"), {qn("w:id"): str(index)}))
    return par


def md_to_docx(md_path, out_path):
    md_path = Path(md_path)
    base = md_path.parent
    lines = Path(md_path).read_text().split("\n")
    doc = Document()
    setup_styles(doc)
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin = sec.bottom_margin = Cm(1.79)
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer_label = "Сокращённая версия исследования" if md_path.parent.name == "short_article" else "Полная версия исследования"
    footer.add_run(footer_label + " • ").font.size = Pt(9)
    field = footer._p.makeelement(qn("w:fldSimple"), {qn("w:instr"): "PAGE"})
    footer._p.append(field)
    i = 0
    in_spoiler = False
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and lines[i + 1].strip().startswith("|")
        ):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        if stripped.startswith("# "):
            doc.add_paragraph(stripped[2:], style="Title")
        elif stripped.startswith("<spoiler"):
            # Спойлер Хабра: в docx блок раскрыт целиком, но выделен заливкой и
            # чертой слева, а начало и конец помечены — при переносе в редактор
            # Хабра его нужно снова свернуть в спойлер.
            m = re.match(r'<spoiler(?:\s+title="(.*?)")?>', stripped)
            title = m.group(1) if m and m.group(1) else "Развёрнутый блок"
            p = doc.add_paragraph()
            r = p.add_run("▼ СПОЙЛЕР: " + title)
            r.bold = True
            r.font.color.rgb = ACCENT
            p.paragraph_format.keep_with_next = True  # заголовок не отрывается от блока
            mark_spoiler(p)
            in_spoiler = True
        elif stripped.startswith("</spoiler>"):
            p = doc.add_paragraph()
            r = p.add_run("▲ конец спойлера")
            r.italic = True
            r.font.size = Pt(9)
            r.font.color.rgb = ACCENT
            mark_spoiler(p)
            in_spoiler = False
        elif re.fullmatch(r"<anchor>[a-zA-Z0-9_-]+</anchor>", stripped):
            # В Habr Flavored Markdown якорь не является видимым текстом.
            pass
        elif stripped.startswith("#### "):
            add_heading(doc, stripped[5:], "Heading 3", i)
        elif stripped.startswith("### "):
            add_heading(doc, stripped[4:], "Heading 2", i)
        elif stripped.startswith("## "):
            add_heading(doc, stripped[3:], "Heading 1", i)
        elif stripped.startswith("<img "):
            src = re.search(r'src="([^"]+)"', stripped)
            alt = re.search(r'alt="([^"]*)"', stripped)
            if src:
                add_image(doc, alt.group(1) if alt else "", src.group(1), base)
        elif stripped.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
            if m:
                add_image(doc, m.group(1), m.group(2), base)
        elif re.match(r"^\*(Рисунок|Таблица) \d+\.", stripped):
            # Подписи и рисунков, и таблиц идут под объектом; рисунок — по центру.
            p = doc.add_paragraph(style="Caption")
            add_runs(p, stripped)
            if stripped.startswith("*Рисунок"):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            else:
                # Отступ от таблицы, чтобы подпись не прилипала к нижней рамке.
                p.paragraph_format.space_before = Pt(8)
            if in_spoiler:
                mark_spoiler(p)
        elif stripped:
            p = doc.add_paragraph()
            add_runs(p, stripped)
            if in_spoiler:
                mark_spoiler(p)
        i += 1
    doc.save(out_path)
    drop_east_asian_fonts(out_path)
    print(
        f"{out_path}: {len(doc.paragraphs)} абзацев, {len(doc.tables)} таблиц, {len(doc.inline_shapes)} рисунков"
    )


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else ROOT / "article.md"
    out = sys.argv[2] if len(sys.argv) > 2 else ROOT / "article.docx"
    md_to_docx(md, out)
