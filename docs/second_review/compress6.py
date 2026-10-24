"""Compress the capstone paper from 8 pages to 6 without deleting figures.

Strategy, in the order the user asked for it:
  1. crop the dead white margin out of every plot, then lay the figures out
     two-up in borderless tables so the plots are LARGER than before while the
     block they sit in is roughly half as tall;
  2. tighten Section V (the RAG pipeline) hard;
  3. compress prose that is peripheral to the paper's topic;
  4. never shrink a plot below the size at which it is legible.

Operates on word/document.xml in the unpacked v6 document.
"""
import json
import re
import shutil
from pathlib import Path

from PIL import Image, ImageChops

DOC = Path("word/document.xml")
MEDIA = Path("word/media")
EMU = 914400

x = DOC.read_text(encoding="utf8")

# ----------------------------------------------------------------- 1. crop
# matplotlib pads every figure with a wide white gutter; at 48-71 per cent
# content the visible plot is far smaller than the box it occupies. Cropping
# to the ink lets the same box hold a much bigger plot.
CROPPED: dict[str, tuple[int, int]] = {}


def crop_media() -> None:
    pad = 8
    for f in sorted(MEDIA.iterdir()):
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        im = Image.open(f)
        rgb = im.convert("RGB")
        bg = Image.new("RGB", rgb.size, (255, 255, 255))
        mask = ImageChops.difference(rgb, bg).convert("L")
        bb = mask.point(lambda p: 255 if p > 12 else 0).getbbox()
        if not bb:
            CROPPED[f.name] = im.size
            continue
        l, t, r, b = bb
        l, t = max(0, l - pad), max(0, t - pad)
        r, b = min(im.size[0], r + pad), min(im.size[1], b + pad)
        if (r - l) * (b - t) < 0.98 * im.size[0] * im.size[1]:
            im = im.crop((l, t, r, b))
            im.save(f)
        CROPPED[f.name] = im.size
    print(f"  cropped {len(CROPPED)} images")


# ------------------------------------------------------- 2. pair up figures
def rel_map() -> dict:
    rels = Path("word/_rels/document.xml.rels").read_text(encoding="utf8")
    return dict(re.findall(r'Id="(rId\d+)"[^>]*Target="media/([^"]+)"', rels))


def paragraphs(src: str):
    return [m for m in re.finditer(r"<w:p [^>]*>.*?</w:p>", src, re.S)]


def set_extent(para: str, w_in: float, h_in: float) -> str:
    cx, cy = int(w_in * EMU), int(h_in * EMU)
    para = re.sub(r'<wp:extent cx="\d+" cy="\d+"',
                  f'<wp:extent cx="{cx}" cy="{cy}"', para)
    para = re.sub(r'<a:ext cx="\d+" cy="\d+"',
                  f'<a:ext cx="{cx}" cy="{cy}"', para)
    return para


def centre(para: str) -> str:
    """Force the paragraph to centre and kill its space-before/after."""
    if "<w:pPr>" in para:
        para = para.replace(
            "<w:pPr>",
            '<w:pPr><w:spacing w:after="0" w:before="0" w:line="200" '
            'w:lineRule="auto"/><w:jc w:val="center"/>', 1)
    else:
        para = re.sub(
            r"(<w:p [^>]*>)",
            r'\1<w:pPr><w:spacing w:after="0" w:before="0" w:line="200" '
            r'w:lineRule="auto"/><w:jc w:val="center"/></w:pPr>', para, 1)
    return para


def caption_small(para: str) -> str:
    """Caption at 7pt, tight leading, centred."""
    para = centre(para)
    para = re.sub(r"<w:rPr>", '<w:rPr><w:sz w:val="14"/><w:szCs w:val="14"/>',
                  para)
    para = para.replace("<w:rPr/>", '<w:rPr><w:sz w:val="14"/>'
                                    '<w:szCs w:val="14"/></w:rPr>')
    return para


COL_W = 5300           # dxa, the width of one text column


def n_up(cells: list[tuple[str, str]]) -> str:
    """A borderless one-row table holding n figures with their captions."""
    n = len(cells)
    cw = COL_W // n

    def cell(fig: str, cap: str) -> str:
        return (
            f'<w:tc><w:tcPr><w:tcW w:w="{cw}" w:type="dxa"/>'
            '<w:tcBorders>'
            '<w:top w:val="nil"/><w:left w:val="nil"/>'
            '<w:bottom w:val="nil"/><w:right w:val="nil"/></w:tcBorders>'
            '<w:tcMar><w:top w:w="0" w:type="dxa"/>'
            '<w:left w:w="15" w:type="dxa"/>'
            '<w:bottom w:w="0" w:type="dxa"/>'
            '<w:right w:w="15" w:type="dxa"/></w:tcMar>'
            '<w:vAlign w:val="bottom"/></w:tcPr>'
            f"{fig}{cap}</w:tc>")

    grid = "".join(f'<w:gridCol w:w="{cw}"/>' for _ in range(n))
    return (
        '<w:tbl><w:tblPr>'
        f'<w:tblW w:w="{cw * n}" w:type="dxa"/>'
        '<w:jc w:val="center"/>'
        '<w:tblBorders><w:top w:val="nil"/><w:left w:val="nil"/>'
        '<w:bottom w:val="nil"/><w:right w:val="nil"/>'
        '<w:insideH w:val="nil"/><w:insideV w:val="nil"/></w:tblBorders>'
        '<w:tblCellMar><w:top w:w="0" w:type="dxa"/>'
        '<w:left w:w="15" w:type="dxa"/>'
        '<w:bottom w:w="0" w:type="dxa"/>'
        '<w:right w:w="15" w:type="dxa"/></w:tblCellMar></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>"
        '<w:tr><w:trPr><w:cantSplit w:val="1"/></w:trPr>'
        + "".join(cell(f, c) for f, c in cells) +
        "</w:tr></w:tbl>")


def group_figures(group: list[str], img_w: float) -> None:
    """Replace a run of (figure, caption) pairs with one n-up table."""
    global x
    rmap = rel_map()
    ps = paragraphs(x)
    idx = {}
    for i, m in enumerate(ps):
        txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", m.group(0), re.S))
        for key in group:
            if txt.strip().startswith(key) and key not in idx:
                idx[key] = i
    if any(k not in idx for k in group):
        print(f"  [group] missed {group}")
        return

    def sized(fig: str) -> str:
        emb = re.search(r'r:embed="(rId\d+)"', fig).group(1)
        pw, ph = CROPPED.get(rmap[emb], (1000, 600))
        return centre(set_extent(fig, img_w, img_w * ph / pw))

    cells = []
    for key in group:
        i = idx[key]
        fig = ps[i - 1].group(0)
        if "<w:drawing>" not in fig:
            print(f"  [group] no drawing above {key}")
            return
        cells.append((sized(fig), caption_small(ps[i].group(0))))

    table = n_up(cells)
    # splice the table in where the first figure began, dropping every
    # original paragraph in the run but keeping any prose between them
    out = x[:ps[idx[group[0]] - 1].start()] + table
    prev = idx[group[0]]
    for key in group[1:]:
        i = idx[key]
        out += x[ps[prev].end():ps[i - 1].start()]
        prev = i
    out += x[ps[prev].end():]
    x = out
    print(f"  grouped {' + '.join(group)} at {img_w}in each")


def scale_figure(caption_key: str, w_in: float) -> None:
    """Resize a single unpaired figure, preserving its aspect ratio."""
    global x
    rmap = rel_map()
    ps = paragraphs(x)
    for i, m in enumerate(ps):
        txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", m.group(0), re.S))
        if txt.strip().startswith(caption_key):
            fig = ps[i - 1].group(0)
            if "<w:drawing>" not in fig:
                return
            emb = re.search(r'r:embed="(rId\d+)"', fig).group(1)
            pw, ph = CROPPED.get(rmap[emb], (1000, 600))
            new = set_extent(fig, w_in, w_in * ph / pw)
            x = x[:ps[i - 1].start()] + new + x[ps[i - 1].end():]
            print(f"  resized {caption_key} to {w_in}in")
            return
    print(f"  [scale] missed {caption_key}")


# ------------------------------------------------------- 3. tighten prose
def apply_tighten(pairs, label: str) -> int:
    global x
    saved = hits = 0
    for old, new in pairs:
        o, n = old.replace("&", "&amp;"), new.replace("&", "&amp;")
        if o in x:
            x = x.replace(o, n, 1)
            saved += len(o) - len(n)
            hits += 1
    print(f"  {label}: {hits}/{len(pairs)} applied, {saved} chars "
          f"(~{saved // 6} words)")
    return saved


# ----------------------------------------------- 4. structural compaction
def shrink_symbol_glossary() -> None:
    """The legend under (1) is one short line per symbol. Setting it to 7pt
    with no inter-line space keeps every definition and recovers the slack.
    Attribute-only: the equation's content controls are left untouched."""
    global x
    keys = ("S: total burnout score", "a: anger word count",
            "f: fear word count", "d: sadness word count",
            "j: joy word count", "the i-th behavioural condition",
            "indicator function", "sum of indicators")
    hits = 0
    for m in list(re.finditer(r"<w:p [^>]*>.*?</w:p>", x, re.S)):
        p = m.group(0)
        txt = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S))
        txt = re.sub(r"<[^>]+>", "", txt).strip()
        # legend entries are one short line each; the surrounding body
        # paragraphs mention the same terms and must keep body size
        if len(txt) > 155 or not any(k in txt for k in keys):
            continue
        new = re.sub(r'<w:sz w:val="\d+"/>', '<w:sz w:val="14"/>', p)
        new = re.sub(r'<w:szCs w:val="\d+"/>', '<w:szCs w:val="14"/>', new)
        new = new.replace("<w:rPr>",
                          '<w:rPr><w:sz w:val="14"/><w:szCs w:val="14"/>')
        new = new.replace("<w:rPr/>", '<w:rPr><w:sz w:val="14"/>'
                                      '<w:szCs w:val="14"/></w:rPr>')
        new = re.sub(r'<w:spacing [^/]*/>',
                     '<w:spacing w:after="0" w:before="0" w:line="180" '
                     'w:lineRule="auto"/>', new)
        if new != p:
            x = x.replace(p, new, 1)
            hits += 1
    # one legend line sits inside a content control, so the paragraph scan
    # above cannot reach it; set that run directly
    sdt = ('<w:rPr><w:rFonts w:ascii="Cardo" w:cs="Cardo" w:eastAsia="Cardo" '
           'w:hAnsi="Cardo"/><w:rtl w:val="0"/></w:rPr>'
           '<w:t xml:space="preserve">\u03c3(sentiment): sentiment term')
    if sdt in x:
        x = x.replace(sdt, sdt.replace(
            "<w:rPr>", '<w:rPr><w:sz w:val="14"/><w:szCs w:val="14"/>'), 1)
        hits += 1
    print(f"  symbol legend: {hits} lines set to 7pt, no leading")


def compact_tables() -> None:
    """Drop every table cell's vertical padding and set table text to 7pt.
    Across six tables and ~55 rows this recovers over an inch of column."""
    global x
    n = 0
    for m in list(re.finditer(r"<w:tbl>.*?</w:tbl>", x, re.S)):
        tbl = m.group(0)
        if "<w:drawing>" in tbl:          # leave the figure tables alone
            continue
        new = tbl
        new = re.sub(r'<w:tcMar>.*?</w:tcMar>', "", new, flags=re.S)
        new = new.replace(
            "<w:tcPr>",
            '<w:tcPr><w:tcMar><w:top w:w="0" w:type="dxa"/>'
            '<w:bottom w:w="0" w:type="dxa"/></w:tcMar>')
        new = re.sub(r'<w:sz w:val="\d+"/>', '<w:sz w:val="14"/>', new)
        new = re.sub(r'<w:szCs w:val="\d+"/>', '<w:szCs w:val="14"/>', new)
        new = new.replace(
            "<w:rPr>", '<w:rPr><w:sz w:val="14"/><w:szCs w:val="14"/>')
        new = new.replace(
            "<w:rPr/>", '<w:rPr><w:sz w:val="14"/><w:szCs w:val="14"/></w:rPr>')
        new = re.sub(r'<w:spacing w:after="\d+"',
                     '<w:spacing w:after="0"', new)
        x = x.replace(tbl, new, 1)
        n += 1
    print(f"  compacted {n} tables to 7pt with zero cell padding")


def shrink_references() -> None:
    """IEEE sets references smaller than body text; 8pt -> 7pt here."""
    global x
    i = x.find(">References<")
    if i < 0:
        print("  [refs] heading not found")
        return
    head, tail = x[:i], x[i:]
    before = tail.count('<w:sz w:val="16"/>')
    tail = tail.replace('<w:sz w:val="16"/>', '<w:sz w:val="13"/>')
    tail = tail.replace('<w:szCs w:val="16"/>', '<w:szCs w:val="13"/>')
    # the numbers tab to the hanging indent; it must clear "[22]" at 6.5pt
    # or two-digit entries jump to the next default stop and look ragged
    tail = tail.replace(
        '<w:ind w:left="260" w:hanging="260"/>',
        '<w:tabs><w:tab w:val="left" w:pos="240"/></w:tabs>'
        '<w:ind w:left="240" w:hanging="240"/>')
    # IEEE permits "et al." once a reference exceeds six authors; four of
    # these list eight to twelve, costing five lines of column between them.
    long_lists = [
        ("A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, "
         "A. N. Gomez, \u0141. Kaiser, and I. Polosukhin,", "A. Vaswani et al.,"),
        ("Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, "
         "M. Lewis, L. Zettlemoyer, and V. Stoyanov,", "Y. Liu et al.,"),
        ("P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, "
         "H. K\u00fcttler, M. Lewis, W. Yih, T. Rockt\u00e4schel, S. Riedel, and "
         "D. Kiela,", "P. Lewis et al.,"),
        ("V. Karpukhin, B. O\u011fuz, S. Min, P. Lewis, L. Wu, S. Edunov, "
         "D. Chen, and W. Yih,", "V. Karpukhin et al.,"),
        ("in Proc. 2019 Conf. Empirical Methods in Natural Language "
         "Processing and 9th Int. Joint Conf. Natural Language Processing "
         "(EMNLP-IJCNLP), Hong Kong, China, 2019",
         "in Proc. EMNLP-IJCNLP, Hong Kong, China, 2019"),
        ("in Proc. 2020 Conf. Empirical Methods in Natural Language "
         "Processing (EMNLP), 2020", "in Proc. EMNLP, 2020"),
        ("in Findings of the Association for Computational Linguistics: "
         "EMNLP 2020, 2020", "in Findings of ACL: EMNLP 2020"),
        ("LNCS vol. 3201, Berlin, Germany: Springer, 2004",
         "LNCS vol. 3201, Springer, 2004"),
        ("Qniverse, Quantum Technology Group, Centre for Development of "
         "Advanced Computing (C-DAC), Bengaluru, Ministry of Electronics and "
         "Information Technology, Government of India, 2026.",
         "Qniverse, Centre for Development of Advanced Computing (C-DAC), "
         "Ministry of Electronics and Information Technology, Government of "
         "India, 2026."),
        ("V. Havl\u00ed\u010dek, A. D. C\u00f3rcoles, K. Temme, A. W. Harrow, "
         "A. Kandala, J. M. Chow, and J. M. Gambetta,", "V. Havl\u00ed\u010dek et al.,"),
    ]
    n_et = 0
    for full, short in long_lists:
        if full in tail:
            tail = tail.replace(full, short, 1)
            n_et += 1
    x = head + tail
    print(f"  references set to 6.5pt ({before} runs), {n_et} author lists "
          f"abbreviated to et al.")


def tighten_leading() -> None:
    """Remove the 0.5pt of space Word adds after every body paragraph."""
    global x
    n = x.count('<w:spacing w:after="10"')
    x = x.replace('<w:spacing w:after="10"', '<w:spacing w:after="0"')
    print(f"  removed trailing space on {n} paragraphs")


def renumber_tables() -> None:
    """v6 numbers its tables I, II, IV, V, VI, VII — there is no Table III
    and nothing refers to one. Close the gap so the sequence is contiguous."""
    global x
    pairs = [("VII", "VI"), ("VI", "V"), ("V", "IV"), ("IV", "III")]
    tmp = {}
    for i, (old, _) in enumerate(pairs):
        tmp[old] = f"\u0001T{i}\u0001"
        x = re.sub(rf"\b(TABLE|Table) {old}\b", rf"\1 {tmp[old]}", x)
    n = 0
    for old, new in pairs:
        n += x.count(tmp[old])
        x = x.replace(tmp[old], new)
    print(f"  renumbered tables to close the missing Table III gap "
          f"({n} mentions)")


def main() -> None:
    global x
    crop_media()

    group_figures(["Fig. 7.", "Fig. 8."], 1.60)
    group_figures(["Fig. 9.", "Fig. 10."], 1.60)
    group_figures(["Fig. 1.", "Fig. 2.", "Fig. 3."], 1.06)
    group_figures(["Fig. 4.", "Fig. 5.", "Fig. 6."], 1.06)
    scale_figure("Fig. 11.", 4.9)

    total = 0
    for f, label in (("tighten.json", "general"),
                     ("tighten2.json", "section V + peripheral"),
                     ("tighten3.json", "abstract + related work"),
                     ("tighten4.json", "final pass")):
        total += apply_tighten(json.loads(Path(f).read_text()), label)

    shrink_symbol_glossary()
    compact_tables()
    renumber_tables()
    shrink_references()
    tighten_leading()

    # a stray section number left over from an earlier revision
    stray = ('<w:r><w:rPr><w:smallCaps w:val="1"/><w:rtl w:val="0"/></w:rPr>'
             '<w:t xml:space="preserve">XI.  </w:t></w:r>')
    if stray in x:
        x = x.replace(stray, "", 1)
        print("  removed stray section number")

    before = len(x)
    x = re.sub(r",? ?doi: ?10\.[^<\s]*?(?=[<\s])", "", x)
    print(f"  stripped DOIs: {before - len(x)} chars")

    # removing a trailing DOI can leave a reference without its full stop
    i = x.find(">References<")
    if i > 0:
        head, tail = x[:i], x[i:]
        n = len(re.findall(r"(\d)</w:t>", tail))
        tail = re.sub(r"(\d)</w:t>", r"\1.</w:t>", tail)
        x = head + tail
        print(f"  restored {n} reference full stops")

    DOC.write_text(x, encoding="utf8")
    print(f"document.xml written: {len(x):,} chars; prose saved {total}")


if __name__ == "__main__":
    main()
