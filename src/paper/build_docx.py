"""Build the Word review copy of the manuscript for the corresponding author (single column, line numbers).

The journal receives the IEEEtran PDF (paper/main.pdf). This script turns the same LaTeX source into a .docx that
Word can comment on: macros expanded, cross-references and citations resolved from paper/main.aux and main.bbl
(so the numbers match the PDF), TikZ figures and algorithms rendered to PNG, equations as native Word equations,
numbered IEEE-style references. Run after a full LaTeX build of paper/main.tex.
Usage: python src/paper/build_docx.py
"""
import re
import shutil
import subprocess
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
BUILD = ROOT / "paper" / "docx_build"
OUT = ROOT / "submission" / "Manuscript_review_copy_for_supervisor.docx"


def read(p):
    return Path(p).read_text(encoding="utf-8")


def flatten(tex):
    def rep(m):
        f = PAPER / (m.group(1) + ("" if m.group(1).endswith(".tex") else ".tex"))
        return flatten(read(f)) if f.exists() else ""
    return re.sub(r"\\input\{([^}]*)\}", rep, tex)


def strip_comments(tex):
    return re.sub(r"(?<!\\)%.*", "", tex)


def balanced(s, i):
    """s[i] == '{'; return index after the matching '}'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def macros():
    m = {}
    for src in (PAPER / "macros_auto.tex", PAPER / "main.tex"):
        t = strip_comments(read(src))
        for g in re.finditer(r"\\newcommand\{\\(\w+)\}\{", t):
            end = balanced(t, g.end() - 1)
            m[g.group(1)] = t[g.end():end - 1]
    return m


def expand_macros(tex, mac):
    names = sorted(mac, key=len, reverse=True)
    pat = re.compile(r"\\(" + "|".join(names) + r")(?![A-Za-z])(\{\})?")
    for _ in range(3):
        tex = pat.sub(lambda g: mac[g.group(1)], tex)
    return tex.replace("{,}", ",")


def labels():
    lab = {}
    aux = read(PAPER / "main.aux")
    for g in re.finditer(r"\\newlabel\{([^}@]*)\}\{\{", aux):
        end = balanced(aux, g.end() - 1)
        val = aux[g.end():end - 1]
        val = re.sub(r"\\mbox\s*\{(.*)\}", r"\1", val)
        lab[g.group(1)] = val.strip()
    return lab


def bib_order():
    return re.findall(r"\\bibitem\{([^}]*)\}", read(PAPER / "main.bbl"))


PREFIX = {"fig": ("Fig.", "Figs."), "tab": ("Table", "Tables"), "sec": ("Section", "Sections"),
          "alg": ("Algorithm", "Algorithms"), "eq": ("", "")}


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]


def sec_arabic(n):
    """IEEE section label 'VI-B' -> '6.2', 'III' -> '3'."""
    part = n.split("-")
    s = str(ROMAN.index(part[0]) + 1)
    return s + (f".{ord(part[1]) - 64}" if len(part) > 1 else "")


def join(items):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def resolve_refs(tex, lab):
    def one(key):
        if key not in lab:
            raise KeyError("undefined label " + key)
        return lab[key]

    def cref(g):
        keys = [k.strip() for k in g.group(2).split(",")]
        kind = keys[0].split(":")[0]
        nums = [one(k) for k in keys]
        if kind == "eq":
            return ("Eq. " if len(nums) == 1 else "Eqs. ") + join([f"({n})" for n in nums])
        sing, plur = PREFIX.get(kind, ("", ""))
        if kind == "sec":  # the docx numbers headings 1., 1.1. (as the eTransportation reference), not I., I-A
            nums = [sec_arabic(n) for n in nums]
        return (sing if len(nums) == 1 else plur) + "~" + join(nums)

    tex = re.sub(r"\\(cref|Cref)\{([^}]*)\}", cref, tex)
    tex = re.sub(r"\\eqref\{([^}]*)\}", lambda g: "(" + one(g.group(1)) + ")", tex)
    tex = re.sub(r"\\ref\{([^}]*)\}", lambda g: one(g.group(1)), tex)
    return tex


def resolve_cites(tex, order):
    num = {k: i + 1 for i, k in enumerate(order)}

    def cite(g):
        ns = sorted(num[k.strip()] for k in g.group(1).split(","))
        return ", ".join(f"[{n}]" for n in ns)  # no dashed ranges (matches \usepackage[nocompress]{cite})

    return re.sub(r"\\cite\{([^}]*)\}", cite, tex)


def bibliography_tex():
    bbl = read(PAPER / "main.bbl")
    items = re.split(r"\\bibitem\{[^}]*\}", bbl.split("\\BIBdecl", 1)[1])[1:]
    out = ["\\section*{References}"]
    for i, it in enumerate(items, 1):
        it = it.replace("\\end{thebibliography}", "")
        it = re.sub(r"\\BIBentry\w+", "", it)
        it = re.sub(r"\\hskip[^\\]*\\relax", " ", it)
        it = it.replace("\\newblock", " ")
        it = re.sub(r"\\bibinfo\{[^}]*\}", "", it)
        it = re.sub(r"\s+", " ", it).strip().replace("--", "-")  # page ranges with a hyphen, no en dash
        out.append(f"[{i}]~{it}\n")
    return "\n\n".join(out)


def preamble_keep():
    """Packages and algorithm/TikZ settings of main.tex needed to render the snippets on their own."""
    pre = read(PAPER / "main.tex").split("\\begin{document}")[0]
    keep = [ln for ln in pre.splitlines()
            if re.match(r"\\(SetKw|DontPrint|SetAlgo|usetikzlibrary|newcommand)", ln)
            or re.match(r"\\usepackage.*\{(amsmath|xcolor|booktabs|tikz|algorithm2e)", ln)]
    return "\n".join(keep) + "\n"



def render_snippets(tex, preamble_extra):
    """Render every tikzpicture and algorithm environment to PNG; replace it by \\includegraphics."""
    BUILD.mkdir(parents=True, exist_ok=True)
    snippets = []

    def grab(env, t):
        pat = re.compile(r"\\begin\{" + env + r"\}(\[[^\]]*\])?(.*?)\\end\{" + env + r"\}", re.S)
        return pat

    # algorithms: whole environment incl. caption (caption numbering forced to the PDF's number)
    alg_pat = re.compile(r"\\begin\{algorithm\}(\[[^\]]*\])?(.*?)\\end\{algorithm\}", re.S)
    tikz_pat = re.compile(r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}", re.S)
    lab = labels()

    def alg_rep(g):
        body = g.group(2)
        lm = re.search(r"\\label\{([^}]*)\}", body)
        n = lab[lm.group(1)] if lm else "?"
        k = len(snippets)
        snippets.append(("alg", n, body))
        return f"\n\n\\begin{{figure}}\\centering\\includegraphics[width=3.5in]{{snip{k}.png}}\\end{{figure}}\n\n"

    def tikz_rep(g):
        k = len(snippets)
        snippets.append(("tikz", None, g.group(0)))
        return f"\\includegraphics[width=3.5in]{{snip{k}.png}}"

    tex = alg_pat.sub(alg_rep, tex)
    tex = tikz_pat.sub(tikz_rep, tex)
    for k, (kind, n, body) in enumerate(snippets):
        if kind == "alg":
            doc = ("\\documentclass[10pt]{article}\n\\usepackage[paperwidth=4in,paperheight=10in,margin=0.15in]"
                   "{geometry}\n" + preamble_keep() + preamble_extra + "\\pagestyle{empty}\n\\begin{document}\n"
                   f"\\setcounter{{algocf}}{{{int(n) - 1}}}\n\\begin{{algorithm}}[H]{body}\\end{{algorithm}}\n"
                   "\\end{document}\n")
        else:
            doc = ("\\documentclass[10pt,border=3pt]{standalone}\n" + preamble_keep() + preamble_extra +
                   "\\begin{document}\n" + body + "\n\\end{document}\n")
        f = BUILD / f"snip{k}.tex"
        f.write_text(doc, encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", f.name], cwd=BUILD,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"snippet {k} failed:\n" + r.stdout[-2000:])
        pdf = fitz.open(BUILD / f"snip{k}.pdf")
        page = pdf[0]
        rects = [fitz.Rect(b[:4]) for b in page.get_text("blocks")]
        rects += [d["rect"] for d in page.get_drawings()]
        clip = rects[0]
        for r_ in rects[1:]:
            clip |= r_
        clip = (clip + (-4, -4, 4, 4)) & page.rect
        page.get_pixmap(dpi=300, clip=clip).save(BUILD / f"snip{k}.png")
    return tex


TABLE_RULES = []  # per tabular, in document order: {row index: "top" | "mid"} and the number of rows


def record_rules(body):
    """Positions of \\toprule / \\midrule / \\bottomrule in every tabular, for drawing the same rules in Word."""
    TABLE_RULES.clear()
    for g in re.finditer(r"\\begin\{tabular\}(\{(?:[^{}]|\{[^{}]*\})*\})?(.*?)\\end\{tabular\}", body, flags=re.S):
        segs = re.split(r"\\\\", g.group(2))
        rules, rows, pending = {}, 0, None
        for seg in segs:
            for m in re.finditer(r"\\(toprule|midrule|bottomrule)", seg):
                pending = "top" if m.group(1) == "toprule" else ("mid" if m.group(1) == "midrule" else pending)
            content = re.sub(r"\\(toprule|midrule|bottomrule|hline)|\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", seg).strip()
            if content:
                if pending:
                    rules[rows] = pending
                    pending = None
                rows += 1
        TABLE_RULES.append((rules, rows))


def to_pandoc_latex(tex, mac):
    body = tex.split("\\begin{document}", 1)[1].split("\\end{document}", 1)[0]
    pre = tex.split("\\begin{document}", 1)[0]
    title = re.search(r"\\title\{(.*?)\}\s*\n", pre + body, re.S).group(1)
    body = re.sub(r"\\title\{.*?\}\s*\n", "", body, flags=re.S)
    # author block -> plain lines
    am = re.search(r"\\author\{", body)
    if am:
        end = balanced(body, am.end() - 1)
        block = body[am.start():end]
        notes = []  # \thanks footnotes of the author block (e.g. the AI-use note) as small paragraphs
        for t in re.finditer(r"\\thanks\{", block):
            notes.append(block[t.end():balanced(block, t.end() - 1) - 1])
        # front matter laid out as in the eTransportation reference (styles applied in postprocess via markers)
        ai_note = " ".join(n.strip() for n in notes if "AI system" in n)
        body = body[:am.start()] + (
            f"ZZTITLE {title}\n\n"
            "ZZAUTHOR Shlok Goenka\\textsuperscript{a}, Ganesh Khekare\\textsuperscript{a,*}\n\n"
            "ZZAFFIL \\textsuperscript{a}School of Computer Science and Engineering, Vellore Institute of Technology, "
            "Vellore, 632014, Tamil Nadu, India\n\n"
            "ZZNOTE *Corresponding author. E-mail addresses: shlok.goenka2023@vitstudent.ac.in (S. Goenka), "
            f"ganesh.khekare@vit.ac.in (G. Khekare). {ai_note}\n\n") + body[end:]
    body = re.sub(r"\\markboth\{.*?\}\{.*?\}\n", "", body)
    body = body.replace("\\maketitle", "")
    body = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w*)\}", r"\1\2", body)
    body = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
                  lambda g: "ZZKEYS Keywords: " + "; ".join(k.strip().rstrip(".") for k in g.group(1).split(","))
                  + "\n\n", body, flags=re.S)
    # abstract in place (pandoc would move it to the top as metadata)
    body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  r"\\section*{Abstract}\n\nZZABSTRACT \1", body, flags=re.S)
    # heading numbers as in the reference: 1., 2. for sections and 2.1., 2.2. for subsections
    cnt = {"s": 0, "ss": 0}

    def head(g):
        kind, star, text = g.group(1), g.group(2), g.group(3)
        if star:
            return g.group(0)
        if kind == "section":
            cnt["s"] += 1
            cnt["ss"] = 0
            return f"\\section*{{{cnt['s']}. {text}}}"
        cnt["ss"] += 1
        return f"\\subsection*{{{cnt['s']}.{cnt['ss']}. {text}}}"
    body = re.sub(r"\\(section|subsection)(\*?)\{([^}]*)\}", head, body)
    body = re.sub(r"\\bibliographystyle\{[^}]*\}", "", body)
    body = re.sub(r"\\bibliography\{[^}]*\}", lambda g: bibliography_tex(), body)
    body = body.replace("\\pct", "\\,\\%")
    # figure/table captions get their numbers (Word does not number pandoc captions)
    lab = labels()

    def cap(env, word, roman):
        def f(g):
            blk = g.group(0)
            lm = re.search(r"\\label\{([^}]*)\}", blk)
            if not lm or "\\caption{" not in blk:
                return blk
            n = lab[lm.group(1)]
            return blk.replace("\\caption{", f"\\caption{{{word} {n}. ", 1)
        return f
    body = re.sub(r"\\begin\{figure\*?\}.*?\\end\{figure\*?\}", cap("figure", "Fig.", False), body, flags=re.S)
    body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", cap("table", "Table", True), body, flags=re.S)
    # constructs pandoc's LaTeX reader does not handle in tables (\quad before a number swallows the number)
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}",
                  lambda g: re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "",
                                   re.sub(r"\\quad\s*", "~~~", g.group(0))), body, flags=re.S)
    body = re.sub(r"(\\multicolumn\{\d+\}\{)@\{\}([lcr])(@\{\})?\}", r"\1\2}", body)
    body = re.sub(r"\\shortstack\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", lambda g: g.group(1).replace("\\\\", " "), body)
    body = re.sub(r"\\(begin|end)\{(table|figure)\*\}", r"\\\1{\2}", body)
    record_rules(body)
    body = body.replace("{figs/", "{../figs/").replace(".pdf}", ".png}")
    # figures at their print width: one column 3.5 in, full width = the 6.5 in text block of the docx
    body = body.replace("width=\\columnwidth", "width=3.5in").replace("width=\\textwidth", "width=6.5in")
    # numbered display equations: append the number (Word equations carry no automatic numbers)

    def eqnum(g):
        env, inner = g.group(1), g.group(2)
        lm = re.search(r"\\label\{([^}]*)\}", inner)
        inner = re.sub(r"\\label\{[^}]*\}", "", inner)
        inner = inner.replace("\\nonumber", "")
        n = lab[lm.group(1)] if lm else None
        tag = f"\\qquad\\text{{({n})}}" if n else ""
        if env.startswith("align"):
            return "\\[\\begin{aligned}" + inner.strip() + "\\end{aligned}" + tag + "\\]"
        return "\\[" + inner.strip() + tag + "\\]"
    body = re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", eqnum, body, flags=re.S)
    head = "\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
    return head + body + "\n\\end{document}\n"


def set_borders(parent, tag, edges):
    """Replace the <w:tag> border block (tblBorders or tcBorders) of parent with the given {edge: (val, size)}."""
    for old in parent.findall(qn(f"w:{tag}")):
        parent.remove(old)
    box = OxmlElement(f"w:{tag}")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        if edge in edges:
            val, sz = edges[edge]
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), val)
            el.set(qn("w:sz"), str(sz))
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "auto")
            box.append(el)
    parent.append(box)


def postprocess(path):
    """Styles of the eTransportation reference (page, fonts, headings come from it via --reference-doc); no line
    numbers, no footer. Front-matter paragraphs are found by their markers and given the reference's styles."""
    d = Document(path)
    names = {s.name for s in d.styles}
    marks = {"ZZTITLE ": "Title", "ZZAUTHOR ": "Author", "ZZAFFIL ": "Affiliation", "ZZNOTE ": "Author Note",
             "ZZABSTRACT ": "Abstract", "ZZKEYS ": "Keywords"}
    in_refs = False
    for p in d.paragraphs:
        for m, style in marks.items():
            if p.text.startswith(m):
                for r in p.runs:
                    if m.strip() in r.text:
                        r.text = r.text.replace(m, "").replace(m.strip(), "")
                        break
                for r in p.runs:  # strip the space left by the marker from the first non-empty run
                    if r.text.strip():
                        r.text = r.text.lstrip()
                        break
                    r.text = ""
                if style in names:
                    p.style = d.styles[style]
        if p.style.name.startswith("Heading"):
            in_refs = p.text.strip() == "References"
        elif in_refs and p.text.strip() and "Bibliography" in names:
            p.style = d.styles["Bibliography"]
        if p.style.name == "Image Caption" and "Figure Caption" in names:
            p.style = d.styles["Figure Caption"]
    # three-line tables as in the reference: 1.5 pt rule on top and bottom, 0.75 pt rule under the header and
    # between row groups (the LaTeX \midrule positions), no vertical or inner lines
    if len(TABLE_RULES) != len(d.tables):
        raise RuntimeError(f"{len(TABLE_RULES)} LaTeX tables but {len(d.tables)} Word tables")
    for t, (rules, nrows) in zip(d.tables, TABLE_RULES):
        if len(t.rows) != nrows:
            raise RuntimeError(f"row count mismatch: LaTeX {nrows}, Word {len(t.rows)} ({t.rows[0].cells[0].text!r})")
        set_borders(t._tbl.tblPr, "tblBorders", {k: ("nil", 0) for k in
                                                 ("top", "left", "bottom", "right", "insideH", "insideV")})
        for i, row in enumerate(t.rows):
            edges = {}
            if rules.get(i) == "top" or i == 0:
                edges["top"] = ("single", 12)
            elif rules.get(i) == "mid":
                edges["top"] = ("single", 6)
            if i == len(t.rows) - 1:
                edges["bottom"] = ("single", 12)
            if edges:
                for tc in row._tr.findall(qn("w:tc")):
                    set_borders(tc.get_or_add_tcPr(), "tcBorders", edges)
    for t in d.tables:  # keep each table, and its caption, on one page
        prev = t._tbl.getprevious()
        while prev is not None and prev.tag.endswith(("bookmarkEnd", "bookmarkStart")):
            prev = prev.getprevious()
        if prev is not None and prev.tag == qn("w:p"):
            ppr = prev.get_or_add_pPr()
            if ppr.find(qn("w:keepNext")) is None:
                ppr.append(OxmlElement("w:keepNext"))
        for i, row in enumerate(t.rows):
            trpr = row._tr.get_or_add_trPr()
            if trpr.find(qn("w:cantSplit")) is None:
                trpr.append(OxmlElement("w:cantSplit"))
            for c in row.cells:  # last row too: the table note below stays with the table
                for par in c.paragraphs:
                    par.paragraph_format.keep_with_next = True
        nxt = t._tbl.getnext()  # the table note: one block, never split over two pages (skip bookmark tags)
        while nxt is not None and nxt.tag != qn("w:p") and nxt.tag.endswith(("bookmarkEnd", "bookmarkStart")):
            nxt = nxt.getnext()
        if nxt is not None and nxt.tag == qn("w:p"):
            ppr = nxt.get_or_add_pPr()
            if ppr.find(qn("w:keepLines")) is None:
                ppr.append(OxmlElement("w:keepLines"))
    for t in d.tables:  # wide tables (Table I: 8 columns) in a smaller font so header words do not break
        if len(t.columns) >= 7:
            for row in t.rows:
                for c in row.cells:
                    for par in c.paragraphs:
                        for r in par.runs:
                            r.font.size = Pt(8.5)
    d.core_properties.title = ("Where Learning Helps in Impedance-Based Fault Location: An Equal-Information Evaluation "
                               "on an Open EMT Benchmark")
    d.core_properties.author = "Shlok Goenka; Ganesh Khekare"
    d.save(path)


def main():
    tex = strip_comments(flatten(read(PAPER / "main.tex")))
    mac = macros()
    pre_extra = "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items())
    tex = expand_macros(tex, mac)
    tex = resolve_refs(tex, labels())
    tex = resolve_cites(tex, bib_order())
    tex = render_snippets(tex, "")
    src = to_pandoc_latex(tex, mac)
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "review.tex").write_text(src, encoding="utf-8")
    ref = ROOT / "submission" / "manuscript_eTransportation.docx"  # style and page reference chosen by the authors
    r = subprocess.run(["pandoc", "review.tex", "-f", "latex", "-t", "docx", "-o", str(OUT), "--reference-doc", str(ref),
                        "--resource-path", str(BUILD)], cwd=BUILD, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    if r.stderr.strip():
        print("pandoc:", r.stderr.strip()[:2000])
    postprocess(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
