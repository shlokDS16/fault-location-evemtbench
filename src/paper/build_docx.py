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
        if kind == "sec":
            nums = [n for n in nums]
        return (sing if len(nums) == 1 else plur) + "~" + join(nums)

    tex = re.sub(r"\\(cref|Cref)\{([^}]*)\}", cref, tex)
    tex = re.sub(r"\\eqref\{([^}]*)\}", lambda g: "(" + one(g.group(1)) + ")", tex)
    tex = re.sub(r"\\ref\{([^}]*)\}", lambda g: one(g.group(1)), tex)
    return tex


def resolve_cites(tex, order):
    num = {k: i + 1 for i, k in enumerate(order)}

    def cite(g):
        ns = sorted(num[k.strip()] for k in g.group(1).split(","))
        out, i = [], 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                j += 1
            out.append(f"[{ns[i]}]" if j == i else (f"[{ns[i]}], [{ns[j]}]" if j == i + 1 else f"[{ns[i]}]--[{ns[j]}]"))
            i = j + 1
        return ", ".join(out)

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
        it = re.sub(r"\s+", " ", it).strip()
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
        names = re.sub(r"\\thanks\{.*", "", block[len("\\author{"):], flags=re.S).replace("~", " ").strip("% \n")
        body = body[:am.start()] + (
            f"\\begin{{center}}{names}\\end{{center}}\n\n"
            + "".join(f"\\noindent\\emph{{{n.strip()}}}\n\n" for n in notes)) + body[end:]
    body = re.sub(r"\\markboth\{.*?\}\{.*?\}\n", "", body)
    body = body.replace("\\maketitle", "")
    body = re.sub(r"\\IEEEPARstart\{(\w)\}\{(\w*)\}", r"\1\2", body)
    body = re.sub(r"\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}",
                  r"\\noindent\\emph{Index Terms}---\1", body, flags=re.S)
    # abstract in place (pandoc would move it to the top as metadata)
    body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
                  r"\\noindent\\textbf{Abstract}---\1", body, flags=re.S)
    # IEEE heading numbers: I., II. for sections, A., B. for subsections
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]
    cnt = {"s": 0, "ss": 0}

    def head(g):
        kind, star, text = g.group(1), g.group(2), g.group(3)
        if star:
            return g.group(0)
        if kind == "section":
            cnt["s"] += 1
            cnt["ss"] = 0
            return f"\\section*{{{roman[cnt['s'] - 1]}. {text}}}"
        cnt["ss"] += 1
        return f"\\subsection*{{{chr(64 + cnt['ss'])}. {text}}}"
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
    body = re.sub(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", cap("table", "TABLE", True), body, flags=re.S)
    # constructs pandoc's LaTeX reader does not handle in tables (\quad before a number swallows the number)
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}",
                  lambda g: re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "",
                                   re.sub(r"\\quad\s*", "~~~", g.group(0))), body, flags=re.S)
    body = re.sub(r"(\\multicolumn\{\d+\}\{)@\{\}([lcr])(@\{\})?\}", r"\1\2}", body)
    body = re.sub(r"\\shortstack\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", lambda g: g.group(1).replace("\\\\", " "), body)
    body = re.sub(r"\\(begin|end)\{(table|figure)\*\}", r"\\\1{\2}", body)
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
    head = ("\\documentclass{article}\n\\usepackage{amsmath}\n\\begin{document}\n"
            f"\\begin{{center}}\\textbf{{{title}}}\\end{{center}}\n")
    return head + body + "\n\\end{document}\n"


def postprocess(path):
    d = Document(path)
    for s in d.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(1.0)
        ln = OxmlElement("w:lnNumType")
        ln.set(qn("w:countBy"), "1")
        ln.set(qn("w:restart"), "continuous")
        ln.set(qn("w:distance"), "300")
        s._sectPr.append(ln)
        fp = s.footer.paragraphs[0]
        fp.text = "Review copy for the corresponding author (single column). The journal version is the IEEEtran PDF."
        fp.runs[0].font.size = Pt(8)
    for st in d.styles:
        try:
            if st.font is not None:
                st.font.name = "Times New Roman"
        except (AttributeError, ValueError):
            pass
    d.core_properties.title = ("Where Learning Helps in Impedance-Based Fault Location: An Equal-Information Evaluation "
                               "on an Open EMT Benchmark (review copy)")
    d.core_properties.author = "Shlok Goenka; Ganesh Khekare"
    d.styles["Normal"].font.size = Pt(11)
    for name, size in (("Heading 1", 13), ("Heading 2", 11.5), ("Heading 3", 11)):
        if name in [s.name for s in d.styles]:
            st = d.styles[name]
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.italic = name != "Heading 1"
            st.font.color.rgb = None
            rpr = st.element.get_or_add_rPr()
            for tag in ("w:rFonts",):
                for el in rpr.findall(qn(tag)):
                    rpr.remove(el)
            f = OxmlElement("w:rFonts")
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
                f.set(qn(a), "Times New Roman")
            rpr.append(f)
            for el in rpr.findall(qn("w:color")):
                rpr.remove(el)
    for t in d.tables:
        t.style = d.styles["Table Grid"] if "Table Grid" in [s.name for s in d.styles] else t.style
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
    ref = BUILD / "reference.docx"
    if not ref.exists():
        with open(ref, "wb") as fh:
            fh.write(subprocess.run(["pandoc", "-o", "-", "--print-default-data-file", "reference.docx"],
                                    capture_output=True).stdout)
    r = subprocess.run(["pandoc", "review.tex", "-f", "latex", "-t", "docx", "-o", str(OUT),
                        "--resource-path", str(BUILD)], cwd=BUILD, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    if r.stderr.strip():
        print("pandoc:", r.stderr.strip()[:2000])
    postprocess(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
