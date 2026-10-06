"""IEEEtran.bst does not print the doi field. Add a hyperlinked DOI note to every bib entry that has a doi, and the
arXiv DOI (10.48550/arXiv.<id>, assigned by arXiv/DataCite to every preprint) to arXiv-only entries. Idempotent
(entries already carrying the marker field doinote are skipped).
Usage: python src/paper/add_doi_notes.py"""
import re
from pathlib import Path

BIB = Path(__file__).resolve().parents[2] / "paper" / "refs.bib"
s = BIB.read_text(encoding="utf-8")


def entry_end(text, start):
    """Index just after the brace that closes the entry whose '@type{' begins at start."""
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced entry")


out, pos, n = [], 0, 0
for m in re.finditer(r"^@\w+\{", s, flags=re.M):
    if m.start() < pos:
        continue
    end = entry_end(s, m.start())
    out.append(s[pos:m.start()])
    e = s[m.start():end]
    pos = end
    if "doinote" in e:
        out.append(e)
        continue
    doi = re.search(r"^\s*doi\s*=\s*\{([^}]*)\}", e, re.M)
    arx = re.search(r"^\s*eprint\s*=\s*\{([^}]*)\}", e, re.M)
    d = doi.group(1) if doi else (f"10.48550/arXiv.{arx.group(1)}" if arx else None)
    if d is None:
        out.append(e)
        continue
    link = f"doi: \\href{{https://doi.org/{d}}}{{{d}}}"
    body = e[:-1].rstrip().rstrip(",")
    if re.search(r"^\s*note\s*=", body, re.M):
        body = re.sub(r"(^\s*note\s*=\s*\{)", lambda g: g.group(1) + link + ". ", body, count=1, flags=re.M)
    else:
        body += f",\n  note          = {{{link}}}"
    out.append(body + ",\n  doinote       = {done}\n}")
    n += 1
out.append(s[pos:])
BIB.write_bytes("".join(out).encode("utf-8"))
print(n, "entries with a DOI note")
