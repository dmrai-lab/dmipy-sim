"""The certificate's scanner classes as a table with their sources, rendered from the catalogue (dmipy-sim#220).

    python tools/scanner_table.py            # Markdown
    python tools/scanner_table.py --latex    # a LaTeX tabular body

One row per class of ``scanner_constants.json``: the machine, its per-axis amplitude and slew rate, the source
each number cites and its confidence. A paper types nothing by hand: the table is the catalogue.
"""
import argparse

from dmipy_sim.acquisition import scanner_constants as scc


def rows():
    """``(class, model, G_max mT/m, slew T/m/s, source, confidence)`` per certificate class."""
    out = []
    for cls, key in scc.SCANNER_CONSTANTS["classes"].items():
        _, entry, _ = scc.resolve(key)
        g, sl = entry["gradient"]["max_amplitude"], entry["gradient"]["max_slew_rate"]
        keys = {g["source_key"], sl["source_key"]}
        src = "; ".join(f"{scc.get_citation(k)['authors']} {scc.get_citation(k)['year']}, {scc.get_citation(k)['doi_or_url']}"
                        for k in sorted(keys))
        conf = g["confidence"] if g["confidence"] == sl["confidence"] else f"{g['confidence']} amplitude, {sl['confidence']} slew"
        out.append((cls, entry.get("model", key), g["value"], sl["value"], src, conf))
    return out


def markdown():
    lines = ["| class | system | G_max (mT/m) | slew (T/m/s) | source | confidence |", "|---|---|---:|---:|---|---|"]
    lines += [f"| `{c}` | {m} | {g:g} | {s:g} | {src} | {conf} |" for c, m, g, s, src, conf in rows()]
    return "\n".join(lines)


def latex():
    esc = lambda t: t.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")
    return "\n".join(f"{esc(c)} & {esc(m)} & {g:g} & {s:g} & {esc(src)} & {esc(conf)} \\\\" for c, m, g, s, src, conf in rows())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--latex", action="store_true", help="a LaTeX tabular body instead of Markdown")
    a = ap.parse_args()
    print(latex() if a.latex else markdown())
