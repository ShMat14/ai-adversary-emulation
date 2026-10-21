# -*- coding: utf-8 -*-
"""The technique-flow drawing from Neo4j Bloom, prepared for print.

Bloom exports the NEXT graph of analysis/v8_technique_graph.py as an SVG whose
labels would print at 3.9 to 5.2 pt at the journal's 119 mm figure width. This
keeps Bloom's drawing (layout, colours, edge widths and shades) and changes
only what print needs:

  * every circle is enlarged by RADIUS_SCALE about its centre, and every
    arrowhead and line end is moved back along its own direction by the same
    amount, so arrowheads still touch the enlarged circle;
  * every label is reset at one size, LABEL_PT at the printed width, centred
    in its circle on Bloom's own line breaks;
  * the background is white and the page is cropped to the drawing.

It refuses to write if a label line leaves its circle, two circles overlap,
or an arrowhead does not end on a circle. Outputs results/figures_v5/
techflow.pdf (vector) and techflow.png (600 dpi at the printed width).

    python analysis/v9_technique_figure.py
"""
from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import fitz

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "technique_graph" / "neo4j_exports" / "technique_flow_neo4j.svg"
OUT = ROOT / "results" / "figures_v5"

WIDTH_MM = 119          # Springer small-format figure width
LABEL_PT = 7.5          # 8 pt would push four labels out of their circles
RADIUS_SCALE = 1.15     # the closest pair of circles keeps an 11-unit gap
MARGIN = 8              # SVG units around the drawing
LINE_HEIGHT = 1.05      # in multiples of the font size
FONT = "Helvetica, Arial, sans-serif"   # MuPDF draws it with Nimbus Sans, metrics of 'helv'

NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"


def pairs(text):
    v = [float(x) for x in re.findall(NUM, text)]
    return list(zip(v[0::2], v[1::2]))


def main():
    svg = SRC.read_text(encoding="utf-8")
    head, rest = svg.split('<g class="node"', 1)
    nodes_part = '<g class="node"' + rest

    # Nodes: centre, radius, Bloom's label lines.
    nodes = []
    for g in re.findall(r'<g class="node"[^>]*>.*?</g>', nodes_part, re.S):
        c = re.search(rf'<circle cx="({NUM})" cy="({NUM})" r="({NUM})" fill="([^"]+)"', g)
        lines = re.findall(r"<tspan>([^<]*)</tspan>", g)
        nodes.append({"x": float(c[1]), "y": float(c[2]), "r": float(c[3]), "fill": c[4], "lines": lines,
                      "id": re.match(r'<g class="node" data-id="([^"]+)"', g)[1]})
    print(f"{len(nodes)} nodes")

    # Edges: a line (polyline or path) followed by its arrowhead polygon.
    edge_re = re.compile(r'(<(?:polyline points|path d)="([^"]+)"[^>]*/>)(<polygon points="([^"]+)"[^>]*/>)')
    edges = edge_re.findall(head)
    print(f"{len(edges)} edges")

    def target_of(tip):
        # Bloom stops each arrowhead 0 to 4 units short of its circle; the
        # next circle is always more than 30 units away.
        near = sorted(nodes, key=lambda n: abs(math.dist(tip, (n["x"], n["y"])) - n["r"]))
        off = math.dist(tip, (near[0]["x"], near[0]["y"])) - near[0]["r"]
        off2 = abs(math.dist(tip, (near[1]["x"], near[1]["y"])) - near[1]["r"])
        if not -0.5 <= off <= 5 or off2 < 20:
            raise SystemExit(f"arrowhead at {tip}: {off:.1f} units off the nearest circle, {off2:.1f} off the next")
        return near[0]

    new_head = head
    for line_tag, geom, poly_tag, poly_pts in edges:
        pts = pairs(poly_pts)
        # The tip is the vertex farthest from the midpoint of the other two.
        def far(i):
            o = [p for j, p in enumerate(pts) if j != i]
            mid = ((o[0][0] + o[1][0]) / 2, (o[0][1] + o[1][1]) / 2)
            return math.dist(pts[i], mid), mid
        ti = max(range(3), key=lambda i: far(i)[0])
        tip, mid = pts[ti], far(ti)[1]
        d = math.dist(tip, mid)
        ux, uy = (tip[0] - mid[0]) / d, (tip[1] - mid[1]) / d
        n = target_of(tip)
        shift = (RADIUS_SCALE - 1) * n["r"]
        dx, dy = -ux * shift, -uy * shift

        moved = " ".join(f"{x + dx},{y + dy}" for x, y in pts)
        new_poly = poly_tag.replace(poly_pts, moved)
        if line_tag.startswith("<polyline"):
            lp = pairs(geom)
            lp[-1] = (lp[-1][0] + dx, lp[-1][1] + dy)
            new_geom = " ".join(f"{x},{y}" for x, y in lp)
        else:
            # Bloom ends a curve with a degenerate Q segment (control = end):
            # move both of its points.
            k = geom.rindex("Q")
            q = pairs(geom[k:])
            new_geom = geom[:k] + "Q" + ",".join(f"{v}" for x, y in q for v in (x + dx, y + dy))
        new_head = new_head.replace(line_tag + poly_tag, line_tag.replace(geom, new_geom) + new_poly, 1)

    # Extent after enlarging, for the crop and the print scale.
    x0 = min(n["x"] - RADIUS_SCALE * n["r"] for n in nodes) - MARGIN
    x1 = max(n["x"] + RADIUS_SCALE * n["r"] for n in nodes) + MARGIN
    y0 = min(n["y"] - RADIUS_SCALE * n["r"] for n in nodes) - MARGIN
    y1 = max(n["y"] + RADIUS_SCALE * n["r"] for n in nodes) + MARGIN
    pt_per_unit = WIDTH_MM / 25.4 * 72 / (x1 - x0)
    F = LABEL_PT / pt_per_unit

    problems = []
    for a, b in itertools.combinations(nodes, 2):
        gap = math.dist((a["x"], a["y"]), (b["x"], b["y"])) - RADIUS_SCALE * (a["r"] + b["r"])
        if gap < 4:
            problems.append(f"circles {a['lines']} and {b['lines']} {gap:.1f} apart")

    groups = []
    for n in nodes:
        r = RADIUS_SCALE * n["r"]
        k = len(n["lines"])
        texts = []
        for i, t in enumerate(n["lines"]):
            base = n["y"] + (i - (k - 1) / 2) * LINE_HEIGHT * F + 0.35 * F
            # Cap height 0.72, descender 0.21 (Helvetica), counted only when
            # the line has a letter that descends.
            desc = 0.21 * F if set(t) & set("gjpqy(),;") else 0
            yy = max(abs(base - 0.72 * F - n["y"]), abs(base + desc - n["y"]))
            room = 2 * math.sqrt(max(r * r - yy * yy, 0))
            if fitz.get_text_length(t, "helv", F) + 6 > room:
                problems.append(f"label line '{t}' does not fit its circle")
            texts.append(f'<text x="{n["x"]}" y="{base:.2f}" text-anchor="middle" font-size="{F:.3f}" '
                         f'font-family="{FONT}" fill="#1a1b1d">{t}</text>')
        groups.append(f'<g class="node" data-id="{n["id"]}"><circle cx="{n["x"]}" cy="{n["y"]}" '
                      f'r="{r:.3f}" fill="{n["fill"]}"/>' + "".join(texts) + "</g>")
    if problems:
        raise SystemExit("not written:\n  " + "\n  ".join(problems))

    # White background; the drawing's own transform stays as Bloom wrote it.
    m = re.search(rf'translate\(({NUM}),({NUM})\) scale\(1\) translate\(({NUM}),({NUM})\)', new_head)
    ox, oy = float(m[1]) + float(m[3]), float(m[2]) + float(m[4])
    new_head = new_head.replace('fill="rgba(0,0,0,0)"', 'fill="#ffffff"').replace(
        "background: rgba(0, 0, 0, 0);", "background: #ffffff;")
    out_svg = new_head + "".join(groups) + "</g></svg>"

    doc = fitz.open("svg", out_svg.encode("utf-8"))
    pdf = fitz.open("pdf", doc.convert_to_pdf())
    page = pdf[0]
    page.set_cropbox(fitz.Rect(x0 + ox, y0 + oy, x1 + ox, y1 + oy))
    # Scale the cropped page so that its width is the printed width.
    out = fitz.open()
    w_pt = WIDTH_MM / 25.4 * 72
    h_pt = w_pt * (y1 - y0) / (x1 - x0)
    p2 = out.new_page(width=w_pt, height=h_pt)
    p2.show_pdf_page(p2.rect, pdf, 0)
    OUT.mkdir(parents=True, exist_ok=True)
    out.save(OUT / "techflow.pdf", garbage=4, deflate=True)
    pix = p2.get_pixmap(dpi=600, alpha=False)
    pix.set_dpi(600, 600)
    pix.save(OUT / "techflow.png")

    sizes = {round(s["size"], 2) for b in p2.get_text("dict")["blocks"] for l in b.get("lines", [])
             for s in l["spans"]}
    print(f"printed {WIDTH_MM} x {h_pt / 72 * 25.4:.0f} mm; label size(s) {sorted(sizes)} pt; "
          f"PNG {pix.width} x {pix.height} px at 600 dpi")


if __name__ == "__main__":
    main()
