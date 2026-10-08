# -*- coding: utf-8 -*-
"""The five box diagrams, redrawn so that nothing overflows.

Labels ran out of their boxes and arrows started in the wrong places. Two causes, both fixed here rather than by eye:

  * Size. Figures 1 and 2 were drawn about 12 in wide and printed at 5.8 in,
    so every label was shrunk to half its size. Everything here is drawn at
    the width it is printed at, with no label under 7 pt.
  * Placement. The old boxes were positioned by hand and never measured
    against their text, so labels ran past box edges and arrows started
    inside boxes. Here every box is sized from its measured text, every
    arrow runs from one box edge to another, and `Canvas.verify()` refuses
    to save a figure in which any label leaves its box or the canvas, any
    two boxes overlap, or any line crosses a box it does not connect.

Figure 2 is also redesigned as a corporate network: Internet, DMZ and the internal
network, the internal network split into its four subnets, with a switch per
subnet and a firewall on every link the environment permits. The environment
has no firewall or switch objects; it enforces reachability as a rule
(`Topology.reachable_from`: same subnet, or an adjacent one), and the devices
are drawn where that rule applies. The caption and Section 3.4 say so.

Writes results/figures_v5/{architecture,topology,mask,detection,scenario}
as 600 dpi PNG and vector PDF.

    python analysis/v7_diagrams.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "figures_v5"

from env.topology_v5 import Topology                               # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42})

W = 5.8          # printed width in the manuscript, inches
MIN_PT = 7.0     # nothing smaller survives a page
LS = 1.30        # line spacing, in units of the font size

INK = "#1f2937"
SOFT = "#4b5563"
LINE = "#6b7280"
PAL = {
    "neutral": ("#f3f4f6", "#4b5563"),
    "blue": ("#dbe7f5", "#2f5d8a"),
    "tan": ("#f6e9d6", "#8a5a1f"),
    "red": ("#f8e1e1", "#a33b3b"),
    "green": ("#e2f0e6", "#1e6b45"),
    "region": ("#fafafa", "#9ca3af"),
    "device": ("#e5e7eb", "#4b5563"),
}


@dataclass
class Box:
    name: str
    x0: float
    y0: float
    x1: float
    y1: float
    kind: str = "solid"        # solid | region | inline
    @property
    def cx(self): return (self.x0 + self.x1) / 2
    @property
    def cy(self): return (self.y0 + self.y1) / 2
    def top(self, f=0.5): return (self.x0 + f * (self.x1 - self.x0), self.y1)
    def bottom(self, f=0.5): return (self.x0 + f * (self.x1 - self.x0), self.y0)
    def left(self, f=0.5): return (self.x0, self.y0 + f * (self.y1 - self.y0))
    def right(self, f=0.5): return (self.x1, self.y0 + f * (self.y1 - self.y0))


Line = tuple  # (text, size, weight, colour)


class Canvas:
    def __init__(self, w: float, h: float):
        self.w, self.h = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")
        self.r = self.fig.canvas.get_renderer()
        self.boxes: dict[str, Box] = {}
        self.texts: list = []          # (artist, container name or None)
        self.segments: list = []       # (p0, p1, endpoint box names)

    # ---------------------------------------------------------------- text
    def measure(self, s: str, size: float, weight: str = "normal") -> float:
        t = self.ax.text(0, 0, s, fontsize=size, fontweight=weight)
        bb = t.get_window_extent(self.r)
        t.remove()
        return bb.width / self.fig.dpi

    def block(self, lines: list[Line]) -> tuple[float, float]:
        wmax = max(self.measure(s, sz, wt) for s, sz, wt, _ in lines)
        h = sum(sz * LS / 72 for _, sz, _, _ in lines)
        return wmax, h

    def write(self, x, y, lines: list[Line], container=None, ha="center"):
        """Lines stacked and centred vertically on y."""
        _, h = self.block(lines)
        top = y + h / 2
        for s, sz, wt, col in lines:
            lh = sz * LS / 72
            t = self.ax.text(x, top - lh / 2, s, fontsize=sz, fontweight=wt, color=col,
                             ha=ha, va="center", zorder=6)
            self.texts.append((t, container))
            top -= lh
        return h

    # --------------------------------------------------------------- boxes
    def size_of(self, lines, padx=0.07, pady=0.05, minw=0.0, minh=0.0):
        bw, bh = self.block(lines)
        return max(bw + 2 * padx, minw), max(bh + 2 * pady, minh)

    def box(self, name, x0, y0, w, h, lines=None, style="neutral", kind="solid",
            lw=0.9, rounding=0.04, ls="-", text_at=None):
        fc, ec = PAL[style]
        p = FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                           fc=fc, ec=ec, lw=lw, ls=ls,
                           zorder=1 if kind == "region" else 3)
        self.ax.add_patch(p)
        b = Box(name, x0, y0, x0 + w, y0 + h, kind)
        assert name not in self.boxes, name
        self.boxes[name] = b
        if lines:
            tx, ty = text_at or (b.cx, b.cy)
            self.write(tx, ty, lines, container=name)
        return b

    def label(self, x, y, lines, container=None, ha="center"):
        return self.write(x, y, lines, container=container, ha=ha)

    # --------------------------------------------------------------- lines
    def path(self, pts, ends=(), arrow=False, colour=LINE, lw=0.9, ls="-", z=2):
        """A polyline; `ends` names the boxes it is allowed to touch."""
        for p0, p1 in zip(pts, pts[1:]):
            self.segments.append((p0, p1, set(ends)))
        xs, ys = zip(*pts)
        if not arrow:
            self.ax.plot(xs, ys, color=colour, lw=lw, ls=ls, zorder=z,
                         solid_capstyle="butt")
            return
        if len(pts) > 2:
            self.ax.plot(xs[:-1], ys[:-1], color=colour, lw=lw, ls=ls, zorder=z,
                         solid_capstyle="butt")
        self.ax.annotate("", xy=pts[-1], xytext=pts[-2],
                         arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22",
                                         color=colour, lw=lw, ls=ls,
                                         shrinkA=0, shrinkB=0, mutation_scale=10),
                         zorder=z)

    def firewall(self, name, cx, cy, w=0.26, h=0.15):
        """A brick wall: the conventional network-diagram firewall."""
        x0, y0 = cx - w / 2, cy - h / 2
        self.ax.add_patch(Rectangle((x0, y0), w, h, fc="#f3d0c4", ec="#a33b3b",
                                    lw=0.8, zorder=5))
        rows = 3
        for i in range(1, rows):
            y = y0 + i * h / rows
            self.ax.plot([x0, x0 + w], [y, y], color="#a33b3b", lw=0.5, zorder=5)
        for i in range(rows):
            ya, yb = y0 + i * h / rows, y0 + (i + 1) * h / rows
            offs = [0.33, 0.66] if i % 2 == 0 else [0.16, 0.5, 0.83]
            for f in offs:
                self.ax.plot([x0 + f * w] * 2, [ya, yb], color="#a33b3b", lw=0.5, zorder=5)
        self.boxes[name] = Box(name, x0, y0, x0 + w, y0 + h, "inline")
        return self.boxes[name]

    # --------------------------------------------------------------- check
    def verify(self, fig_name):
        dpi = self.fig.dpi
        errs = []
        solids = [b for b in self.boxes.values() if b.kind != "region"]

        def inside(a, b, tol=0.012):
            return (a[0] >= b.x0 + tol and a[2] <= b.x1 - tol
                    and a[1] >= b.y0 + tol / 2 and a[3] <= b.y1 - tol / 2)

        def hits(a, b, tol=0.004):
            return not (a[2] <= b.x0 + tol or a[0] >= b.x1 - tol
                        or a[3] <= b.y0 + tol or a[1] >= b.y1 - tol)

        for t, cont in self.texts:
            bb = t.get_window_extent(self.r)
            a = (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
            s = t.get_text()
            if t.get_fontsize() < MIN_PT:
                errs.append(f"{s!r} is {t.get_fontsize()} pt")
            if a[0] < 0.01 or a[1] < 0.01 or a[2] > self.w - 0.01 or a[3] > self.h - 0.01:
                errs.append(f"{s!r} leaves the canvas")
            if cont and not inside(a, self.boxes[cont]):
                errs.append(f"{s!r} overflows its box {cont!r}")
            for b in solids:
                if b.name != cont and hits(a, b):
                    errs.append(f"{s!r} collides with box {b.name!r}")
        regions = [b for b in self.boxes.values() if b.kind == "region"]
        for t, cont in self.texts:
            bb = t.get_window_extent(self.r)
            a = (bb.x0 / dpi, bb.y0 / dpi, bb.x1 / dpi, bb.y1 / dpi)
            for b in regions:
                if b.name != cont and hits(a, b) and not inside(a, b, tol=0.0):
                    errs.append(f"{t.get_text()!r} straddles the border of {b.name!r}")
        for i, a in enumerate(solids):
            for b in solids[i + 1:]:
                if hits((a.x0, a.y0, a.x1, a.y1), b, tol=0.0):
                    errs.append(f"boxes {a.name!r} and {b.name!r} overlap")
        for b in self.boxes.values():
            if b.x0 < 0 or b.y0 < 0 or b.x1 > self.w or b.y1 > self.h:
                errs.append(f"box {b.name!r} leaves the canvas")
        # A line may touch only the boxes it connects and the glyphs on it.
        for p0, p1, ends in self.segments:
            for b in solids:
                if b.name in ends or b.kind == "inline":
                    continue
                if seg_hits_box(p0, p1, b):
                    errs.append(f"a line crosses box {b.name!r}")
        for t, cont in self.texts:
            bb = t.get_window_extent(self.r)
            tb = Box("_t", bb.x0 / dpi - 0.01, bb.y0 / dpi - 0.005,
                     bb.x1 / dpi + 0.01, bb.y1 / dpi + 0.005)
            for p0, p1, _ in self.segments:
                if seg_hits_box(p0, p1, tb, tol=0.0):
                    errs.append(f"a line runs through the label {t.get_text()!r}")
        if errs:
            raise SystemExit(f"{fig_name}: layout check failed\n  " + "\n  ".join(errs))

    def save(self, name):
        self.verify(name)
        for ext in ("png", "pdf"):
            self.fig.savefig(OUT / f"{name}.{ext}", dpi=600 if ext == "png" else None,
                             facecolor="white")
        plt.close(self.fig)
        print(f"  {name:13} {self.w:.2f} x {self.h:.2f} in, "
              f"{len(self.texts)} labels, {len(self.boxes)} boxes, layout check passed")


def seg_hits_box(p0, p1, b, tol=0.01):
    """Does the segment pass through the interior of b (shrunk by tol)?"""
    x0, y0, x1, y1 = b.x0 + tol, b.y0 + tol, b.x1 - tol, b.y1 - tol
    if x0 >= x1 or y0 >= y1:
        return False
    # Liang-Barsky clipping
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - x0), (dx, x1 - p0[0]), (-dy, p0[1] - y0), (dy, y1 - p0[1])):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return t1 - t0 > 1e-6


def L(s, size=7.5, weight="normal", colour=INK) -> Line:
    return (s, size, weight, colour)


def title(s, size=8.0) -> Line:
    return (s, size, "bold", INK)


def sub(s, size=7.0) -> Line:
    return (s, size, "normal", SOFT)


def row_layout(c: Canvas, specs, y0, h, x_left, x_right, min_gap=0.22):
    """Widths from the text, the slack shared out as gaps; fails if it cannot fit."""
    widths = [c.size_of(lines)[0] for lines, _ in specs]
    slack = (x_right - x_left) - sum(widths)
    gap = slack / (len(specs) - 1)
    if gap < min_gap:
        raise SystemExit(f"row does not fit: needs {sum(widths) + min_gap * (len(specs) - 1):.2f} in, "
                         f"has {x_right - x_left:.2f}")
    boxes, x = [], x_left
    for (lines, style), w in zip(specs, widths):
        boxes.append(c.box(f"r{len(c.boxes)}", x, y0, w, h, lines, style))
        x += w + gap
    for a, b in zip(boxes, boxes[1:]):
        c.path([a.right(), b.left()], ends=(a.name, b.name), arrow=True)
    return boxes


def two_up(c: Canvas, y0, h, specs, notes, x_left=0.12, x_right=None, gap=0.3):
    """Two outcome boxes side by side with a result line under each."""
    x_right = x_right or c.w - 0.12
    w = (x_right - x_left - gap) / 2
    out = []
    for i, ((lines, style), note) in enumerate(zip(specs, notes)):
        x = x_left + i * (w + gap)
        b = c.box(f"o{i}", x, y0, w, h, lines, style, lw=1.1)
        out.append(b)
        if note:
            c.label(b.cx, y0 - 0.13, [note])
    return out


# =========================================================================== 1
def architecture():
    c = Canvas(W, 3.05)
    hb = 0.78
    top_y, bot_y = 1.95, 0.62
    specs = {
        "agent": ([title("Agent"), L("MaskablePPO;"), L("sees only"),
                   L("masked logits")], "neutral"),
        "env": ([title("Environment"), L("Gymnasium wrapper;"), L("computes the mask,"),
                 L("assigns the reward")], "blue"),
        "wm": ([title("World model"), L("45 techniques"), L("× 12 hosts;"),
                L("preconditions,"), L("effects, detection")], "blue"),
        "tel": ([title("Telemetry"), L("ATT&CK-labelled;"), L("31,925 events,"),
                 L("all 45 techniques")], "green"),
        "http": ([title("HTTP target"), L("serves the same model,"),
                  L("so tests transport,"), L("not transfer")], "blue"),
        "vuln": ([title("Vulnerable web app"), L("own code, data, logins;"),
                  L("15 techniques"), L("executed for real")], "red"),
    }
    wid = {k: c.size_of(v[0])[0] for k, v in specs.items()}
    # Top row: agent, environment, world model, telemetry.
    xl, xr = 0.08, W - 0.08
    order = ["agent", "env", "wm", "tel"]
    gap = (xr - xl - sum(wid[k] for k in order)) / 3
    assert gap >= 0.3, f"top row too wide, gap {gap:.2f}"
    x, B = xl, {}
    hw = max(c.size_of(specs["wm"][0])[1], c.size_of(specs["tel"][0])[1], hb)
    for k in order:
        lines, style = specs[k]
        B[k] = c.box(k, x, top_y, wid[k], hw, lines, style,
                     lw=1.6 if k == "wm" else 1.0)
        x += wid[k] + gap
    # Bottom row: HTTP target under the environment, the vulnerable app
    # under the world model.
    hh = max(c.size_of(specs["http"][0])[1], c.size_of(specs["vuln"][0])[1])
    B["http"] = c.box("http", B["env"].cx - wid["http"] / 2, bot_y, wid["http"], hh,
                      *specs["http"], lw=1.0)
    B["vuln"] = c.box("vuln", xr - wid["vuln"], bot_y, wid["vuln"], hh, *specs["vuln"],
                      lw=1.2)
    assert B["vuln"].x0 > B["http"].x1 + 0.8

    mid = top_y + hw / 2
    c.path([B["agent"].right(0.62), B["env"].left(0.62)], ("agent", "env"), arrow=True)
    c.path([B["env"].left(0.38), B["agent"].right(0.38)], ("agent", "env"), arrow=True)
    c.label((B["agent"].x1 + B["env"].x0) / 2, top_y + hw + 0.12,
            [sub("action; observation, reward")])
    c.path([B["env"].right(), B["wm"].left()], ("env", "wm"), arrow=True)
    c.label((B["env"].x1 + B["wm"].x0) / 2, mid + 0.13, [sub("steps")])
    c.path([B["wm"].right(), B["tel"].left()], ("wm", "tel"), arrow=True,
           colour=PAL["green"][1])
    c.label((B["wm"].x1 + B["tel"].x0) / 2, mid + 0.13, [sub("emits")])
    # The environment can forward over HTTP; the server imports the model.
    c.path([B["env"].bottom(), B["http"].top()], ("env", "http"), arrow=True)
    c.label(B["env"].cx + 0.06, (top_y + bot_y + hh) / 2, [sub("or over HTTP")], ha="left")
    fx = B["wm"].x0 + 0.18
    c.path([B["http"].right(0.7), (fx, B["http"].y0 + 0.7 * hh), B["wm"].bottom(
        (fx - B["wm"].x0) / (B["wm"].x1 - B["wm"].x0))], ("http", "wm"), arrow=True)
    c.label(fx - 0.06, B["http"].y1 + 0.2, [sub("imports")], ha="right")
    # The real arm decides a technique's outcome by executing it.
    sx = B["wm"].x1 - 0.22
    ymid = (B["wm"].y0 + B["vuln"].y1) / 2
    c.path([B["wm"].bottom((sx - B["wm"].x0) / (B["wm"].x1 - B["wm"].x0)), (sx, ymid),
            (B["vuln"].cx, ymid), B["vuln"].top()],
           ("wm", "vuln"), arrow=True, colour=PAL["red"][1], ls=(0, (3, 2)))
    c.label((sx + B["vuln"].cx) / 2, ymid + 0.11,
            [("real arm", 7, "normal", PAL["red"][1])])
    # Legend.
    for i, (style, text) in enumerate(
            [("blue", "the world model and what imports it: cannot disagree with it"),
             ("red", "shares no code with it: can disagree, and does")]):
        y = 0.34 - i * 0.18
        fc, ec = PAL[style]
        c.ax.add_patch(Rectangle((0.1, y - 0.06), 0.16, 0.12, fc=fc, ec=ec, lw=0.9))
        c.label(0.32, y, [sub(text)], ha="left")
    c.save("architecture")


# =========================================================================== 2
def topology():
    topo = Topology("enterprise")
    S = topo.subnets
    assert sorted(S) == [1, 2, 3, 4, 5]
    edges = {(a, b) for a in range(6) for b in range(6) if a < b and topo.adjacent(a, b)}
    # Drawn for exactly this reachability rule. If the estate changes, the
    # figure must be redrawn rather than left silently wrong.
    assert edges == {(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 4)}, edges
    NAME = {1: "DMZ", 2: "Workstations", 3: "Servers", 4: "Admin tier", 5: "Domain core"}

    hw, hh = 0.74, 0.28            # host box
    HDR, SWH, SWGAP, VGAP, BOT = 0.18, 0.13, 0.12, 0.05, 0.06
    row_h = HDR + SWH + SWGAP + hh + BOT                       # side-by-side subnet
    stack_h = lambda n: HDR + SWH + 0.08 + n * hh + (n - 1) * VGAP + BOT
    INET, FWGAP, ROWGAP = 0.42, 0.20, 0.40
    H = (0.06 + INET + 2 * FWGAP + row_h + 2 * FWGAP + 0.18 + row_h + ROWGAP
         + stack_h(max(len(S[3]), len(S[4]), len(S[5]))) + 0.12)
    c = Canvas(W, round(H, 2))
    osline = {"windows": "Windows", "linux": "Linux"}
    hstyle = {"windows": "blue", "linux": "tan"}

    def host(name, x0, y0):
        lines = [L(name + (" ★" if name == topo.goal else ""), 7.5, "bold"),
                 sub(osline[topo.os_of(name)])]
        return c.box(name, x0, y0, hw, hh, lines, hstyle[topo.os_of(name)],
                     lw=1.5 if name == topo.goal else 0.8)

    def header(sid, x0, y_top):
        c.label(x0 + 0.08, y_top - HDR / 2 - 0.01,
                [(f"Subnet {sid} · {NAME[sid]}", 7.0, "bold", INK)],
                container=f"s{sid}", ha="left")

    def side_by_side(sid, cx, y_top, extra=()):
        hosts = list(extra) + S[sid]
        w = 3.5
        inner = w - 0.24
        step_x = (inner - hw) / (len(hosts) - 1)
        x0 = cx - w / 2
        reg = c.box(f"s{sid}", x0, y_top - row_h, w, row_h, style="region", kind="region")
        header(sid, x0, y_top)
        sw = c.box(f"sw{sid}", x0 + 0.12, y_top - HDR - SWH, inner, SWH,
                   [("switch", 7, "normal", SOFT)], "device", rounding=0.02, lw=0.7)
        for i, hname in enumerate(hosts):
            hx = x0 + 0.12 + i * step_x
            if hname == "WAF":
                b = c.box("waf", hx, sw.y0 - SWGAP - hh, hw, hh,
                          [L("WAF", 7.5, "bold"), sub("log source")], "device", lw=0.8)
            else:
                b = host(hname, hx, sw.y0 - SWGAP - hh)
            c.path([(b.cx, sw.y0), b.top()], ends=(sw.name, b.name), lw=0.7)
        return reg, sw

    def stacked(sid, x0, w, y_top):
        n = len(S[sid])
        reg = c.box(f"s{sid}", x0, y_top - stack_h(n), w, stack_h(n), style="region",
                    kind="region")
        header(sid, x0, y_top)
        sw = c.box(f"sw{sid}", x0 + 0.1, y_top - HDR - SWH, w - 0.2, SWH,
                   [("switch", 7, "normal", SOFT)], "device", rounding=0.02, lw=0.7)
        bus_x = x0 + 0.22
        y = sw.y0 - 0.08
        last = None
        for hname in S[sid]:
            b = host(hname, bus_x + 0.14, y - hh)
            c.path([(bus_x, b.cy), b.left()], ends=(b.name,), lw=0.7)
            last = b
            y -= hh + VGAP
        c.path([(bus_x, sw.y0), (bus_x, last.cy)], ends=(sw.name,), lw=0.7)
        return reg, sw

    # Internet, perimeter firewall, DMZ.
    inet = c.box("inet", W / 2 - 0.95, c.h - 0.06 - INET, 1.9, INET,
                 [L("Internet", 8, "bold"), sub("the attacker starts here")], "red",
                 rounding=0.2, lw=1.0)
    fw1_y = inet.y0 - FWGAP
    dmz, sw1 = side_by_side(1, W / 2, fw1_y - FWGAP, extra=("WAF",))
    c.firewall("fw1", W / 2, fw1_y)
    c.path([inet.bottom(), (W / 2, sw1.y1)], ends=("inet", "sw1", "fw1"))
    c.label(W / 2 + 0.2, fw1_y, [sub("perimeter firewall")], ha="left")
    # Internal firewall, then the internal network.
    fw2_y = dmz.y0 - FWGAP
    internal_top = fw2_y - 0.12
    ws, sw2 = side_by_side(2, W / 2, fw2_y - FWGAP - 0.18)
    c.firewall("fw2", W / 2, fw2_y)
    c.path([(W / 2, dmz.y0), (W / 2, sw2.y1)], ends=("sw2", "fw2"))
    c.label(W / 2 + 0.2, fw2_y, [sub("internal firewall")], ha="left")

    bw = 1.62
    gap = (W - 0.2 - 3 * bw) / 2
    assert gap >= 0.36, gap
    xs = [0.1, 0.1 + bw + gap, 0.1 + 2 * (bw + gap)]
    row_top = ws.y0 - ROWGAP
    srv, _ = stacked(3, xs[0], bw, row_top)
    adm, _ = stacked(4, xs[1], bw, row_top)
    core, _ = stacked(5, xs[2], bw, row_top)
    assert abs(adm.cx - W / 2) < 1e-6

    # One firewall on every link the environment permits between subnets.
    y_link = row_top - HDR - SWH - 0.08 - hh / 2
    at = lambda b: 1 - (b.y1 - y_link) / (b.y1 - b.y0)
    c.path([srv.right(at(srv)), adm.left(at(adm))], ends=("s3", "s4"))
    c.firewall("fw34", (srv.x1 + adm.x0) / 2, y_link)
    c.path([adm.right(at(adm)), core.left(at(core))], ends=("s4", "s5"))
    c.firewall("fw45", (adm.x1 + core.x0) / 2, y_link)
    c.path([ws.left(0.5), (srv.cx, ws.cy), srv.top()], ends=("s2", "s3"))
    c.firewall("fw23", srv.cx, (ws.y0 + srv.y1) / 2)
    c.path([ws.bottom(), adm.top()], ends=("s2", "s4"))
    c.firewall("fw24", W / 2, (ws.y0 + adm.y1) / 2)

    in_y0 = min(srv.y0, adm.y0, core.y0) - 0.06
    c.ax.add_patch(FancyBboxPatch((0.04, in_y0), W - 0.08, internal_top - in_y0,
                                  boxstyle="round,pad=0,rounding_size=0.06", fc="none",
                                  ec="#9ca3af", lw=0.9, ls=(0, (4, 2)), zorder=0))
    c.boxes["internal"] = Box("internal", 0.04, in_y0, W - 0.04, internal_top, "region")
    c.label(0.12, internal_top - 0.11, [L("Internal network", 7.5, "bold", SOFT)],
            container="internal", ha="left")

    # Legend, top left, clear of the DMZ.
    lx, ly = 0.1, c.h - 0.12
    for i, (style, text) in enumerate([("blue", "Windows host"), ("tan", "Linux host"),
                                       ("device", "switch")]):
        fc, ec = PAL[style]
        y = ly - i * 0.16
        c.ax.add_patch(Rectangle((lx, y - 0.05), 0.16, 0.10, fc=fc, ec=ec, lw=0.8))
        c.label(lx + 0.22, y, [sub(text)], ha="left")
    c.firewall("fw_legend", lx + 0.08, ly - 3 * 0.16, w=0.16, h=0.10)
    c.label(lx + 0.22, ly - 3 * 0.16, [sub("firewall")], ha="left")
    c.label(lx + 0.02, ly - 4 * 0.16, [sub("★  objective")], ha="left")
    assert lx + 1.0 < dmz.x0, "legend runs into the DMZ"
    c.save("topology")


# =========================================================================== 3
def mask():
    c = Canvas(W, 2.36)
    h = 0.62
    y0 = c.h - 0.12 - h
    specs = [
        ([L(r"state $s$ and"), L(r"scenario $\omega$")], "neutral"),
        ([L("check preconditions"), L("of all 540 actions:"),
          sub("45 techniques × 12 hosts")], "blue"),
        ([L(r"mask $\mu(s,\omega)$:"), L("one bit per action")], "blue"),
        ([L("masked logits"), L(r"set to $-\infty$")], "blue"),
        ([L("legal action"), L("sampled")], "green"),
    ]
    row = row_layout(c, specs, y0, h, 0.06, W - 0.06, min_gap=0.2)
    # "computed every step" spans the first three boxes.
    by = y0 - 0.08
    c.path([(row[1].x0, by), (row[2].x1, by)], lw=0.7)
    c.label((row[1].x0 + row[2].x1) / 2, by - 0.12,
            [("computed by the environment at every step; never learned", 7, "normal", SOFT)])
    c.ax.plot([0.12, W - 0.12], [by - 0.32] * 2, color="#d1d5db", lw=0.8)
    c.label(0.12, by - 0.47, [title("An action whose preconditions do not hold")], ha="left")
    oh = 0.55
    two_up(c, 0.36, oh,
           [([L("under the prerequisite mask"), L("receives no probability mass"),
              L("and cannot be proposed at all")], "green"),
            ([L("under reward shaping"), L("is proposed, executed, refused,"),
              L("and penalised afterwards")], "red")],
           [("0.0% of actions infeasible", 7.5, "bold", PAL["green"][1]),
            ("10.3% infeasible after training, from 99%", 7.5, "bold", PAL["red"][1])])
    c.save("mask")


# =========================================================================== 4
def detection():
    c = Canvas(W, 2.10)
    h = 0.62
    y0 = c.h - 0.12 - h
    specs = [
        ([L("technique"), L("executes"), L("on a host")], "neutral"),
        ([L("emits the"), L("records it would"), L("really leave")], "blue"),
        ([L("a named rule"), L("matches the"), L("event identifier")], "blue"),
        ([L("suspicion"), L("accumulates")], "blue"),
        ([L("incident"), L("declared; the"), L("engagement ends")], "red"),
    ]
    row_layout(c, specs, y0, h, 0.06, W - 0.06, min_gap=0.2)
    c.ax.plot([0.12, W - 0.12], [y0 - 0.2] * 2, color="#d1d5db", lw=0.8)
    c.label(0.12, y0 - 0.36,
            [title("Nine log channels; each platform emits only its own records")], ha="left")
    bh = 0.72
    specs2 = [
        ([L("Windows hosts", 7.5, "bold"), L("Security, System, Sysmon,"),
          L("PowerShell, TerminalServices,"), L("WMI-Activity")], "blue"),
        ([L("Linux hosts", 7.5, "bold"), L("auditd, syslog"),
          L("no Windows event identifiers")], "tan"),
        ([L("Web front end", 7.5, "bold"), L("WAF")], "device"),
    ]
    widths = [c.size_of(s[0])[0] for s in specs2]
    gap = (W - 0.24 - sum(widths)) / 2
    assert gap > 0.15
    x = 0.12
    for i, ((lines, style), w) in enumerate(zip(specs2, widths)):
        c.box(f"p{i}", x, 0.12, w, bh, lines, style, lw=1.0)
        x += w + gap
    c.save("detection")


# =========================================================================== 5
def scenario():
    c = Canvas(W, 2.12)
    h = 0.62
    y0 = c.h - 0.12 - h
    specs = [
        ([L("engagement drawn"), L("at episode start")], "neutral"),
        ([L("one technique withdrawn"), L("from each substitution"),
          L("group; requirements set")], "blue"),
        ([L(r"scenario $\omega$ fixed"), L("for the episode")], "blue"),
        ([L("mask becomes"), L(r"$\mu(s,\omega)$, not $\mu(s)$")], "green"),
    ]
    row_layout(c, specs, y0, h, 0.08, W - 0.08, min_gap=0.25)
    c.ax.plot([0.12, W - 0.12], [y0 - 0.2] * 2, color="#d1d5db", lw=0.8)
    c.label(0.12, y0 - 0.36, [title("Why this decides whether masking matters")], ha="left")
    two_up(c, 0.36, 0.50,
           [([L(r"legality a fixed function of state, $\mu(s)$"),
              L("an agent can learn it from experience")], "red"),
            ([L(r"legality depends on the engagement, $\mu(s,\omega)$"),
              L("there is nothing stable to learn")], "green")],
           [("masked 99.7%, unmasked 99.8%", 7.5, "bold", PAL["red"][1]),
            ("masked 69.7%, unmasked 0.0%", 7.5, "bold", PAL["green"][1])])
    c.save("scenario")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (architecture, topology, mask, detection, scenario):
        f()
