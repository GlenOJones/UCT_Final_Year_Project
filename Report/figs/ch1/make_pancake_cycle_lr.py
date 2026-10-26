import base64, math, random, sys, urllib.parse, zlib
from xml.sax.saxutils import escape

INK, NEUTRAL, WHITE, ACCENT = "#333333", "#F5F5F5", "#FFFFFF", "#1A73E8"
SW, SH = 460, 200
X0, Y0 = 80, 135
BW, BH = 220, 45
OX, OY = 50, -100
FONT = 34
PITCH = 510


def P(u, z, lift=0.0):
    return X0 + u + z * OX, Y0 + z * OY - lift


def wave(u, phase):
    return 1.6 * math.sin(u / 17 + phase) + 0.8 * math.sin(u / 7 + 2 * phase)


def poly(pts, op):
    d = f'<move x="{pts[0][0]:.1f}" y="{pts[0][1]:.1f}"/>'
    d += "".join(f'<line x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts[1:])
    return f"<path>{d}<close/></path><{op}/>"


def fill(c):
    return f'<fillcolor color="{c}"/>'


def line(a, b):
    return f'<path><move x="{a[0]:.1f}" y="{a[1]:.1f}"/><line x="{b[0]:.1f}" y="{b[1]:.1f}"/></path><stroke/>'


def text(s, x, y, align):
    return f'<text str="{s}" x="{x:.1f}" y="{y:.1f}" align="{align}" valign="middle"/>'


def chaikin(pts, n):
    for _ in range(n):
        out = []
        for i, (x0, y0) in enumerate(pts):
            x1, y1 = pts[(i + 1) % len(pts)]
            out += [(0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1), (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)]
        pts = out
    return pts


def floe(rnd, u, z, d, lift=0, thick=8, rough=0.3, smooth=2):
    d *= BW * 0.95
    k = rnd.randint(7, 10)
    stretch, rot = rnd.uniform(0.85, 1.2), rnd.uniform(-0.3, 0.3)
    shape = []
    for i in range(k):
        a = 2 * math.pi * (i + rnd.uniform(-0.3, 0.3)) / k
        r = d / 2 * rnd.uniform(1 - rough, 1 + rough * 0.4)
        x, y = r * math.cos(a) * stretch, r * math.sin(a) / stretch
        x, y = x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)
        shape.append((x, y * 0.45))
    shape = chaikin(shape, smooth)
    hx = max(abs(x) for x, _ in shape) + 3
    hz = (max(abs(y) for _, y in shape) + 2) / -OY
    u = min(max(u * BW, hx), BW - hx)
    z = min(max(z, hz * 0.4), 1 - hz - lift / -OY)
    cx, cy = P(u, z, lift)
    top = [(cx + x, cy + y) for x, y in shape]
    low = [(x, y + thick) for x, y in top]
    out = fill(NEUTRAL) + poly(low, "fillstroke")
    for i in range(len(top)):
        j = (i + 1) % len(top)
        out += poly([top[i], top[j], low[j], low[i]], "fill")
    for x, y in (min(top), max(top)):
        out += line((x, y), (x, y + thick))
    out += fill(WHITE) + poly(top, "fillstroke")
    return out


def slab(rnd, u0, u1, depth, h=8, ang=0.0):
    x, y, w = X0 + u0 * BW, Y0 + depth, (u1 - u0) * BW
    c, s = math.cos(ang), math.sin(ang)
    top = [(w * i / 6, rnd.uniform(-0.8, 0.8)) for i in range(7)]
    bot = [(w * i / 6, h + rnd.uniform(-0.8, 0.8)) for i in range(6, -1, -1)]
    pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in top + bot]
    return fill(WHITE) + poly(pts, "fillstroke")


def dots(rnd, n, region):
    out = ""
    for _ in range(n):
        x, y = region(rnd)
        out += f'<ellipse x="{x:.1f}" y="{y:.1f}" w="3" h="3"/><fill/>'
    return out


def on_top(rnd):
    return P(rnd.uniform(3, BW - 3), rnd.uniform(0.04, 0.96))


def under_surface(rnd):
    u = rnd.uniform(3, BW - 3)
    return X0 + u, Y0 + wave(u, 0) + 2.5 + abs(rnd.gauss(0, 2.5))


FLOES = {
    2: [(0.38, 0.88, 0.40, 0), (0.86, 0.86, 0.30, 0), (0.55, 0.42, 0.36, 0),
        (0.22, 0.10, 0.24, 0, 10), (0.75, 0.12, 0.25, 0, 10)],
    3: [(0.42, 0.80, 0.42, 5), (0.80, 0.58, 0.32, 0), (0.70, 0.62, 0.38, 3),
        (0.58, 0.40, 0.34, 0), (0.62, 0.20, 0.34, 0, 10), (0.32, 0.12, 0.46, 3, 12)],
    4: [(0.32, 0.88, 0.18, 0), (0.62, 0.80, 0.42, 0), (0.86, 0.70, 0.28, 3), (0.74, 0.58, 0.26, 0),
        (0.50, 0.55, 0.22, 0), (0.30, 0.34, 0.36, 12, 10), (0.56, 0.30, 0.30, 10, 10),
        (0.22, 0.10, 0.30, 2, 10), (0.50, 0.10, 0.30, 2, 10)],
}
SLABS = {
    3: [(0.30, 0.66, 12)],
    4: [(0.12, 0.42, 10), (0.30, 0.55, 20), (0.20, 0.38, 30)],
}
SCALES = {1: ("0 10 20 30", "cm", "5cm"), 2: ("0 25 50", "cm", "20cm"),
          3: ("0 50 100", "cm", "20cm"), 4: ("0 1 2 3", "m", "50cm")}


def block(stage, seed):
    rnd = random.Random(seed)
    ph = rnd.uniform(0, 6)
    front = [(X0 + u, Y0 + wave(u, ph)) for u in range(0, BW + 1, 3)]
    back = [(x + OX, y + OY) for x, y in front]
    out = f'<strokecolor color="{INK}"/><strokewidth width="2"/><fontcolor color="{INK}"/>'
    out += f'<fontfamily family="Helvetica"/><fontsize size="{FONT}"/>'
    out += fill(NEUTRAL) + poly(front + [(X0 + BW, Y0 + BH), (X0, Y0 + BH)], "fillstroke")
    out += poly([front[-1], back[-1], (back[-1][0], back[-1][1] + BH), (X0 + BW, Y0 + BH)], "fillstroke")
    out += fill(WHITE) + poly(front + back[::-1], "fillstroke")
    out += fill(INK)

    if stage == 1:
        out += dots(rnd, 150, on_top) + dots(rnd, 70, under_surface)
    elif stage == 2:
        out += dots(rnd, 90, on_top) + dots(rnd, 50, under_surface)
    else:
        out += dots(rnd, 30, under_surface)
    for u0, u1, depth in SLABS.get(stage, []):
        out += slab(rnd, u0, u1, depth)
    if stage == 4:
        out += slab(rnd, 0.58, 0.80, 22, ang=-0.3)
    for f in sorted(FLOES.get(stage, []), key=lambda f: (-f[1], f[3])):
        u, z, d, lift = f[:4]
        thick = f[4] if len(f) > 4 else 8
        out += floe(rnd, u, z, d, lift, thick, 0.15 if stage == 2 else 0.3, 3 if stage == 2 else 2)

    ticks, unit, depth = SCALES[stage]
    ticks = ticks.split()
    a = (X0 - 10, Y0)
    out += line(a, (a[0] + OX, a[1] + OY))
    for i, t in enumerate(ticks):
        f = i / (len(ticks) - 1)
        tx, ty = a[0] + f * OX, a[1] + f * OY
        out += line((tx - 8, ty), (tx, ty)) + text(t, tx - 12, ty, "right")
    out += text(unit, X0 - 35, Y0 + 32, "center")

    rx, ry = X0 + BW + OX + 10, Y0 + 0.5 * OY
    out += line((rx, ry), (rx, ry + BH)) + line((rx, ry), (rx + 8, ry)) + line((rx, ry + BH), (rx + 8, ry + BH))
    out += text("0", rx + 12, ry, "left") + text(depth, rx + 12, ry + BH, "left")

    xml = (f'<shape name="pancakecyclelr{stage}" w="{SW}" h="{SH}" aspect="variable" strokewidth="inherit">'
           f'<connections/><background/><foreground>{out}</foreground></shape>')
    enc = urllib.parse.quote(xml, safe="~()*!.'-_")
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.b64encode(co.compress(enc.encode()) + co.flush()).decode()


LABELS = [
    "1. New ice formation:<br>frazil ice",
    "2. Formation of<br>pancake ice",
    "3. Freezing together and<br>rafting of pancake ice",
    "4. Further rafting and<br>new ice formation",
]

cells = []
for i, label in enumerate(LABELS):
    x0 = i * PITCH
    cells.append(
        f'<mxCell id="t{i+1}" value="{escape(label)}" style="text;html=1;whiteSpace=wrap;align=center;'
        f'verticalAlign=middle;fontFamily=Helvetica;fontSize={FONT};fontColor={INK};" vertex="1" parent="1">'
        f'<mxGeometry x="{x0}" y="0" width="{SW}" height="80" as="geometry"/></mxCell>')
    cells.append(
        f'<mxCell id="b{i+1}" value="" style="shape=stencil({block(i + 1, 4022 + i)});'
        f'strokeColor={INK};fillColor={INK};" vertex="1" parent="1">'
        f'<mxGeometry x="{x0}" y="80" width="{SW}" height="{SH}" as="geometry"/></mxCell>')
for i in range(3):
    cells.append(
        f'<mxCell id="e{i+1}" value="" style="edgeStyle=orthogonalEdgeStyle;html=1;strokeWidth=2;'
        f'strokeColor={ACCENT};endArrow=block;endFill=1;exitX=1;exitY=0.54;entryX=0;entryY=0.54;" '
        f'edge="1" parent="1" source="b{i+1}" target="b{i+2}"><mxGeometry relative="1" as="geometry"/></mxCell>')

out = ('<mxfile><diagram id="pancakecyclelr" name="Pancake cycle (left to right)"><mxGraphModel adaptiveColors="auto" '
       'grid="1" gridSize="10"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
open(sys.argv[1], "w").write(out)
