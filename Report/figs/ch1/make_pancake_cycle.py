import base64, math, random, sys, urllib.parse, zlib
from xml.sax.saxutils import escape

INK, NEUTRAL, WHITE = "#333333", "#F5F5F5", "#FFFFFF"
SW, SH = 520, 170
X0, Y0 = 60, 95
BW, BH = 300, 45
OX, OY = 90, -70


def P(u, z, lift=0.0):
    return X0 + u + z * OX, Y0 + z * OY - lift


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


def floe(rnd, u, z, d, lift, thick, rough, rim, smooth=2):
    cx, cy = P(u, z, lift)
    k = rnd.randint(7, 11)
    stretch, rot = rnd.uniform(0.75, 1.25), rnd.uniform(0, math.pi)
    shape = []
    for i in range(k):
        a = 2 * math.pi * (i + rnd.uniform(-0.3, 0.3)) / k
        r = d / 2 * rnd.uniform(1 - rough, 1 + rough * 0.5)
        x, y = r * math.cos(a) * stretch, r * math.sin(a) / stretch
        x, y = x * math.cos(rot) - y * math.sin(rot), x * math.sin(rot) + y * math.cos(rot)
        shape.append((x, y * 0.4))
    shape = chaikin(shape, smooth)
    top = [(cx + x, cy + y) for x, y in shape]
    low = [(x, y + thick) for x, y in top]
    out = fill(NEUTRAL) + poly(low, "fillstroke")
    for i in range(len(top)):
        j = (i + 1) % len(top)
        out += poly([top[i], top[j], low[j], low[i]], "fill")
    for x, y in (min(top), max(top)):
        out += line((x, y), (x, y + thick))
    out += fill(WHITE) + poly(top, "fillstroke")
    if rim:
        out += poly([(cx + 0.78 * x, cy + 0.78 * y) for x, y in shape], "stroke")
    return out


def slab(x, y, w, h, ang=0.0, rnd=None):
    rnd = rnd or random.Random(int(x * 7 + y))
    c, s = math.cos(ang), math.sin(ang)
    top = [(w * i / 6, rnd.uniform(-1.2, 1.2)) for i in range(7)]
    bot = [(w * i / 6 + rnd.uniform(-2, 2) * (i in (0, 6)), h + rnd.uniform(-1, 1)) for i in range(6, -1, -1)]
    pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in top + bot]
    return fill(WHITE) + poly(pts, "fillstroke")


def frazil(rnd, n, depth_band=False):
    out = ""
    for _ in range(n):
        x, y = P(rnd.uniform(4, BW - 4), rnd.uniform(0.03, 0.97))
        out += f'<ellipse x="{x:.1f}" y="{y:.1f}" w="1.6" h="1.6"/><fill/>'
    if depth_band:
        for _ in range(n // 3):
            x = X0 + rnd.uniform(3, BW - 3)
            y = Y0 + 2 + abs(rnd.gauss(0, 3))
            out += f'<ellipse x="{x:.1f}" y="{y:.1f}" w="1.6" h="1.6"/><fill/>'
    return out


def place(rnd, n, dmin, dmax, overlap, lift=0):
    placed = []
    for _ in range(4000):
        if len(placed) == n:
            break
        d = rnd.uniform(dmin, dmax)
        zh = d * 0.2 / -OY + 0.04
        u, z = rnd.uniform(d / 2 + 4, BW - d / 2 - 4), rnd.uniform(zh, 1 - zh - (lift + 5) / -OY)
        clash = any(abs(u - pu) < (d + pd) / 2 * overlap and abs(z - pz) < (zh + pzh) * overlap
                    for pu, pz, pd, pzh in placed)
        if not clash:
            placed.append((u, z, d, zh))
    return sorted(placed, key=lambda p: -p[1])


def block(stage, seed):
    rnd = random.Random(seed)
    out = f'<strokecolor color="{INK}"/><strokewidth width="1"/><fontcolor color="{INK}"/>'
    out += '<fontfamily family="Helvetica"/><fontsize size="14"/>'
    out += fill(NEUTRAL) + poly([P(0, 0), P(BW, 0), (P(BW, 0)[0], Y0 + BH), (X0, Y0 + BH)], "fillstroke")
    out += poly([P(BW, 0), P(BW, 1), (P(BW, 1)[0], P(BW, 1)[1] + BH), (P(BW, 0)[0], Y0 + BH)], "fillstroke")
    out += fill(WHITE) + poly([P(0, 0), P(BW, 0), P(BW, 1), P(0, 1)], "fillstroke")
    out += fill(INK)

    if stage == 1:
        out += frazil(rnd, 160, depth_band=True)
    elif stage == 2:
        out += frazil(rnd, 90, depth_band=True)
        for u, z, d, _ in place(rnd, 5, 50, 75, 1.15):
            out += floe(rnd, u, z, d, 0, 4, 0.25, True, 3)
        out += slab(X0 + 40, Y0 - 2, 40, 5) + slab(X0 + 200, Y0 - 2, 55, 5)
    elif stage == 3:
        out += frazil(rnd, 40)
        for i, (u, z, d, _) in enumerate(place(rnd, 8, 50, 80, 0.6, 6)):
            out += floe(rnd, u, z, d, 6 * (i % 2), 5, 0.4, False, 2)
        out += slab(X0 + 50, Y0 - 3, 90, 6) + slab(X0 + 140, Y0 - 3, 70, 6)
        out += slab(X0 + 105, Y0 + 7, 75, 6) + slab(X0 + 190, Y0 - 9, 60, 6)
    else:
        out += frazil(rnd, 40)
        for i, (u, z, d, _) in enumerate(place(rnd, 16, 35, 80, 0.4, 14)):
            out += floe(rnd, u, z, d, 7 * (i % 3), 5, 0.5, False, 1)
        out += slab(X0 + 25, Y0 - 3, 85, 6) + slab(X0 + 110, Y0 - 9, 75, 6) + slab(X0 + 185, Y0 - 3, 90, 6)
        out += slab(X0 + 60, Y0 + 5, 70, 6) + slab(X0 + 150, Y0 + 9, 60, 6)
        out += slab(X0 + 225, Y0 + 4, 55, 6, 0.35)

    ticks, unit, depth = {1: ("0 10 20 30", "cm", "5cm"), 2: ("0 25 50", "cm", "20cm"),
                          3: ("0 50 100", "cm", "20cm"), 4: ("0 1 2 3", "m", "50cm")}[stage]
    ticks = ticks.split()
    a, b = (X0 - 10, Y0), (X0 - 10 + OX, Y0 + OY)
    out += line(a, b)
    for i, t in enumerate(ticks):
        f = i / (len(ticks) - 1)
        tx, ty = a[0] + f * OX, a[1] + f * OY
        out += line((tx - 6, ty), (tx, ty)) + text(t, tx - 9, ty, "right")
    out += text(unit, X0 - 25, Y0 + 18, "center")

    rx = P(BW, 1)[0] + 10
    ry = Y0 + 0.5 * OY
    out += line((rx, ry), (rx, ry + BH)) + line((rx, ry), (rx + 6, ry))
    out += line((rx, ry + BH), (rx + 6, ry + BH))
    out += text("0", rx + 10, ry, "left") + text(depth, rx + 10, ry + BH, "left")

    xml = (f'<shape name="pancakecycle{stage}" w="{SW}" h="{SH}" aspect="variable" strokewidth="inherit">'
           f'<connections/><background/><foreground>{out}</foreground></shape>')
    enc = urllib.parse.quote(xml, safe="~()*!.'-_")
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.b64encode(co.compress(enc.encode()) + co.flush()).decode()


LABELS = [
    "New ice formation: frazil ice",
    "Formation of pancake ice",
    "Freezing together and rafting of pancake ice",
    "Further rafting and new ice formation",
]

cells = []
for i, label in enumerate(LABELS):
    y0 = i * 180
    cells.append(
        f'<mxCell id="b{i+1}" value="" style="shape=stencil({block(i + 1, 4022 + i)});'
        f'strokeColor={INK};fillColor={INK};" vertex="1" parent="1">'
        f'<mxGeometry x="0" y="{y0}" width="{SW}" height="{SH}" as="geometry"/></mxCell>')
    cells.append(
        f'<mxCell id="t{i+1}" value="{escape(label)}" style="text;html=1;whiteSpace=wrap;align=left;'
        f'verticalAlign=middle;fontFamily=Helvetica;fontSize=14;fontColor={INK};" vertex="1" parent="1">'
        f'<mxGeometry x="530" y="{y0 + 50}" width="200" height="60" as="geometry"/></mxCell>')

out = ('<mxfile><diagram id="pancakecycle" name="Pancake cycle"><mxGraphModel adaptiveColors="auto" '
       'grid="1" gridSize="10"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
open(sys.argv[1], "w").write(out)
