import base64, math, random, sys, urllib.parse, zlib
from xml.sax.saxutils import escape

W, H = 640, 130
INK = "#333333"


def surface(x, ripple_end):
    y = 65 + 30 * math.sin(2 * math.pi * (x - 210) / 480)
    if x < ripple_end:
        taper = math.cos(0.5 * math.pi * x / ripple_end) ** 0.5
        y += 5 * taper * math.sin(2 * math.pi * x / 36)
    return y


def slope(x, ripple_end):
    return math.atan2(surface(x + 0.5, ripple_end) - surface(x - 0.5, ripple_end), 1)


def lens(cx, cy, ang, length, thick, n=6):
    tx, ty = math.cos(ang), math.sin(ang)
    nx, ny = -ty, tx
    pts = []
    for i in range(n + 1):
        s = -1 + 2 * i / n
        h = thick / 2 * (1 - s * s) ** 0.5
        pts.append((cx + s * length / 2 * tx - h * nx, cy + s * length / 2 * ty - h * ny))
    for i in range(n - 1, 0, -1):
        s = -1 + 2 * i / n
        h = thick / 2 * (1 - s * s) ** 0.5
        pts.append((cx + s * length / 2 * tx + h * nx, cy + s * length / 2 * ty + h * ny))
    d = f'<move x="{pts[0][0]:.1f}" y="{pts[0][1]:.1f}"/>'
    d += "".join(f'<line x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts[1:])
    return f"<path>{d}<close/></path><fill/>"


def bent(cx, off, length, thick, ripple_end, n=10):
    top, bot = [], []
    for i in range(n + 1):
        u = -1 + 2 * i / n
        x = cx + u * length / 2
        yc = surface(x, ripple_end) + off
        h = thick / 2 * (1 - u * u) ** 0.35
        top.append((x, yc - h))
        bot.append((x, yc + h))
    pts = top + bot[::-1][1:-1]
    d = f'<move x="{pts[0][0]:.1f}" y="{pts[0][1]:.1f}"/>'
    d += "".join(f'<line x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts[1:])
    return f"<path>{d}<close/></path><fill/>"


def panel(stage, seed):
    rnd = random.Random(seed)
    ripple_end = {1: 300, 2: 270, 3: 0, 4: 0}[stage]
    ice_start = {1: 290, 2: 260, 3: 10, 4: 150}[stage]
    els = []

    pts = [(x, surface(x, ripple_end)) for x in range(10, 631, 3)]
    d = f'<move x="{pts[0][0]}" y="{pts[0][1]:.1f}"/>'
    d += "".join(f'<line x="{x}" y="{y:.1f}"/>' for x, y in pts[1:])
    els.append(f'<strokewidth width="1.5"/><path>{d}</path><stroke/>')

    def discs(count, depth, x0=ice_start, x1=625):
        for _ in range(count):
            x = rnd.uniform(x0, x1)
            dy = abs(rnd.gauss(0, depth / 2))
            els.append(lens(x, surface(x, ripple_end) + 2 + dy,
                            slope(x, ripple_end) + rnd.gauss(0, 0.25),
                            rnd.uniform(5, 8), 2.2))

    def floes(x0, x1, length, thick, gap, jitter, rows=1):
        for r in range(rows):
            x = x0 + rnd.uniform(0, length) * r
            while x < x1:
                L = min(length * rnd.uniform(0.75, 1.2), x1 - x)
                if L < length * 0.5:
                    break
                c = x + L / 2
                off = thick / 2 - 1.5 + r * thick * 0.8 + rnd.gauss(0, jitter)
                els.append(bent(c, off, L, thick, ripple_end))
                x += L + gap * rnd.uniform(0.5, 1.5)

    if stage == 1:
        discs(260, 14)
    elif stage == 2:
        discs(150, 14)
        floes(ice_start + 20, 620, 18, 5, 10, 1.0)
    elif stage == 3:
        discs(90, 12)
        floes(ice_start, 625, 18, 4.5, 0, 1.2, rows=3)
    else:
        discs(50, 14)
        floes(ice_start, 628, 50, 8, 5, 0.4)

    xml = (f'<shape name="pancake{stage}" w="{W}" h="{H}" aspect="variable" strokewidth="inherit">'
           f'<connections/><background/><foreground>'
           f'<strokecolor color="{INK}"/><fillcolor color="{INK}"/>' + "".join(els) +
           '</foreground></shape>')
    enc = urllib.parse.quote(xml, safe="~()*!.'-_")
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.b64encode(co.compress(enc.encode()) + co.flush()).decode()


LABELS = [
    "(i) Frazil layer forms by vertical accumulation and rafting of small frazil discs",
    "(ii) Freezing forms small primary pancakes",
    "(iii) High-frequency waves damped, primary pancakes raft under the longer waves, frazil production reduced",
    "(iv) Freezing of primary pancakes forms mature pancakes, frazil production further reduces",
]

cells = []
for i, label in enumerate(LABELS):
    y0 = i * 200
    cells.append(
        f'<mxCell id="t{i+1}" value="{escape(label)}" style="text;html=1;whiteSpace=wrap;align=left;'
        f'verticalAlign=bottom;fontFamily=Helvetica;fontSize=14;fontColor={INK};" vertex="1" parent="1">'
        f'<mxGeometry x="0" y="{y0}" width="640" height="50" as="geometry"/></mxCell>')
    cells.append(
        f'<mxCell id="p{i+1}" value="" style="shape=stencil({panel(i+1, 4022 + i)});'
        f'strokeColor={INK};fillColor={INK};" vertex="1" parent="1">'
        f'<mxGeometry x="0" y="{y0 + 50}" width="{W}" height="{H}" as="geometry"/></mxCell>')

out = ('<mxfile><diagram id="pancake" name="Pancake-ice formation"><mxGraphModel adaptiveColors="auto" '
       'grid="1" gridSize="10"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
open(sys.argv[1], "w").write(out)
