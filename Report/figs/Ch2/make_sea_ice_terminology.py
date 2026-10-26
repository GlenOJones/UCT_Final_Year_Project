import base64, math, random, sys, urllib.parse, zlib
from xml.sax.saxutils import escape

# Colours taken from figs/ch1/ROVIceSetup.pdf
INK, WATER, ICE, SNOW, RED = "#000000", "#1C1A6E", "#C9DDF5", "#FFFFFF", "#D93025"
W, H = 430, 172
FONT = 14

SNOW_TOP, ICE_TOP, SEA, ICE_BOT = 26, 48, 72, 150
L, R = 150, 280          # floe extent at the snow-ice interface
LA, LB = 126, 140        # left dimension lines: total freeboard, thickness
RA = 292                 # right dimension line


def poly(pts, op):
    d = f'<move x="{pts[0][0]:.1f}" y="{pts[0][1]:.1f}"/>'
    d += "".join(f'<line x="{x:.1f}" y="{y:.1f}"/>' for x, y in pts[1:])
    return f"<path>{d}<close/></path><{op}/>"


def fill(c):
    return f'<fillcolor color="{c}"/>'


def line(a, b):
    return f'<path><move x="{a[0]:.1f}" y="{a[1]:.1f}"/><line x="{b[0]:.1f}" y="{b[1]:.1f}"/></path><stroke/>'


def edge(rnd, a, b, n, amp):
    (x0, y0), (x1, y1) = a, b
    return [(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n + (rnd.uniform(-amp, amp) if 0 < i < n else 0))
            for i in range(n + 1)]


def dim(x, y0, y1):
    out = line((x, y0 + 5), (x, y1 - 5))
    out += poly([(x, y0), (x - 3, y0 + 6), (x + 3, y0 + 6)], "fill")
    out += poly([(x, y1), (x - 3, y1 - 6), (x + 3, y1 - 6)], "fill")
    return out


def ext(x0, x1, y):
    return '<dashed dashed="1"/>' + line((x0, y), (x1, y)) + '<dashed dashed="0"/>'


def scene(seed):
    rnd = random.Random(seed)
    out = f'<strokecolor color="{INK}"/><strokewidth width="2"/>'
    out += f'<strokecolor color="none"/>' + fill(WATER) + poly([(0, SEA), (W, SEA), (W, H), (0, H)], "fill")

    # Floe: flat-ish top, slanted jagged sides and a rough underside, like the ROV figure's floes
    top = edge(rnd, (L, ICE_TOP), (R, ICE_TOP), 8, 0.6)
    right = [(R, ICE_TOP), (R + 4, ICE_TOP + 30), (R - 2, ICE_BOT - 30), (R - 8, ICE_BOT)]
    bot = edge(rnd, (R - 8, ICE_BOT), (L + 6, ICE_BOT), 9, 2.5)
    left = [(L + 6, ICE_BOT), (L - 3, ICE_BOT - 40), (L + 2, ICE_TOP + 25), (L, ICE_TOP)]
    floe = top + right[1:] + bot[1:] + left[1:-1]
    out += f'<strokecolor color="{INK}"/>' + fill(ICE) + poly(floe, "fillstroke")

    # Snow: wavy top sitting on the floe
    snow = [(L + 4 + (R - L - 8) * i / 40, SNOW_TOP + 1.5 * math.sin(i / 3.1) + 0.8 * math.sin(i / 1.3))
            for i in range(41)]
    snow = [(L, ICE_TOP), (L + 2, SNOW_TOP + 6)] + snow + [(R - 2, SNOW_TOP + 6), (R, ICE_TOP)]
    out += fill(SNOW) + poly(snow, "fillstroke")

    # Dimension lines
    out += f'<strokecolor color="{RED}"/><strokewidth width="1.5"/>' + fill(RED)
    out += ext(LA - 6, L, SNOW_TOP) + ext(LB - 6, L, ICE_TOP) + ext(LB - 6, L + 6, ICE_BOT)
    out += ext(R, RA + 6, SNOW_TOP) + ext(R, RA + 6, ICE_TOP) + ext(R - 8, RA + 6, ICE_BOT)
    out += dim(LA, SNOW_TOP, SEA) + dim(LB, ICE_TOP, ICE_BOT)
    out += dim(RA, SNOW_TOP, ICE_TOP) + dim(RA, ICE_TOP, SEA) + dim(RA, SEA, ICE_BOT)

    xml = (f'<shape name="seaiceterms" w="{W}" h="{H}" aspect="variable" strokewidth="inherit">'
           f'<connections/><background/><foreground>{out}</foreground></shape>')
    enc = urllib.parse.quote(xml, safe="~()*!.'-_")
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    return base64.b64encode(co.compress(enc.encode()) + co.flush()).decode()


# (text, x, y, width, align, colour)
LABELS = [
    ("Total freeboard", 0, SNOW_TOP + 13, LA - 8, "right", INK),
    ("Sea ice thickness", 0, 100, LB - 8, "right", SNOW),
    ("Snow", L, SNOW_TOP + 1, R - L, "center", INK),
    ("Sea ice", L, 90, R - L, "center", INK),
    ("Snow depth", RA + 8, SNOW_TOP + 1, 130, "left", INK),
    ("Sea ice freeboard", RA + 8, ICE_TOP + 2, 130, "left", INK),
    ("Sea ice draft", RA + 8, 100, 130, "left", SNOW),
    ("Seawater", 8, H - 28, 100, "left", SNOW),
]

cells = [f'<mxCell id="scene" value="" style="shape=stencil({scene(4022)});strokeColor={INK};fillColor={INK};" '
         f'vertex="1" parent="1"><mxGeometry x="0" y="0" width="{W}" height="{H}" as="geometry"/></mxCell>']
for i, (s, x, y, w, align, col) in enumerate(LABELS):
    cells.append(
        f'<mxCell id="l{i}" value="{escape(s)}" style="text;html=1;align={align};verticalAlign=middle;'
        f'fontFamily=Helvetica;fontSize={FONT};fontColor={col};spacing=0;" vertex="1" parent="1">'
        f'<mxGeometry x="{x}" y="{y}" width="{w}" height="20" as="geometry"/></mxCell>')

out = ('<mxfile><diagram id="seaiceterms" name="Sea ice terminology"><mxGraphModel '
       'grid="1" gridSize="10" page="0"><root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) +
       '</root></mxGraphModel></diagram></mxfile>')
open(sys.argv[1], "w").write(out)
