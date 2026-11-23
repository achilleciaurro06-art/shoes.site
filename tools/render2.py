"""Running-shoe material renders (cool grey palette) built on tools/render.py.

Mesh, pebbled foam, rubber lugs, TPU and yarn on a light grey surface.
Usage: python3 tools/render2.py [name ...]     (RENDER_OUT overrides the output folder)
"""
import sys
import numpy as np
from PIL import Image, ImageDraw
import render as R
from render import Scene, fbm, blur, smooth, poly_mask, sole_pts, resample, bezier, save, STEEL

BG = np.array([.66, .70, .73]); FOAM = np.array([.80, .80, .78]); MESHC = np.array([.84, .84, .82])
GREY = np.array([.50, .53, .56]); CHAR = np.array([.115, .125, .14]); RUB = np.array([.09, .095, .10])
ORANGE = np.array([.95, .33, .11]); WHITE = np.array([.93, .93, .92]); UNDER = np.array([.10, .115, .13])
YARN = [np.array(c) for c in ([.86, .86, .84], [.55, .58, .61], [.30, .33, .36], [.95, .33, .11], [.72, .75, .78])]
COOL = dict(tint=(.96, .99, 1.02), lift=1.25, expo=2.6, vig=.28, grain=.022)


def new(W, H, seed):
    s = Scene(W, H, seed, base=BG, table=False)
    n = fbm(H, W, 3, 3, seed + 3)
    s.a *= (.9 + .18 * n)[..., None]
    s.h += 1.0 * fbm(H, W, 60, 2, seed + 4)
    return s


def foam(s, mask, color=FOAM, amp=4.0, scale=1.0, height=3.0, seed=0):
    H, W = s.H, s.W
    f = fbm(H, W, int(46 / scale), 3, s.seed + seed + 31)
    peb = np.abs(2 * f - 1) ** .55
    hh = amp * peb * s.u + 1.0 * fbm(H, W, 260, 2, s.seed + seed + 5)
    var = .86 + .24 * fbm(H, W, 4, 3, s.seed + seed + 12)
    alb = color[None, None, :] * var[..., None] * (.9 + .16 * peb[..., None])
    s.paint(mask, hh, alb, .08, height, 3)


def mesh(s, mask, color=MESHC, under=UNDER, p=15.0, height=1.5):
    k = 2 * np.pi / (p * s.u)
    X, Y = s.X, s.Y
    h1 = np.cos(k * X) + np.cos(k * (X * .5 + Y * .866)) + np.cos(k * (X * .5 - Y * .866))
    holes = np.clip((h1 - 1.35) * 3.2, 0, 1)
    weave = .5 + .5 * np.sin(k * (X + Y) * 3.4)
    hh = (1.8 + .5 * weave) * s.u * (1 - holes) - 2.6 * s.u * holes
    var = .88 + .2 * fbm(s.H, s.W, 5, 3, s.seed + 44)
    alb = color[None, None, :] * (.82 + .2 * weave)[..., None] * var[..., None] * (1 - holes[..., None] * .9) + under * holes[..., None]
    s.paint(mask, hh, alb, .10, height, 2)


def rubber(s, mask, color=RUB, a=26.0, b=15.0, height=3.0):
    X, Y = s.X, s.Y
    a *= s.u; b *= s.u
    row = np.floor(X / a)
    gx = np.abs(np.sin(np.pi * X / a)); gy = np.abs(np.sin(np.pi * (Y + (row % 2) * b / 2) / b))
    blocks = smooth((gx * gy - .22) * 5)
    alb = color[None, None, :] * (.65 + .9 * blocks)[..., None] * (.85 + .3 * fbm(s.H, s.W, 120, 2, s.seed + 9))[..., None]
    s.paint(mask, 3.0 * blocks * s.u + .6 * fbm(s.H, s.W, 200, 2, 3), alb, .26, height, 3)


def tpu(s, mask, color=CHAR, height=3.0):
    alb = color[None, None, :] * (.9 + .25 * fbm(s.H, s.W, 6, 2, s.seed + 8))[..., None] * np.ones((s.H, s.W, 3))
    s.paint(mask, .5 * fbm(s.H, s.W, 80, 2, 2), alb, .8, height, 4)


def layer(s, pts, kind, **kw):
    m = poly_mask(s.H, s.W, pts, 1.1)
    s.cast(m, 7, 11, 9, .5)
    {'foam': foam, 'mesh': mesh, 'rubber': rubber, 'tpu': tpu}[kind](s, m, **kw)
    return m


def swatch(s, cx, cy, w, h, ang, kind, **kw):
    c, sn = np.cos(ang), np.sin(ang)
    pts = [(cx + x * c - y * sn, cy + x * sn + y * c) for x, y in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]]
    return layer(s, pts, kind, **kw)


# ------------------------------------------------------------------ images
def img_grain(W=800, H=600):
    s = new(W, H, 12)
    mesh(s, np.ones((H, W), np.float32), p=13)
    band = poly_mask(H, W, [(-40, .78 * H), (W + 40, .30 * H), (W + 40, .52 * H), (-40, 1.1 * H)], 1.2)
    s.cast(band, 6, 10, 9, .6); tpu(s, band, CHAR, 3.5)
    stripe = poly_mask(H, W, [(-40, .70 * H), (W + 40, .22 * H), (W + 40, .245 * H), (-40, .725 * H)], .8)
    s.paint(stripe, 0 * stripe, np.ones((H, W, 3)) * ORANGE, .5, 1.0, 1)
    return s.render(light=(-.5, -.7, .45), lamp=(.3, .15), dof=(.62, .5), **COOL)


def img_toe(W=900, H=675):
    s = new(W, H, 22)
    full = np.ones((H, W), np.float32)
    foam(s, full, FOAM, 5.0, .9)
    bump = poly_mask(H, W, [(-30, .55 * H), (.18 * W, .50 * H), (.40 * W, .66 * H), (.62 * W, .84 * H), (.7 * W, H + 30), (-30, H + 30)], 1.2)
    s.cast(bump, 5, 8, 9, .6); rubber(s, bump, RUB, 24, 14, 4)
    pl = bezier((-20, .46 * H), (.25 * W, .40 * H), (.6 * W, .60 * H), (W + 20, .50 * H), 60)
    s.groove(pl, 7, 4.0, .6)
    ln = poly_mask(H, W, [(-20, .30 * H), (W + 20, .34 * H), (W + 20, .345 * H), (-20, .305 * H)], .8)
    s.paint(ln, 0 * ln, np.ones((H, W, 3)) * ORANGE, .4, .6, 1)
    return s.render(light=(-.6, -.55, .45), lamp=(.28, .2), dof=(.5, .6), **COOL)


def img_laces(W=640, H=800):
    s = new(W, H, 32)
    mesh(s, np.ones((H, W), np.float32), MESHC, UNDER, p=14)
    strip = smooth((1 - np.abs(s.X - .5 * W) / (.09 * W)) * 3.0)
    s.cast(strip, 2, 6, 6, .7)
    mesh(s, strip, GREY * .7, UNDER, p=11)
    xl, xr = .39 * W, .61 * W
    ys = np.linspace(.09 * H, .93 * H, 8)
    for x in (.18 * W, .82 * W):
        pts = np.c_[x + 3 * s.u * np.sin(np.linspace(0, 6, 60)), np.linspace(-10, H + 10, 60)]
        s.groove(pts, 3, 1.4, .25); s.stitch(pts, 17, 10, 1.7, WHITE * .9, 1.4)
    for y in ys:
        for x in (xl, xr):
            s.hole(x, y, 12, 2.4)
    for i in range(len(ys) - 1):
        for (x0, x1) in [(xl, xr), (xr, xl)]:
            pts = bezier((x0, ys[i]), (x0 + (x1 - x0) * .3, ys[i] + 4), (x1 - (x1 - x0) * .3, ys[i + 1] - 8), (x1, ys[i + 1]), 28)
            s.cord(pts, 8.5, CHAR * 1.7, twill=3.2)
    return s.render(light=(-.7, -.4, .42), lamp=(.2, .1), dof=(.45, .55), **COOL)


def spool(s, cx, cy, Rr, color):
    R.spool(s, cx, cy, Rr, color)


def img_needle(W=900, H=711):
    s = new(W, H, 42)
    spool(s, .30 * W, .42 * H, 190, YARN[1]); spool(s, .72 * W, .30 * H, 150, YARN[0]); spool(s, .66 * W, .76 * H, 120, YARN[3])
    s.cord([(.05 * W, .92 * H), (.95 * W, .58 * H)], 7.5, STEEL, style='steel')
    return s.render(light=(-.5, -.65, .45), lamp=(.3, .15), dof=(.62, .45), **COOL)


def img_sole(W=1100, H=360):
    s = new(W, H, 52)
    Yp = s.Y + 14 * s.u * np.sin(np.pi * s.X / W) - 10 * s.u * (s.X / W)
    def band(a, b): return R.band(Yp, a * H, b * H, 1.6)
    m1, m2, m3, m4 = band(.06, .36), band(.36, .43), band(.43, .66), band(.66, .90)
    for m, fn, kw in [(m1, foam, dict(color=FOAM, amp=4.5)), (m2, tpu, dict(color=CHAR)), (m3, foam, dict(color=FOAM * .9, amp=4.0, seed=3)), (m4, rubber, dict(color=RUB, a=22, b=12))]:
        s.cast(m, 0, 5, 5, .6); fn(s, m, **kw)
    return s.render(light=(-.35, -.8, .45), lamp=(.35, .1), **COOL)


def img_stack(W=1400, H=675):
    s = new(W, H, 62)
    kinds = [('rubber', dict(color=RUB, a=26, b=15)), ('foam', dict(color=FOAM * .92, amp=4)), ('tpu', dict(color=CHAR)), ('foam', dict(color=FOAM, amp=4, seed=4)), ('mesh', dict(color=MESHC, under=UNDER))]
    for i, (k, kw) in enumerate(kinds):
        layer(s, sole_pts(.50 * W - i * 8 * s.u, .62 * H - i * 26 * s.u, .74 * W, .38 * H, -.10 + i * .004), k, **kw)
    return s.render(light=(-.5, -.6, .5), lamp=(.3, .2), dof=(.5, .5), **COOL)


def img_insole(W=1200, H=440):
    s = new(W, H, 72)
    pts = sole_pts(.5 * W, .50 * H, .86 * W, .62 * H, -.07)
    m = layer(s, pts, 'foam', color=FOAM * .95, amp=2.5)
    for x in np.arange(.14 * W, .86 * W, 30 * s.u):
        for y in np.arange(.28 * H, .72 * H, 30 * s.u):
            if m[int(y), int(x)] > .9: s.hole(x + 15 * s.u * (int(y / (30 * s.u)) % 2), y, 3.2, 1.8)
    heel = poly_mask(H, W, sole_pts(.80 * W, .5 * H, .22 * W, .36 * H, -.07), 3)
    s.paint(heel * m, 0 * heel, np.ones((H, W, 3)) * ORANGE * .9, .2, 0, 2)
    return s.render(light=(-.5, -.65, .5), lamp=(.3, .2), dof=(.5, .6), **COOL)


def img_threads(W=1400, H=554):
    s = new(W, H, 82)
    layout = [(.12, .50, 108, YARN[0]), (.30, .38, 92, YARN[1]), (.47, .60, 112, YARN[2]), (.65, .40, 96, YARN[3]), (.84, .58, 104, YARN[4])]
    for x, y, Rr, c in layout: spool(s, x * W, y * H, Rr, c)
    s.cord([(.20 * W, .86 * H), (.60 * W, .12 * H)], 7, STEEL, style='steel')
    s.cord([(.46 * W, .90 * H), (.90 * W, .74 * H)], 6.5, STEEL, style='steel')
    return s.render(light=(-.5, -.55, .5), lamp=(.4, .15), dof=(.5, .7), **COOL)


def img_wide(W=1400, H=741):
    s = new(W, H, 92)
    kinds = [('mesh', dict(color=MESHC)), ('mesh', dict(color=GREY, under=UNDER)), ('foam', dict(color=FOAM, amp=4.5)), ('rubber', dict(color=RUB)), ('tpu', dict(color=CHAR)), ('mesh', dict(color=ORANGE * .95, under=np.array([.35, .10, .04])))]
    rng = np.random.default_rng(4); k = 0
    for r in range(2):
        for c in range(3):
            kind, kw = kinds[k]
            swatch(s, (.19 + c * .31) * W, (.27 + r * .46) * H, .26 * W, .38 * H, rng.uniform(-.06, .06), kind, **kw); k += 1
    return s.render(light=(-.5, -.6, .5), lamp=(.35, .2), dof=(.5, .7), **COOL)


def img_exploded(W=1400, H=708):
    s = new(W, H, 102)
    layer(s, sole_pts(.30 * W, .66 * H, .50 * W, .26 * H, -.12), 'rubber', color=RUB)
    layer(s, sole_pts(.33 * W, .54 * H, .48 * W, .25 * H, -.12), 'foam', color=FOAM, amp=4)
    layer(s, sole_pts(.70 * W, .34 * H, .46 * W, .25 * H, .10), 'tpu', color=CHAR)
    layer(s, sole_pts(.72 * W, .72 * H, .42 * W, .22 * H, .06), 'mesh', color=MESHC)
    cx, cy = .55 * W, .22 * H
    th = np.linspace(0, 6 * np.pi, 160)
    coil = np.c_[cx + (14 + 10 * th) * s.u * .9 * np.cos(th) * .6, cy + (14 + 10 * th) * s.u * .9 * np.sin(th) * .38]
    s.cord(coil, 6, CHAR * 1.7, twill=3.0)
    s.cord([(.12 * W, .12 * H), (.40 * W, .08 * H)], 6.5, STEEL, style='steel')
    return s.render(light=(-.5, -.55, .5), lamp=(.35, .2), dof=(.5, .7), **COOL)


def img_about(W=720, H=900):
    s = new(W, H, 112)
    swatch(s, .36 * W, .26 * H, .46 * W, .26 * H, -.07, 'mesh', color=MESHC)
    swatch(s, .60 * W, .46 * H, .46 * W, .24 * H, .06, 'foam', color=FOAM, amp=4.5)
    swatch(s, .38 * W, .68 * H, .46 * W, .24 * H, -.04, 'rubber', color=RUB)
    spool(s, .78 * W, .82 * H, 70, YARN[3]); spool(s, .20 * W, .90 * H, 52, YARN[0])
    s.cord([(.12 * W, .46 * H), (.55 * W, .90 * H)], 6, STEEL, style='steel')
    return s.render(light=(-.5, -.6, .5), lamp=(.3, .15), dof=(.45, .6), **COOL)


JOBS = dict(grain=img_grain, toe=img_toe, laces=img_laces, needle=img_needle, sole=img_sole, stack=img_stack,
            insole=img_insole, threads=img_threads, wide=img_wide, exploded=img_exploded, about=img_about)

if __name__ == '__main__':
    for n in (sys.argv[1:] or list(JOBS)):
        save(JOBS[n](), n)
