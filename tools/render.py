"""Procedural renderer for the footwear-craft imagery used on the site.

Everything is built as a height-map + albedo-map, then lit with a raking lamp
(diffuse + specular + cavity occlusion), depth-of-field, vignette and film grain.
Pure numpy + Pillow. Usage: python3 tools/render.py [name ...]
"""
import sys, os
import numpy as np
from PIL import Image, ImageDraw

OUT = os.environ.get('RENDER_OUT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'img')

# ----- palette (linear-ish) ------------------------------------------------
TAN = np.array([.70, .42, .20]); COGNAC = np.array([.52, .27, .11]); CHESTNUT = np.array([.33, .165, .08])
BLACK = np.array([.05, .045, .04]); SUEDE = np.array([.52, .38, .25]); BEIGE = np.array([.80, .68, .50])
THREAD = np.array([.86, .76, .58]); WAX = np.array([.20, .11, .06]); STEEL = np.array([.72, .72, .74])
TABLE = np.array([.040, .031, .026])


# ----- helpers -------------------------------------------------------------
def blur(a, s):
    """Gaussian blur via FFT (no scipy). a: HxW or HxWxC."""
    if s <= .05: return a
    if a.ndim == 3: return np.dstack([blur(a[..., i], s) for i in range(a.shape[2])])
    p = int(s * 3) + 2
    b = np.pad(a, p, mode='reflect')
    H, W = b.shape
    fy = np.fft.fftfreq(H)[:, None]; fx = np.fft.rfftfreq(W)[None, :]
    k = np.exp(-2 * np.pi ** 2 * s ** 2 * (fx ** 2 + fy ** 2))
    r = np.fft.irfft2(np.fft.rfft2(b) * k, s=b.shape)
    return r[p:-p, p:-p]


def vnoise(H, W, cx, cy, seed):
    r = np.random.default_rng(seed)
    g = r.random((max(2, cy) + 2, max(2, cx) + 2)).astype(np.float32)
    return np.clip(np.asarray(Image.fromarray(g, mode='F').resize((W, H), Image.BICUBIC)), 0, 1)


def fbm(H, W, base, octv=4, seed=0, gain=.5, aspect=1.0):
    t = 0; amp = 1; n = 0
    for o in range(octv):
        c = base * 2 ** o
        t = t + amp * vnoise(H, W, c, max(1, int(c * H / W * aspect)), seed + o * 7); n += amp; amp *= gain
    return t / n


def smooth(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def segd(X, Y, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]; l2 = dx * dx + dy * dy + 1e-9
    t = np.clip(((X - a[0]) * dx + (Y - a[1]) * dy) / l2, 0, 1)
    return np.hypot(X - (a[0] + t * dx), Y - (a[1] + t * dy)), t


def resample(pts, step):
    pts = np.array(pts, float); out = [pts[0]]; acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        L = np.hypot(*(b - a)); d = step - acc
        while d <= L:
            out.append(a + (b - a) * d / L); d += step
        acc = L - (d - step)
    return np.array(out)


def bezier(p0, p1, p2, p3, n=80):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * np.array(p0) + 3 * (1 - t) ** 2 * t * np.array(p1) + 3 * (1 - t) * t ** 2 * np.array(p2) + t ** 3 * np.array(p3)


def poly_mask(H, W, pts, aa=1.0):
    s = 2
    im = Image.new('L', (W * s, H * s), 0)
    ImageDraw.Draw(im).polygon([(x * s, y * s) for x, y in pts], fill=255)
    m = np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255
    return blur(m, aa) if aa > 0 else m


def sole_pts(cx, cy, L, Wd, ang, n=160):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = np.cos(t); y = np.sin(t)
    tx = (x + 1) / 2                                   # 0 = heel, 1 = toe
    w = (0.66 + .34 * smooth(tx * 1.6)) * (1 - .24 * np.exp(-((tx - .42) / .16) ** 2 * 1.0))
    w = w * np.where(tx > .88, 1 - (tx - .88) * 1.2, 1)
    px = x * L / 2; py = y * Wd / 2 * w
    c, s = np.cos(ang), np.sin(ang)
    return np.c_[cx + px * c - py * s, cy + px * s + py * c]


# ----- scene ---------------------------------------------------------------
class Scene:
    def __init__(s, W, H, seed=1, base=TABLE, table=True):
        s.W, s.H, s.seed = W, H, seed
        s.Y, s.X = np.mgrid[0:H, 0:W].astype(np.float32)
        s.u = W / 1000.0
        s.h = np.zeros((H, W), np.float32)
        s.a = np.ones((H, W, 3), np.float32) * base
        s.gloss = np.full((H, W), .12, np.float32)
        if table:
            wood = fbm(H, W, 3, 3, seed + 90, aspect=14)
            s.a *= (.7 + .9 * wood[..., None]); s.h += 1.2 * wood

    # surfaces
    def leather(s, mask, color, grain=2.2, scale=1.0, gloss=.35, height=0.0, bevel=3, seed=0):
        H, W = s.H, s.W
        f = fbm(H, W, int(92 / scale), 3, s.seed + seed + 1)
        g = (1 - np.abs(2 * f - 1)) ** 2.3
        fine = fbm(H, W, int(420 / scale), 2, s.seed + seed + 5)
        wr = (1 - np.abs(2 * fbm(H, W, 4, 3, s.seed + seed + 9) - 1)) ** 4
        hh = grain * .55 * (g * .8 + .3 * fine) * s.u + 1.6 * wr * s.u
        var = .82 + .36 * fbm(H, W, 3, 3, s.seed + seed + 12)
        alb = color[None, None, :] * var[..., None] * (.82 + .3 * g[..., None])
        s.paint(mask, hh, alb, gloss, height, bevel)

    def paint(s, mask, hh, alb, gloss=.3, height=0.0, bevel=3):
        m = np.clip(mask, 0, 1)
        hb = blur(m, bevel * s.u) * height * s.u
        base_h = s.h
        s.h = base_h * (1 - m) + (hb + hh + base_h * 0) * m + (hb * 0)
        s.a = s.a * (1 - m[..., None]) + alb * m[..., None]
        s.gloss = s.gloss * (1 - m) + gloss * m

    def cast(s, mask, dx, dy, r, strength=.6):
        sh = np.roll(np.roll(blur(mask, r * s.u), int(dy * s.u), 0), int(dx * s.u), 1)
        s.a *= (1 - strength * np.clip(sh, 0, 1))[..., None]

    # features
    def groove(s, pts, width, depth, dark=.5):
        d = np.full((s.H, s.W), 1e4, np.float32)
        for a, b in zip(pts[:-1], pts[1:]):
            d = np.minimum(d, segd(s.X, s.Y, a, b)[0])
        g = np.exp(-(d / (width * s.u)) ** 2)
        s.h -= depth * s.u * g; s.a *= (1 - dark * g[..., None])

    def stitch(s, pts, spacing, length, width, color=THREAD, lift=1.8, tilt=0.0, shadow=True):
        P = resample(pts, spacing * s.u)
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            tg = (b - a) / (np.hypot(*(b - a)) + 1e-9); nr = np.array([-tg[1], tg[0]])
            c = (a + b) / 2; h = tg * length * s.u / 2 + nr * tilt * length * s.u * .25
            p0, p1 = c - h, c + h
            r = (width + 2.5) * s.u
            x0, x1 = int(max(0, min(p0[0], p1[0]) - r - 2)), int(min(s.W, max(p0[0], p1[0]) + r + 2))
            y0, y1 = int(max(0, min(p0[1], p1[1]) - r - 2)), int(min(s.H, max(p0[1], p1[1]) + r + 2))
            if x1 <= x0 or y1 <= y0: continue
            X, Y = s.X[y0:y1, x0:x1], s.Y[y0:y1, x0:x1]
            d, _ = segd(X, Y, p0, p1)
            prof = np.sqrt(np.clip(1 - (d / (width * s.u)) ** 2, 0, 1))
            m = (d < width * s.u).astype(np.float32)
            if shadow:
                sd = np.exp(-(d / (width * s.u * 1.9)) ** 2) * .35
                s.a[y0:y1, x0:x1] *= (1 - sd)[..., None]
            s.h[y0:y1, x0:x1] += lift * s.u * prof * m
            sh = (.65 + .5 * prof)[..., None]
            s.a[y0:y1, x0:x1] = s.a[y0:y1, x0:x1] * (1 - m[..., None]) + color * sh * m[..., None]
            s.gloss[y0:y1, x0:x1] = np.where(m > 0, .12, s.gloss[y0:y1, x0:x1])

    def hole(s, cx, cy, r, depth=3.0, ring=True):
        r = r * s.u
        x0, x1 = int(max(0, cx - r * 2)), int(min(s.W, cx + r * 2)); y0, y1 = int(max(0, cy - r * 2)), int(min(s.H, cy + r * 2))
        if x1 <= x0 or y1 <= y0: return
        d = np.hypot(s.X[y0:y1, x0:x1] - cx, s.Y[y0:y1, x0:x1] - cy)
        m = np.clip((r - d) / 1.3, 0, 1)
        rim = np.clip(1 - np.abs(d - r) / (1.6 * s.u), 0, 1) if ring else 0
        s.h[y0:y1, x0:x1] += -depth * s.u * m + .9 * s.u * rim
        s.a[y0:y1, x0:x1] = s.a[y0:y1, x0:x1] * (1 - m[..., None]) + np.array([.012, .008, .006]) * m[..., None]

    def cord(s, pts, r, color, style='cord', shadow=True, twill=0.0, lift=0.0):
        pts = np.asarray(pts, float); R = r * s.u
        mask_all = np.zeros((s.H, s.W), np.float32)
        for a, b in zip(pts[:-1], pts[1:]):
            x0, x1 = int(max(0, min(a[0], b[0]) - R * 2 - 2)), int(min(s.W, max(a[0], b[0]) + R * 2 + 2))
            y0, y1 = int(max(0, min(a[1], b[1]) - R * 2 - 2)), int(min(s.H, max(a[1], b[1]) + R * 2 + 2))
            if x1 <= x0 or y1 <= y0: continue
            d, _ = segd(s.X[y0:y1, x0:x1], s.Y[y0:y1, x0:x1], a, b)
            mask_all[y0:y1, x0:x1] = np.maximum(mask_all[y0:y1, x0:x1], np.clip((R - d) / 1.2, 0, 1))
        if shadow: s.cast(mask_all, R * .55, R * .75, R * .9, .62)
        for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
            x0, x1 = int(max(0, min(a[0], b[0]) - R * 2 - 2)), int(min(s.W, max(a[0], b[0]) + R * 2 + 2))
            y0, y1 = int(max(0, min(a[1], b[1]) - R * 2 - 2)), int(min(s.H, max(a[1], b[1]) + R * 2 + 2))
            if x1 <= x0 or y1 <= y0: continue
            X, Y = s.X[y0:y1, x0:x1], s.Y[y0:y1, x0:x1]
            d, t = segd(X, Y, a, b)
            m = np.clip((R - d) / 1.2, 0, 1)
            prof = np.sqrt(np.clip(1 - (d / R) ** 2, 0, 1))
            dx, dy = b - a; l = np.hypot(dx, dy) + 1e-9
            side = ((X - a[0]) * -dy + (Y - a[1]) * dx) / l / R          # -1..1 across the cord
            hh = lift * s.u + R * 1.1 * prof
            s.h[y0:y1, x0:x1] = np.where(m > .02, np.maximum(s.h[y0:y1, x0:x1] * (1 - m) + hh * m, 0 * hh), s.h[y0:y1, x0:x1]) if lift == 0 else np.where(m > .02, hh, s.h[y0:y1, x0:x1])
            if style == 'steel':
                alb = (.10 + .95 * (.5 + .5 * np.cos(3.2 * side + .7)) ** 3)[..., None] * STEEL
                gl = .95
            else:
                tw = 1.0
                if twill: tw = .78 + .22 * np.sin((np.hypot(X - a[0], Y - a[1]) + i * l) / (twill * s.u) * 6.28 + side * 2.2)
                alb = color * (.55 + .55 * prof)[..., None] * (tw[..., None] if twill else 1)
                gl = .35
            s.a[y0:y1, x0:x1] = s.a[y0:y1, x0:x1] * (1 - m[..., None]) + alb * m[..., None]
            s.gloss[y0:y1, x0:x1] = np.where(m > .02, gl, s.gloss[y0:y1, x0:x1])

    # lighting
    def render(s, light=(-.55, -.6, .5), lamp=(.33, .22), lamp_r=.85, vig=.62, dof=None, expo=2.0, grain=.03, warm=1.0, tint=(1.0, .93, .80), lift=1.0):
        H, W = s.H, s.W
        gy, gx = np.gradient(s.h)
        n = np.dstack([-gx, -gy, np.ones_like(s.h)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
        L = np.array(light, float); L /= np.linalg.norm(L)
        diff = np.clip(n @ L, 0, 1)
        Hh = L + np.array([0, 0, 1.]); Hh /= np.linalg.norm(Hh)
        spec = np.clip(n @ Hh, 0, 1) ** 34 * s.gloss
        cav = np.clip(blur(s.h, 7 * s.u) - s.h, 0, None) * .55
        light_amt = (.18 + .82 * diff) * (1 - np.clip(cav, 0, .75))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        lampf = np.exp(-(((xx - lamp[0] * W) / (lamp_r * W)) ** 2 + ((yy - lamp[1] * H) / (lamp_r * H)) ** 2))
        img = s.a * (light_amt * (.38 + .95 * lampf) * lift)[..., None] * np.array(tint) * warm
        img = img + (spec * (.35 + .9 * lampf))[..., None] * np.array([1, .9, .76])
        if dof is not None:                              # (focus_y, falloff)
            fy, fo = dof
            mk = smooth((np.abs(yy / H - fy) - .12) / fo)[..., None]
            img = img * (1 - mk) + blur(img, 6 * s.u) * mk
        v = 1 - vig * (((xx / W - .5) * 1.5) ** 2 + ((yy / H - .5) * 1.5) ** 2)
        img = img * np.clip(v, .08, 1)[..., None]
        img = 1 - np.exp(-img * expo)                    # filmic roll-off
        img = img ** (1 / 1.15)
        r = np.random.default_rng(s.seed + 5)
        img = img + (r.standard_normal((H, W, 1)) * grain)
        return np.clip(img * 255, 0, 255).astype(np.uint8)


def save(arr, name, q=80):
    Image.fromarray(arr).save(os.path.join(OUT, name + '.jpg'), quality=q, optimize=True, progressive=True)
    print('saved', name, arr.shape[1], 'x', arr.shape[0], flush=True)


# ===========================================================================
# IMAGES
# ===========================================================================
def img_grain(W=800, H=600):
    s = Scene(W, H, 11, table=False)
    full = np.ones((H, W), np.float32)
    s.leather(full, TAN * .95, grain=3.4, scale=1.0, gloss=.42)
    seam = bezier((-20, .66 * H), (.3 * W, .56 * H), (.65 * W, .74 * H), (W + 20, .60 * H), 60)
    s.groove(seam, 5, 3.2, .45)
    up = seam + np.array([0, -16 * s.u]); dn = seam + np.array([0, 16 * s.u])
    s.stitch(seam, 26, 15, 2.3, THREAD, 2.0)
    return s.render(light=(-.5, -.7, .42), lamp=(.3, .15), dof=(.62, .5), vig=.5)


def panel_edge(s, mask, depth=4):
    s.cast(mask, 5, 7, 7, .7)


def img_toe(W=900, H=675, wide=False):
    s = Scene(W, H, 21, table=False)
    full = np.ones((H, W), np.float32)
    s.leather(full, COGNAC * 1.05, grain=3.0, gloss=.38, seed=2)
    cx, cy = (.02 * W, .50 * H) if not wide else (.10 * W, .52 * H)
    a, b = .72 * W, .78 * H
    if wide: a, b = .60 * W, .70 * H
    ang = np.linspace(0, 2 * np.pi, 300)
    X, Y = s.X, s.Y
    inside = ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2
    cap = np.clip((1 - inside) * 40, 0, 1)
    cap = blur(cap, 1.0)
    s.cast(cap, 7, 10, 9, .75)
    s.leather(cap, TAN * 1.02, grain=2.6, gloss=.45, height=3.6, bevel=5, seed=3)
    # scalloped ring of perforations + stitches along the cap edge
    for k, (off, rr) in enumerate([(30, 5.2), (52, 0)]):
        pts = np.c_[cx + (a - off * s.u) * np.cos(ang), cy + (b - off * s.u) * np.sin(ang)]
        if rr:
            for p in resample(pts, 24 * s.u): s.hole(p[0], p[1], rr, 3.2)
        else:
            s.stitch(pts, 18, 11, 1.9, THREAD * .95, 1.6)
    # medallion
    rings = [(0, 1, 8.5), (26, 6, 6.2), (52, 12, 5.0), (78, 18, 4.2)]
    for R, N, rr in rings:
        for i in range(N):
            t = 2 * np.pi * i / max(N, 1) + (.2 if R else 0)
            px, py = cx + .42 * a + R * s.u * 1.0 * np.cos(t), cy + R * s.u * 1.25 * np.sin(t)
            if ((px - cx) / (a - 70 * s.u)) ** 2 + ((py - cy) / (b - 70 * s.u)) ** 2 < 1:
                s.hole(px, py, rr, 3.4)
    return s.render(light=(-.6, -.55, .45), lamp=(.28, .2), dof=(.5, .6), vig=.55)


def img_laces(W=640, H=800):
    s = Scene(W, H, 31, table=False)
    full = np.ones((H, W), np.float32)
    s.leather(full, COGNAC * 1.1, grain=3.0, gloss=.40, seed=4)
    # recessed tongue strip
    strip = smooth((1 - np.abs(s.X - .5 * W) / (.085 * W)) * 3.0)
    s.cast(strip, 2, 6, 6, .8)
    s.paint(strip, 1.4 * fbm(H, W, 120, 2, 3), (SUEDE * .75)[None, None, :] * (.8 + .4 * fbm(H, W, 90, 2, 5))[..., None], .08, -3, 3)
    xl, xr = .395 * W, .605 * W
    ys = np.linspace(.09 * H, .93 * H, 8)
    # borders: stitches and brogue dots
    for x, sgn in [(.17 * W, 1), (.83 * W, -1)]:
        pts = np.c_[x + 5 * s.u * np.sin(np.linspace(0, 6, 60)), np.linspace(-10, H + 10, 60)]
        s.groove(pts, 3, 1.6, .25); s.stitch(pts, 17, 10, 1.8, THREAD, 1.5)
    for x in (.255 * W, .745 * W):
        for y in np.arange(.02 * H, H, 19 * s.u): s.hole(x + 3 * s.u * np.sin(y / 60), y, 3.6, 2.4)
    for y in ys:
        for x in (xl, xr): s.hole(x, y, 13, 2.0);
    # criss-cross laces
    for i in range(len(ys) - 1):
        for (x0, x1) in [(xl, xr), (xr, xl)]:
            mid = ((x0 + x1) / 2, (ys[i] + ys[i + 1]) / 2 + 6 * s.u)
            pts = bezier((x0, ys[i]), (x0 + (x1 - x0) * .3, ys[i] + 4), (x1 - (x1 - x0) * .3, ys[i + 1] - 8), (x1, ys[i + 1]), 28)
            s.cord(pts, 8.5, WAX * 1.5, twill=3.2, lift=0)
    return s.render(light=(-.7, -.4, .42), lamp=(.2, .1), dof=(.45, .55), vig=.6)


def img_needle(W=900, H=711):
    s = Scene(W, H, 41, table=False)
    s.leather(np.ones((H, W), np.float32), TAN * .98, grain=3.4, gloss=.42, seed=6)
    seam = bezier((-20, .68 * H), (.2 * W, .64 * H), (.45 * W, .72 * H), (.58 * W, .67 * H), 40)
    s.groove(seam, 5, 3.2, .45)
    s.stitch(seam, 34, 19, 3.0, THREAD, 2.2, tilt=.25)
    tip = (.60 * W, .665 * H)
    tail = (.97 * W, .08 * H)
    # thread: from last stitch up in a loose loop to the eye
    eye = (tail[0] - (tail[0] - tip[0]) * .09, tail[1] + (tip[1] - tail[1]) * .09)
    th = np.vstack([bezier(tip, (.62 * W, .36 * H), (.40 * W, .22 * H), (.50 * W, .10 * H), 50),
                    bezier((.50 * W, .10 * H), (.66 * W, -.02 * H), (.85 * W, .02 * H), eye, 50)[1:]])
    s.cord(th, 3.2, THREAD * .95, twill=2.0)
    s.cord([tip, tail], 8.5, STEEL, style='steel')
    return s.render(light=(-.5, -.65, .45), lamp=(.3, .15), dof=(.62, .45), vig=.55)


def band(Yp, y0, y1, soft=2.0):
    return smooth((Yp - y0) / soft) * smooth((y1 - Yp) / soft)


def img_sole(W=1100, H=360):
    s = Scene(W, H, 51, table=True)
    Yp = s.Y + 14 * s.u * np.sin(np.pi * s.X / W) - 10 * s.u * (s.X / W)
    bands = [(.05, .34, COGNAC * 1.15, 3.0, 'upper'), (.34, .52, TAN, 2.4, 'welt'), (.52, .68, BEIGE * .85, 2.0, 'mid'), (.68, .90, CHESTNUT, 2.2, 'out')]
    for k, (a, b, col, gr, nm) in enumerate(bands):
        m = band(Yp, a * H, b * H, 1.6)
        s.cast(m, 0, 5, 5, .7)
        s.leather(m, col, grain=gr, gloss=.4 if nm != 'out' else .25, height=3.5, bevel=3, seed=10 + k)
    # welt crimp + stitches
    crimp = .5 + .5 * np.sin(s.X / (5.0 * s.u) * 6.28)
    wm = band(Yp, .50 * H, .53 * H, 1.2)
    s.h += crimp * wm * 1.6 * s.u
    pts = np.c_[np.linspace(-10, W + 10, 200), .43 * H - 14 * s.u * np.sin(np.pi * np.linspace(-10, W + 10, 200) / W) + 10 * s.u * (np.linspace(-10, W + 10, 200) / W)]
    s.groove(pts, 3, 1.5, .3)
    s.stitch(pts, 17, 11, 2.0, THREAD, 1.8, tilt=.5)
    return s.render(light=(-.35, -.8, .45), lamp=(.35, .1), vig=.7, dof=None)


def shape_layers(s, items):
    """items: list of (pts, color, grain, height, gloss)."""
    for pts, col, gr, hgt, gl, seed in items:
        m = poly_mask(s.H, s.W, pts, 1.1)
        s.cast(m, 6, 9, 8, .75)
        s.leather(m, col, grain=gr, gloss=gl, height=hgt, bevel=3, seed=seed)


def img_stack(W=1400, H=675):
    s = Scene(W, H, 61)
    cols = [CHESTNUT, COGNAC, TAN, BEIGE * .8, COGNAC * .9, CHESTNUT * .8]
    items = []
    for i in range(6):
        items.append((sole_pts(.50 * W - i * 8 * s.u, .60 * H - i * 22 * s.u, .74 * W, .38 * H, -.10 + i * .004), cols[i], 2.4, 4.2, .3, 20 + i))
    shape_layers(s, items)
    top = items[-1][0]
    inner = sole_pts(.50 * W - 5 * 8 * s.u, .60 * H - 5 * 22 * s.u, .70 * W, .33 * H, -.10 + 5 * .004)
    s.stitch(inner, 18, 11, 2.0, THREAD, 1.6)
    return s.render(light=(-.5, -.6, .5), lamp=(.3, .2), dof=(.5, .5), vig=.65)


def img_insole(W=1200, H=440):
    s = Scene(W, H, 71)
    pts = sole_pts(.5 * W, .50 * H, .86 * W, .62 * H, -.07)
    m = poly_mask(H, W, pts, 1.1)
    s.cast(m, 8, 12, 10, .8)
    alb = SUEDE[None, None, :] * (.78 + .4 * fbm(H, W, 6, 3, 7))[..., None]
    s.paint(m, 1.4 * fbm(H, W, 220, 2, 9) + .6 * fbm(H, W, 40, 3, 4), alb, .06, 3.2, 3)
    inner = sole_pts(.5 * W, .50 * H, .82 * W, .57 * H, -.07)
    s.groove(inner, 3, 1.4, .3); s.stitch(inner, 17, 11, 1.9, THREAD * .95, 1.5)
    deb = poly_mask(H, W, sole_pts(.46 * W, .5 * H, .52 * W, .34 * H, -.07), 14)
    s.h -= 2.2 * deb * s.u; s.a *= (1 - .22 * deb[..., None])
    return s.render(light=(-.5, -.65, .5), lamp=(.3, .2), dof=(.5, .6), vig=.62)


def spool(s, cx, cy, R, color, seed=0):
    d = np.hypot(s.X - cx, s.Y - cy)
    R *= s.u
    m = np.clip((R - d) / 1.2, 0, 1)
    s.cast(m, 7, 10, 8, .75)
    wind = .5 + .5 * np.sin(d / (2.1 * s.u) * 6.28 + 3.0 * np.sin(np.arctan2(s.Y - cy, s.X - cx) * 3))
    hh = (R * .55 * np.sqrt(np.clip(1 - (d / R) ** 2 * .9, 0, 1)) + 1.4 * wind * s.u)
    alb = color[None, None, :] * (.62 + .5 * wind)[..., None]
    s.paint(m, hh, alb, .22, 0, 2)
    hole = np.clip((R * .16 - d) / 1.2, 0, 1)
    s.paint(hole, np.zeros_like(hole) - 1.5 * s.u, (np.ones((s.H, s.W, 3)) * .02), .1, 0, 1)


def img_threads(W=1400, H=554):
    s = Scene(W, H, 81)
    layout = [(.12, .50, 108, THREAD), (.30, .38, 92, TAN), (.47, .60, 112, CHESTNUT * 1.5), (.65, .40, 96, BEIGE), (.84, .58, 104, COGNAC * 1.2)]
    for x, y, R, c in layout: spool(s, x * W, y * H, R, c)
    s.cord([(.20 * W, .86 * H), (.60 * W, .12 * H)], 7, STEEL, style='steel')
    s.cord([(.46 * W, .90 * H), (.90 * W, .74 * H)], 6.5, STEEL, style='steel')
    return s.render(light=(-.5, -.55, .5), lamp=(.4, .15), dof=(.5, .7), vig=.7)


def swatch(s, cx, cy, w, h, ang, col, grain, gloss, seed, stitched=True):
    c, sn = np.cos(ang), np.sin(ang)
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    pts = [(cx + x * c - y * sn, cy + x * sn + y * c) for x, y in corners]
    m = poly_mask(s.H, s.W, pts, 1.0)
    s.cast(m, 6, 9, 8, .75)
    s.leather(m, col, grain=grain, gloss=gloss, height=3.0, bevel=3, seed=seed)
    if stitched:
        k = .86
        pts2 = [(cx + x * k * c - y * k * sn, cy + x * k * sn + y * k * c) for x, y in corners]; pts2.append(pts2[0])
        s.stitch(pts2, 17, 11, 1.9, THREAD, 1.5)


def img_wide(W=1400, H=741):
    s = Scene(W, H, 91)
    cols = [(TAN, 3.0, .45), (COGNAC, 3.0, .4), (CHESTNUT, 2.6, .35), (BLACK * 1.3, 2.6, .5), (SUEDE, 1.2, .06), (BEIGE * .85, 2.4, .3)]
    rng = np.random.default_rng(4)
    k = 0
    for r in range(2):
        for c in range(3):
            col, g, gl = cols[k]
            swatch(s, (.19 + c * .31) * W, (.27 + r * .46) * H, .26 * W, .38 * H, rng.uniform(-.06, .06), col, g, gl, 30 + k, stitched=(k % 2 == 0)); k += 1
    return s.render(light=(-.5, -.6, .5), lamp=(.35, .2), dof=(.5, .7), vig=.7)


def img_exploded(W=1400, H=708):
    s = Scene(W, H, 101)
    items = [
        (sole_pts(.30 * W, .62 * H, .50 * W, .26 * H, -.12), CHESTNUT, 2.4, 4.0, .28, 40),
        (sole_pts(.33 * W, .52 * H, .48 * W, .25 * H, -.12), TAN, 2.4, 3.2, .3, 41),
        (sole_pts(.70 * W, .34 * H, .46 * W, .25 * H, .10), SUEDE, 1.0, 3.0, .06, 42),
    ]
    shape_layers(s, items)
    s.stitch(sole_pts(.33 * W, .52 * H, .45 * W, .22 * H, -.12), 17, 11, 1.9, THREAD, 1.5)
    # heel stack
    for i in range(5):
        swatch(s, (.80 - i * .0) * W, (.72 - i * .02) * H, .12 * W, .09 * H, .1, [CHESTNUT, COGNAC][i % 2], 2.2, .3, 60 + i, False)
    # lace coil
    cx, cy = .58 * W, .73 * H
    th = np.linspace(0, 6 * np.pi, 160)
    coil = np.c_[cx + (14 + 10 * th) * s.u * .9 * np.cos(th) * .6, cy + (14 + 10 * th) * s.u * .9 * np.sin(th) * .38]
    s.cord(coil, 6, WAX * 1.6, twill=3.0)
    s.cord([(.52 * W, .14 * H), (.10 * W, .22 * H)], 6.5, STEEL, style='steel')
    return s.render(light=(-.5, -.55, .5), lamp=(.35, .2), dof=(.5, .7), vig=.68)


def img_about(W=720, H=900):
    s = Scene(W, H, 111)
    swatch(s, .36 * W, .26 * H, .46 * W, .26 * H, -.07, TAN, 3.0, .45, 70)
    swatch(s, .60 * W, .46 * H, .46 * W, .24 * H, .06, COGNAC, 3.0, .4, 71, False)
    swatch(s, .38 * W, .68 * H, .46 * W, .24 * H, -.04, CHESTNUT, 2.6, .35, 72)
    spool(s, .78 * W, .80 * H, 70, THREAD)
    spool(s, .20 * W, .90 * H, 52, BEIGE)
    s.cord([(.12 * W, .46 * H), (.55 * W, .90 * H)], 6, STEEL, style='steel')
    return s.render(light=(-.5, -.6, .5), lamp=(.3, .15), dof=(.45, .6), vig=.7)


def img_poster(W=1600, H=900):
    return img_toe(W, H, wide=True)


JOBS = dict(grain=img_grain, toe=img_toe, laces=img_laces, needle=img_needle, sole=img_sole, stack=img_stack,
            insole=img_insole, threads=img_threads, wide=img_wide, exploded=img_exploded, about=img_about, poster=img_poster)

if __name__ == '__main__':
    names = sys.argv[1:] or list(JOBS)
    for n in names:
        save(JOBS[n](), n)
