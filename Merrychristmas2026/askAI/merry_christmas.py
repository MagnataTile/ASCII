#!/usr/bin/env python3
"""
merry_christmas.py - arvore de Natal ASCII 3D que cresce(la ele), gira e acende.

Os galhos brotam de baixo para cima, depois vem o festao dourado, as bolas,
as luzinhas, os presentes, a estrela e por fim a mensagem FELIZ NATAL.
Roda em loop infinito: ao fim do ciclo a cena apaga e a arvore cresce de novo.

Fonte: video aleatorio do instagram de um criador @ nove nove nove dois zero zero seis zero um um nove
Requer: pip install numpy pygame-ce   (ou pygame)

Teclas: Espaco pausa · R reinicia · S captura de tela · M musica · Esc sai
"""
import os
import sys

import numpy as np
import pygame

TAU = 2 * np.pi

# scene (world units, y up)

GROUND = 0.0
Y0, Y1 = 0.32, 2.62                 # bottom and top of the foliage
R_BASE = 1.05                       # radius of the lowest whorl

# timeline (seconds)

T_GROW = (0.6, 6.2)
T_GARLAND = (6.2, 8.6)
T_BALLS = (7.8, 10.2)
T_LIGHTS = (8.5, 11.0)
T_GIFTS = (10.2, 12.4)
T_STAR = 12.8
T_MSG = 13.6
AUDIO = "https://cdn.pixabay.com/audio/2026/10/09/audio_a0a5d47856.mp3"   # musica de fundo (tocada pelo link)
AUDIO_TIME = 47.07                  # duracao da musica em segundos: o show dura isso e recomeca junto com ela
CYCLE = AUDIO_TIME                  # one full show, then it starts over (forever), in step with the music
FADE = 1.6                          # fade-out at the end of each cycle
ROT_SPEED = 0.31                    # rad/s, ~17.8 degrees per second

MESSAGE = "FELIZ NATAL"             # drawn big, as block letters made of characters
YEAR = "2026"                       # second line, same letter size as MESSAGE
NAME_DISPLAY = "MagnataTile"        # assinatura no canto inferior esquerdo: troque pelo seu nome
BANNER_ROWS = 6.0                   # letter height of the big message, in text rows

# palette (linear 0..1)

WHITE = (1.0, 1.0, 1.0)
GOLD = (1.0, 0.78, 0.28)
BAUBLE_GOLD = (1.0, 0.66, 0.18)
RED = (0.95, 0.13, 0.12)
GREEN_BOX = (0.22, 0.80, 0.32)
BLUE = (0.22, 0.38, 1.0)
SILVER = (0.86, 0.88, 0.96)
WARM = (1.0, 0.80, 0.42)

# what each point is made of; every kind has its own ramp of glyphs, dark -> bright

NEEDLE, SNOW, INNER, TRUNK, TINSEL, BALL, LIGHT, BOX, RIBBON = range(9)
RAMPS = {
    NEEDLE: ".-^vx*%&#W",
    SNOW: "-+x*%#@",
    INNER: ".,:-=",
    TRUNK: ".:|lH#",
    TINSEL: "-~=+*#@",
    BALL: ".oO0%&@",
    LIGHT: "+*o@@",
    BOX: ".:=x%#",
    RIBBON: "-+xX#",
}
LUT = np.zeros((len(RAMPS), 16), np.int32)
for _k, _s in RAMPS.items():
    LUT[_k] = [ord(_s[min(len(_s) - 1, i * len(_s) // 16)]) for i in range(16)]

BIAS = np.zeros(len(RAMPS))         # depth bias so small bright things win their cell
BIAS[[LIGHT, BALL, TINSEL, INNER]] = 0.07, 0.02, 0.03, -0.10
FLASHES = np.zeros(len(RAMPS), bool)  # kinds that burst white while they grow
FLASHES[[NEEDLE, SNOW, INNER, TRUNK, TINSEL]] = True

# camera

CAM_PITCH = 0.17
CAM_DIST = 8.0
YC = 1.22                           # world height at the screen anchor row
ROW_C = 0.40                        # anchor row as a fraction of the screen
LIGHT_DIR = np.array([-0.5, 0.55, 0.67]) / np.linalg.norm([-0.5, 0.55, 0.67])
HALF = (LIGHT_DIR + [0, 0, 1]) / np.linalg.norm(LIGHT_DIR + [0, 0, 1])


# helpers

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)


def rot_y(p, a):
    c, s = np.cos(a), np.sin(a)
    return np.stack([p[:, 0] * c - p[:, 2] * s, p[:, 1], p[:, 0] * s + p[:, 2] * c], 1)


def envelope(y):
    # cone radius at height y, with a faint ripple so the tiers read
    u = np.clip((np.asarray(y, float) - Y0) / (Y1 - Y0), 0, 1)
    return R_BASE * (1 - u) ** 1.05 * (1 + 0.04 * np.sin(u * TAU * 4))


def grow_time(y):
    return T_GROW[0] + (T_GROW[1] - T_GROW[0]) * np.clip((y - GROUND) / (Y1 - GROUND), 0, 1)


def to_view(v, rot):
    c, s = np.cos(rot), np.sin(rot)
    x = v[:, 0] * c - v[:, 2] * s
    z = v[:, 0] * s + v[:, 2] * c
    cp, sp = np.cos(CAM_PITCH), np.sin(CAM_PITCH)
    return np.stack([x, v[:, 1] * cp - z * sp, v[:, 1] * sp + z * cp], 1)


def project(p, cols, rows, aspect, rot=0.0):
    v = to_view(p - [0, YC, 0], rot)
    f = CAM_DIST / (CAM_DIST - v[:, 2])
    scale = rows * 0.21
    return cols / 2 + v[:, 0] * f * scale * aspect, rows * ROW_C - v[:, 1] * f * scale, v[:, 2]


class Parts:
    """Point cloud accumulator: position, normal, colour, kind, birth time..."""

    def __init__(self):
        self.chunks = []

    def add(self, p, n, col, kind, t0, phase=0.0, base=1.0, centre=None, pop=False):
        p = np.asarray(p, float).reshape(-1, 3)
        m = len(p)

        def full(v, shape):
            return np.broadcast_to(np.asarray(v, float), shape).copy()

        self.chunks.append(dict(
            p=p, n=unit(full(n, (m, 3))), c=full(col, (m, 3)),
            k=full(kind, (m,)).astype(np.int32), t0=full(t0, (m,)), ph=full(phase, (m,)),
            b=full(base, (m,)), cen=p.copy() if centre is None else full(centre, (m, 3)),
            pop=np.broadcast_to(pop, (m,)).copy()))

    def done(self, rng):
        d = {key: np.concatenate([c[key] for c in self.chunks]) for key in self.chunks[0]}
        d["jit"] = rng.random(len(d["p"]))
        for key in ("p", "n", "c", "t0", "ph", "b", "cen", "jit"):
            d[key] = d[key].astype(np.float32)
        return d


def build():
    rng = np.random.default_rng(7)
    q = Parts()
    tips = []

    # needles: whorls of drooping branches, denser and longer at the bottom
    for i, y in enumerate(np.linspace(Y0 + 0.02, Y1 - 0.06, 24)):
        u = (y - Y0) / (Y1 - Y0)
        R = envelope(y) + 0.05
        nb = int(6 + 12 * (1 - u))
        for b in range(nb):
            a = TAU * (b + 0.5 * rng.random()) / nb + i * 0.77
            L = R * rng.uniform(0.85, 1.08)
            n = int(25 + 270 * L)
            s = rng.random(n) ** 0.65                               # 0 trunk .. 1 tip
            d = np.array([np.cos(a), 0, np.sin(a)])
            side = np.array([-np.sin(a), 0, np.cos(a)])
            spread = (0.025 + 0.09 * s) * rng.normal(0, 1, n)
            sl = s * L
            p = d * sl[:, None] + side * spread[:, None]
            p[:, 1] = y + 0.06 * sl - 0.20 * sl ** 2 + 0.012 * rng.normal(0, 1, n)
            nrm = unit(d * 0.75 + np.array([0, 0.55, 0]) + 0.3 * rng.normal(0, 1, (n, 3)))
            ao = (0.30 + 0.70 * s ** 1.3) * rng.uniform(0.75, 1.1, n)
            snow = (s > 0.5) & (rng.random(n) < 0.06 * s) & (nrm[:, 1] > 0.5)
            green = np.array([0.10, 0.48, 0.18]) + np.array([0.12, 0.38, 0.14]) * rng.random((n, 1))
            q.add(p, nrm, np.where(snow[:, None], WHITE, green), np.where(snow, SNOW, NEEDLE),
                  grow_time(y) + 0.45 * s, base=np.where(snow, 1.0, ao))
            tl = 0.93 * L
            tips.append(d * tl + [0, y + 0.06 * tl - 0.20 * tl ** 2, 0])

    # dark inner cone so nothing shows through the branches
    m = 6000
    y = Y0 + (Y1 - 0.1 - Y0) * rng.random(m)
    phi = rng.random(m) * TAU
    out = np.stack([np.cos(phi), np.full(m, 0.3), np.sin(phi)], 1)
    r = 0.6 * envelope(y)
    q.add(np.stack([np.cos(phi) * r, y, np.sin(phi) * r], 1), out, (0.10, 0.42, 0.16), INNER,
          grow_time(y), base=0.75)

    # trunk
    m = 1500
    y, phi = GROUND + rng.random(m) * (Y0 + 0.1 - GROUND), rng.random(m) * TAU
    n = np.stack([np.cos(phi), np.zeros(m), np.sin(phi)], 1)
    bark = 0.75 + 0.25 * np.sign(np.sin(phi * 30 + y * 5))
    q.add(n * 0.08 + np.stack([np.zeros(m), y, np.zeros(m)], 1), n,
          np.array([0.42, 0.27, 0.14]) * bark[:, None], TRUNK, grow_time(y))

    # tinsel garland: a gold spiral drooping in swags between hanging points
    u = np.linspace(0, 1, 2600)
    turns = 3.6
    yg = Y0 + 0.15 + (Y1 - Y0 - 0.45) * u
    ag = TAU * turns * u + 0.9
    swag = -0.07 * np.abs(np.sin(ag * 2.5))
    rg = envelope(yg) + 0.07
    line = np.stack([np.cos(ag) * rg, yg + swag, np.sin(ag) * rg], 1)
    for _ in range(4):
        nrm = np.stack([np.cos(ag), np.full_like(u, 0.3), np.sin(ag)], 1) + 0.6 * rng.normal(0, 1, (len(u), 3))
        q.add(line + 0.016 * rng.normal(0, 1, line.shape), nrm, GOLD, TINSEL,
              T_GARLAND[0] + u * (T_GARLAND[1] - T_GARLAND[0]), phase=rng.random(len(u)) * TAU)

    # baubles: mostly red, some gold and silver
    sphere = fib_sphere(160)
    for k in range(34):
        u = (k + rng.random()) / 34
        y = Y0 + 0.08 + u * (Y1 - Y0 - 0.38)
        a = k * 2.39996 + rng.random() * 0.4
        c = np.array([np.cos(a), 0, np.sin(a)]) * envelope(y) + [0, y - 0.03, 0]
        rad = 0.058 + 0.032 * (1 - u)
        col = RED if k % 2 == 0 else (BAUBLE_GOLD if k % 6 else SILVER)  # half red, the rest gold/silver
        q.add(c + sphere * rad, sphere, col, BALL, T_BALLS[0] + rng.random() * (T_BALLS[1] - T_BALLS[0]),
              centre=c, pop=True)

    # fairy lights on the branch tips
    tips = np.array(tips)
    pick = rng.choice(len(tips), min(170, len(tips)), replace=False)
    bulb = fib_sphere(16)
    light_t0 = T_LIGHTS[0] + rng.random(len(pick)) * (T_LIGHTS[1] - T_LIGHTS[0])
    for c, t0 in zip(tips[pick], light_t0):
        col = WARM if rng.random() < 0.8 else (1.0, 0.95, 0.85)
        q.add(c + bulb * 0.03, bulb, col, LIGHT, t0, centre=c, phase=rng.random() * TAU, pop=True)

    # presents: boxes with a crossed ribbon and a bow
    gifts = [(0.3, 1.12, 0.30, 0.24, 0.28, 0.4, RED, GOLD), (1.5, 1.18, 0.26, 0.30, 0.22, 1.1, SILVER, RED),
             (2.6, 1.10, 0.34, 0.18, 0.26, 1.0, GOLD, RED), (3.7, 1.18, 0.24, 0.26, 0.26, 0.7, BLUE, SILVER),
             (4.7, 1.12, 0.30, 0.22, 0.30, 0.1, GREEN_BOX, RED), (5.6, 1.22, 0.22, 0.20, 0.22, 0.2, RED, WHITE)]
    loop = fib_sphere(80)
    for g, (a, dist, w, h, d, rot, body, rib) in enumerate(gifts):
        areas = np.array([w * d, h * d, h * d, h * w, h * w])
        m = int(5000 * areas.sum())
        face = rng.choice(5, m, p=areas / areas.sum())
        fa, fb = rng.random(m) - 0.5, rng.random(m) - 0.5
        lx = np.select([face == 0, face == 1, face == 2], [fa * w, np.full(m, w / 2), np.full(m, -w / 2)], fa * w)
        ly = np.where(face == 0, h, (fb + 0.5) * h)
        lz = np.select([face == 0, face <= 2, face == 3], [fb * d, fa * d, np.full(m, d / 2)], np.full(m, -d / 2))
        n = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1.0]])[face]
        on_x, on_z = np.abs(lx) < 0.022, np.abs(lz) < 0.022
        ribbon = np.select([face == 0, face <= 2], [on_x | on_z, on_z], on_x)
        bow = np.concatenate([loop * [0.05, 0.03, 0.025] + [sx * 0.045, h + 0.025, 0] for sx in (-1, 1)])
        p = np.concatenate([np.stack([lx, ly, lz], 1), bow])
        nrm = np.concatenate([n, loop, loop])
        rib_mask = np.concatenate([ribbon, np.ones(len(bow), bool)])
        cen = np.array([np.cos(a) * dist, GROUND, np.sin(a) * dist])
        q.add(rot_y(p, rot) + cen, rot_y(nrm, rot), np.where(rib_mask[:, None], rib, body),
              np.where(rib_mask, RIBBON, BOX), T_GIFTS[0] + g / len(gifts) * (T_GIFTS[1] - T_GIFTS[0]),
              centre=cen, pop=True)

    P = q.done(rng)
    P["light_t0"] = light_t0
    return P


# sky: falling snow over a dotted ground plane

class Sky:
    def __init__(self, cols, rows, n=430, seed=3):
        rng = np.random.default_rng(seed)
        self.cols, self.rows = cols, rows
        self.x, self.y = rng.random(n) * cols, rng.random(n) * rows
        self.v = rng.uniform(1.2, 4.5, n)
        self.ph = rng.random(n) * TAU
        self.size = rng.random(n)
        self.front = rng.random(n) < 0.12
        self.glyph = np.where(self.size > 0.62, ord("+"), np.where(self.size > 0.3, ord("*"), ord(".")))
        self.ground = np.zeros((rows, cols), np.int32)
        self.gcol = np.zeros((rows, cols, 3), np.float32)
        horizon = int(rows * 0.49)
        for r in range(horizon, rows):
            u = (r - horizon) / (rows - horizon)
            keep = rng.random(cols) < 0.35 + 0.45 * u
            ch = np.where(rng.random(cols) < 0.06 + 0.1 * u, ord("="), ord("-" if r % 2 else "."))
            self.ground[r] = np.where(keep, ch, 0)
            self.gcol[r] = np.array([0.55, 0.60, 0.78]) * (0.16 + 0.14 * u)

    def step(self, dt, t):
        self.y += self.v * dt
        self.x += np.sin(t * 0.8 + self.ph) * 0.7 * dt
        wrap = self.y >= self.rows
        self.y[wrap] -= self.rows
        self.x %= self.cols

    def draw(self, char, col, t, front):
        if not front:
            char[:] = self.ground
            col[:] = self.gcol
        m = self.front == front
        ix, iy = self.x[m].astype(int), self.y[m].astype(int)
        lum = (0.35 + 0.65 * self.size[m]) * (0.7 + 0.3 * np.sin(t * 4 + self.ph[m]))
        char[iy, ix] = self.glyph[m]
        col[iy, ix] = np.array([0.85, 0.90, 1.0]) * lum[:, None]


# drawing into the character grid

def put(char, col, y, x, ch, tint, power):
    char[y, x] = ord(ch)
    col[y, x] = np.asarray(tint, np.float32) * power


def in_poly(px, py, poly):
    res = np.zeros(px.shape, bool)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        res ^= ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1)
    return res


def star(char, col, cx, cy, size, t, power, cell, flash=0.0):
    # thin rays first, then a filled five-pointed star on top;
    # when it lights up it bursts white with long rays, then settles to gold
    tint = np.array(GOLD) + (1 - np.array(GOLD)) * flash
    power = power * (1 + 1.6 * flash)
    for k in range(8):
        ang = k * TAU / 8 + 0.10 * np.sin(t * 0.7)
        if np.sin(ang) < -0.3:                                      # no rays down into the tree
            continue
        reach = size * (3.0 + 0.8 * np.sin(t * 3 + k * 1.3)) * (1.0 if k % 2 == 0 else 0.6) * (1 + 2.5 * flash)
        ch = "-|/\\"[[0, 2, 1, 3][int(round((np.degrees(ang) % 180) / 45)) % 4]]
        for s in np.arange(size * 1.1, reach, 0.5):
            y, x = int(round(cy - np.sin(ang) * s)), int(round(cx + np.cos(ang) * s * cell))
            if 0 <= y < char.shape[0] and 0 <= x < char.shape[1] and col[y, x].max() < 0.3:
                put(char, col, y, x, ch, tint, power * 0.6 * (1 - s / reach))
    poly = [(cx + np.cos(a) * r * cell, cy - np.sin(a) * r)
            for i in range(10)
            for a, r in [(np.pi / 2 + i * TAU / 10, size if i % 2 == 0 else size * 0.45)]]
    ys, xs = np.mgrid[int(cy - size) - 1:int(cy + size) + 2, int(cx - size * cell) - 1:int(cx + size * cell) + 2]
    inside = in_poly(xs + 0.0, ys + 0.0, poly)
    for y, x in zip(ys[inside], xs[inside]):
        if 0 <= y < char.shape[0] and 0 <= x < char.shape[1]:
            rr = np.hypot((x - cx) / cell, y - cy) / size
            put(char, col, y, x, "@" if rr < 0.35 else ("#" if rr < 0.75 else "*"), tint, power * (1.25 - 0.4 * rr))


_BANNER = {}


def banner(cols):
    # MESSAGE and YEAR rasterised with a heavy font at one shared size, as coverage per text cell
    # (rows x cols) plus which letter owns each column, so every letter gets its own colour
    if cols not in _BANNER:
        pygame.font.init()
        font = pygame.font.SysFont("arialblack,impact,arial", 160, bold=True)
        k = None
        lines = []
        for text in (MESSAGE, YEAR):
            surf = font.render(text, True, (255, 255, 255))
            box = surf.get_bounding_rect()
            tw, th = box.size
            if k is None:                                           # sized by the first line, kept for both
                k = min(BANNER_ROWS * CH / th, 0.86 * (cols - 4) * CW / tw)
            w, h = max(1, round(tw * k / CW)), max(1, round(th * k / CH))
            small = pygame.transform.smoothscale(surf.subsurface(box), (w * 4, h * 4))
            cov = (pygame.surfarray.array_alpha(small).T / 255).reshape(h, 4, w, 4).mean((1, 3))
            edges = (np.array([font.size(text[:i + 1])[0] for i in range(len(text))]) - box.x) * k / CW
            letter = np.minimum(np.searchsorted(edges, np.arange(w) + 0.5), len(text) - 1)
            lines.append((cov.astype(np.float32), letter))
        _BANNER[cols] = lines
    return _BANNER[cols]


def render(P, t, rot, cols, rows, aspect, sky):
    char = np.zeros((rows, cols), np.int32)
    col = np.zeros((rows, cols, 3), np.float32)
    sky.draw(char, col, t, front=False)
    fade = float(np.clip((CYCLE - t) / FADE, 0, 1))
    info = {"rotation": np.degrees(rot) % 360, "lights": int((t >= P["light_t0"]).sum())}

    idx = np.nonzero(t >= P["t0"])[0]
    if len(idx):
        whole = len(idx) == len(P["t0"])                          # skip the gathers once all is grown
        pick = (lambda a: a) if whole else (lambda a: a[idx])
        p, k, age = pick(P["p"]), pick(P["k"]), t - pick(P["t0"])
        grow = np.nonzero(pick(P["pop"]) & (age < 0.35))[0]
        if len(grow):                                               # ease-out-back scale from the centre
            e = age[grow] / 0.35 - 1
            sc = 1 + 2.70158 * e ** 3 + 1.70158 * e ** 2
            cen = pick(P["cen"])[grow]
            p = p.copy()
            p[grow] = cen + (p[grow] - cen) * sc[:, None]
        sx, sy, dep = project(p, cols, rows, aspect, rot)
        cx, cy = np.round(sx).astype(np.int32), np.round(sy).astype(np.int32)
        ok = np.nonzero((cx >= 0) & (cx < cols) & (cy >= 0) & (cy < rows))[0]
        cell = cy[ok] * cols + cx[ok]
        key = cell + np.clip((dep[ok] + BIAS[k[ok]] + 4) / 8, 0, 0.999)
        order = np.argsort(key)                                     # nearest point owns its cell
        cs = cell[order]
        win = ok[order[np.r_[cs[1:] != cs[:-1], True]]]             # winners only from here on
        cell = cy[win] * cols + cx[win]
        src = win if whole else idx[win]
        k, age = k[win], age[win]

        nv = to_view(P["n"][src], rot)
        lam = np.clip(nv @ LIGHT_DIR, 0, 1)
        shade = P["b"][src] * (0.22 + 0.78 * lam) * (0.75 + 0.25 * np.clip(nv[:, 2], 0, 1))
        shade += 0.14 * (P["jit"][src] - 0.5)
        c = P["c"][src] * 1.0
        ph = P["ph"][src]

        m = k == TINSEL
        shade[m] += 0.6 * np.maximum(0, np.sin(t * 5 + ph[m])) ** 12
        m = k == BALL
        spec = np.clip(nv[m] @ HALF, 0, 1) ** 24
        shade[m] += 0.8 * spec
        m = k == LIGHT
        shade[m] = 0.55 + 0.45 * (0.5 + 0.5 * np.sin(t * 3.1 + ph[m]))
        m = k == INNER
        shade[m] *= 0.45

        c *= (0.30 + 0.95 * np.clip(shade, 0, 1.4))[:, None]
        c[k == BALL] += 0.8 * spec[:, None]
        c[k == LIGHT] *= 1.35
        fl = np.clip(1 - age / 0.30, 0, 1) * FLASHES[k]             # growth front bursts white
        c = c * (1 - fl[:, None]) + 1.1 * fl[:, None]
        g = LUT[k, np.clip(shade * 15.99, 0, 15).astype(int)]
        g[fl > 0.35] = ord("#")
        char.flat[cell] = g
        col.reshape(-1, 3)[cell] = c * fade

    sky.draw(char, col, t, front=True)

    if t > T_STAR:
        on = float(np.clip((t - T_STAR) / 0.6, 0, 1))
        flash = float(np.clip(1 - (t - T_STAR) / 1.0, 0, 1) ** 2)
        tx, ty, _ = project(np.array([[0, Y1 + 0.09, 0]]), cols, rows, aspect)
        star(char, col, int(tx[0]), int(ty[0]), (2.0 + 2.2 * on) * (1 + 0.05 * np.sin(t * 2.3)), t,
             fade * (0.5 + 0.6 * on), aspect, flash=flash * fade)

    if t > T_MSG:
        typed = int((t - T_MSG) * 9)                                # letters typed so far
        y, first = rows - 17, 0                                     # just below the presents
        for line, (cov, letter) in enumerate(banner(cols)):
            h, w = cov.shape
            x0 = (cols - w) // 2
            vis = np.nonzero(letter < typed - first)[0]
            if len(vis):
                char[y - 1:y + h + 1, max(x0 - 3, 0):min(cols, x0 + vis[-1] + 4)] = 0   # clear a band
                sub = cov[:, vis]
                ys, xs = np.nonzero(sub > 0.12)
                i = first + letter[vis[xs]]                         # letter number across both lines
                a = sub[ys, xs]
                if line == 0:
                    tints = np.array([RED, GOLD, GREEN_BOX])[(i + int(t * 1.5)) % 3]
                else:
                    tints = np.broadcast_to(np.array(GOLD), (len(i), 3))
                power = (0.45 + 0.7 * a) * (0.85 + 0.25 * np.sin(t * 5 + i + ys * 0.4)) * fade
                pop = np.clip(1 - (t - T_MSG - i / 9) / 0.25, 0, 1)[:, None]   # each new letter flashes white
                char[y + ys, x0 + vis[xs]] = np.array([ord(g) for g in ".:+*#@"])[np.minimum((a * 6).astype(int), 5)]
                col[y + ys, x0 + vis[xs]] = (tints * (1 - pop) + pop) * power[:, None]
            first += len(MESSAGE) if line == 0 else 0
            y += h + 1
    info["letters"] = int((char > 0).sum())
    return char, col, info


# window

TITLE = "Arvore de Natal ASCII"
CW, CH, FONT_PX = 9, 14, 15
COLS, ROWS = 88, 74
W, H = COLS * CW, ROWS * CH
MONO = "consolas,menlo,dejavusansmono,couriernew,monospace"
HINT = "Espaco pausa · R reinicia · S captura · M musica · Esc sai"
DIM, ACCENT, SIGN = (96, 104, 122), (255, 196, 90), (176, 168, 196)
BG_RGB = (7, 9, 17)


class Renderer:
    def __init__(self, cols=COLS, rows=ROWS, cw=CW, ch=CH, font_px=FONT_PX, hud_px=15):
        self.cols, self.rows, self.cw, self.ch = cols, rows, cw, ch
        self.w, self.h = cols * cw, rows * ch
        font = pygame.font.SysFont(MONO, font_px)
        atlas = []
        for code in range(32, 127):
            s = pygame.Surface((cw, ch))
            g = font.render(chr(code), True, (255, 255, 255))
            s.blit(g, ((cw - g.get_width()) // 2, (ch - g.get_height()) // 2))
            atlas.append(pygame.surfarray.array3d(s).transpose(1, 0, 2))
        self.atlas = np.stack(atlas)[..., 0].astype(np.float32) / 255  # glyph, CH, CW
        self.atlas_t = np.ascontiguousarray(self.atlas.transpose(0, 2, 1) * 255).astype(np.uint8)  # glyph, CW, CH
        self.art = pygame.Surface((self.w, self.h), depth=32)
        self.tint = pygame.Surface((cols, rows))
        self.glow = pygame.Surface((cols, rows))
        self.hud = pygame.font.SysFont(MONO, hud_px)

    def draw(self, screen, char, col, info, t, paused):
        # glyph masks as a grey image, multiplied by the cell colours scaled up (nearest)
        code = np.where(char > 0, np.clip(char - 32, 0, 94), 0).T             # cols, rows
        mask = self.atlas_t[code].transpose(0, 2, 1, 3).reshape(self.w, self.h)
        pygame.surfarray.blit_array(self.art, mask.astype(np.uint32) * 0x010101)
        pygame.surfarray.blit_array(self.tint, np.clip(col * 255, 0, 255).astype(np.uint8).transpose(1, 0, 2))
        self.art.blit(pygame.transform.scale(self.tint, (self.w, self.h)), (0, 0),
                      special_flags=pygame.BLEND_MULT)
        screen.fill(BG_RGB)
        screen.blit(self.art, (0, 0), special_flags=pygame.BLEND_ADD)

        # bloom: blur the bright cells at grid resolution and add them back, scaled up
        g = np.maximum(col * (char > 0)[..., None] - 0.30, 0) ** 1.3
        for _ in range(3):
            p = np.pad(g, ((1, 1), (1, 1), (0, 0)), mode="edge")
            g = (p[1:-1, 1:-1] * 2 + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 6
        pygame.surfarray.blit_array(self.glow, np.clip(g * 130, 0, 255).astype(np.uint8).transpose(1, 0, 2))
        screen.blit(pygame.transform.smoothscale(self.glow, (self.w, self.h)), (0, 0),
                    special_flags=pygame.BLEND_ADD)

        # hud
        lh = self.hud.get_linesize()
        txt = self.hud.render(f"t {t:05.2f}s", True, ACCENT)
        screen.blit(txt, (self.w - txt.get_width() - 12, 10))
        pygame.draw.circle(screen, ACCENT, (self.w - txt.get_width() - 22, 10 + lh // 2), 4)
        y = self.h - 3 * lh - 12
        screen.blit(self.hud.render(NAME_DISPLAY, True, SIGN), (14, y))
        screen.blit(self.hud.render(f"rotation {info['rotation']:06.2f}°     lights {info['lights']:4d}",
                                    True, DIM), (14, y + lh + 6))
        screen.blit(self.hud.render(f"letters {info['letters']:,}", True, DIM), (14, y + 2 * lh + 6))
        hint = self.hud.render("PAUSADO · " + HINT if paused else HINT, True, ACCENT if paused else DIM)
        screen.blit(hint, (self.w - hint.get_width() - 12, self.h - lh - 8))


def fetch_audio(box):
    # streams AUDIO into memory (nothing is saved to disk); stays silent if offline
    try:
        import io
        import urllib.request
        req = urllib.request.Request(AUDIO, headers={"User-Agent": "Mozilla/5.0"})
        box["data"] = io.BytesIO(urllib.request.urlopen(req, timeout=15).read())
    except Exception as e:
        print("musica indisponivel:", e)


def main():
    shot = None
    if "--shot" in sys.argv:                    # --shot SECONDS FILE.png : render one frame and exit
        i = sys.argv.index("--shot")
        shot = float(sys.argv[i + 1]), sys.argv[i + 2]
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    if shot:
        window = None
        screen = pygame.display.set_mode((W, H))
    else:
        # the art is drawn at full size off-screen and scaled to fit the monitor;
        # the window can also be resized or maximized freely
        dw, dh = pygame.display.get_desktop_sizes()[0]
        k = min(1.0, 0.88 * dh / H, 0.95 * dw / W)
        window = pygame.display.set_mode((int(W * k), int(H * k)), pygame.RESIZABLE)
        screen = pygame.Surface((W, H))
    pygame.display.set_caption(TITLE)
    rend = Renderer()
    P = build()
    sky = Sky(COLS, ROWS)
    aspect = CH / CW
    clock = pygame.time.Clock()
    t, paused, shots = 0.0, False, 0
    music, muted = {}, False
    if not shot:
        import threading
        threading.Thread(target=fetch_audio, args=(music,), daemon=True).start()

    if shot:
        for k in range(int(shot[0] * 30)):
            sky.step(1 / 30, k / 30)
        char, col, info = render(P, shot[0] % CYCLE, ROT_SPEED * shot[0], COLS, ROWS, aspect, sky)
        rend.draw(screen, char, col, info, shot[0] % CYCLE, False)
        pygame.image.save(screen, shot[1])
        return

    while True:                                 # infinite loop: the show repeats every CYCLE seconds
        dt = clock.tick(30) / 1000
        if "data" in music and "on" not in music:   # music arrived: start it and restart the show with it
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(music["data"], "mp3")
                pygame.mixer.music.play(-1)
                music["on"] = True
                t = 0.0
                if paused:
                    pygame.mixer.music.pause()
            except Exception as e:
                print("musica indisponivel:", e)
                music["on"] = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                return
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                    if music.get("on"):
                        (pygame.mixer.music.pause if paused else pygame.mixer.music.unpause)()
                elif ev.key == pygame.K_r:
                    t = 0.0
                    if music.get("on"):
                        pygame.mixer.music.play(-1)
                elif ev.key == pygame.K_m and music.get("on"):
                    muted = not muted
                    pygame.mixer.music.set_volume(0 if muted else 1)
                elif ev.key == pygame.K_s:
                    shots += 1
                    pygame.image.save(screen, f"arvore_natal_{shots:02d}.png")
        if not paused:
            t += dt
            sky.step(dt, t)
        tc = t % CYCLE
        char, col, info = render(P, tc, ROT_SPEED * t, COLS, ROWS, aspect, sky)
        rend.draw(screen, char, col, info, tc, paused)
        ww, wh = window.get_size()
        k = min(ww / W, wh / H)
        fit = (max(1, int(W * k)), max(1, int(H * k)))
        window.fill(BG_RGB)
        window.blit(pygame.transform.smoothscale(screen, fit), ((ww - fit[0]) // 2, (wh - fit[1]) // 2))
        pygame.display.flip()


if __name__ == "__main__":
    main()
