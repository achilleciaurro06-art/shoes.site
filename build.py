import re

src = open('src/index.src.html').read()

# --- small source cleanups -------------------------------------------------
def rep(a, b):
    global src
    if a in src:
        src = src.replace(a, b)

rep("$('#hero').classList.add('x');\n/* the 'is-ready' class lives on .hero's ancestor chain for CSS (.is-ready .hero__...) */\n", "")
rep("$$('.pj [data-pj]').length; ", "")
rep("const t = s.offsetTop + (s.offsetParent && s.offsetParent !== document.body ? s.offsetParent.offsetTop : 0); ", "")
rep("let lastY = -1, lastDir = 0;", "let lastY = -1;")
rep("    lastDir = y - lastY; lastY = y;", "    lastY = y;")
i = src.find("    if (!menuOpen) nav.classList.toggle('is-hidden'")
j = src.find("    // theme under nav")
if i != -1 and j != -1:
    src = src[:i] + "    if (!menuOpen) { if (y > vh * .5 && y > lastY + 2) nav.classList.add('is-hidden'); else if (y < lastY - 2 || y <= vh * .5) nav.classList.remove('is-hidden'); }\n" + src[j:]
rep("if (Math.abs(scrollY - cur) > 3 && !animating) target = scrollY;", "if (Math.abs(scrollY - cur) > 3 && !animating) { cur = target = scrollY; }")
open('src/index.src.html', 'w').write(src)


# --- technical drawing (inline SVG) ----------------------------------------
def tech(mode):
    d = ' data-draw' if mode == 'draw' else ''
    late = 'late' if mode == 'draw' else ''
    return f'''<svg class="tech" viewBox="0 0 980 410" role="img" aria-hidden="true" preserveAspectRatio="xMidYMid meet">
<line class="dim" x1="0" y1="324" x2="980" y2="324" stroke-dasharray="2 7"/>
<path{d} d="M46 300C150 314 330 306 520 292C650 282 760 296 868 300"/>
<path{d} d="M40 318C150 334 330 328 520 314C650 304 760 320 872 322"/>
<path{d} d="M40 318C30 310 36 304 46 300"/>
<path{d} d="M742 322V348H874L878 322"/>
<path{d} d="M46 300C34 268 70 238 140 228C230 214 310 202 392 172C430 158 470 138 520 114C560 106 640 124 720 130C800 136 852 148 872 174C890 204 886 264 868 300"/>
<path{d} d="M140 228C170 250 176 280 158 304"/>
<path{d} d="M770 138C758 200 758 262 770 300"/>
<path{d} class="soft" d="M520 114C560 130 640 146 720 150C790 154 840 164 862 184"/>
<path{d} d="M506 134C524 106 556 100 552 120C548 136 524 128 506 134C520 150 530 168 524 184"/>
<g class="{late}">
<path class="dots" d="M126 232C154 254 160 280 146 302"/>
<path class="stitch-l" d="M60 306C150 320 330 312 520 298C650 288 760 302 866 306"/>
<path class="stitch-l" d="M200 252C300 246 400 234 470 192"/>
<circle cx="418" cy="174" r="3"/><circle cx="440" cy="164" r="3"/><circle cx="462" cy="154" r="3"/><circle cx="484" cy="144" r="3"/><circle cx="506" cy="134" r="3"/>
<path class="dim" d="M40 372H872M40 362V382M872 362V382"/>
<path class="dim" d="M936 114V322M926 114H946M926 322H946"/>
<text x="456" y="398" text-anchor="middle">L — Last length</text>
<text x="964" y="226" transform="rotate(90 964 226)" text-anchor="middle">H — Heel height</text>
<text x="40" y="60">Fig. 02 — Oxford, cap toe, side elevation</text>
</g>
</svg>'''


out = (src.replace('{{TECH_DRAW}}', tech('draw'))
          .replace('{{TECH_TILE}}', tech('tile'))
          .replace('{{TECH_FOOT}}', tech('foot')))
out = re.sub(r'\{\{img:(\w+)\}\}', lambda m: f'img/{m.group(1)}.jpg', out)
open('index.html', 'w').write(out)
print('built', len(out))
