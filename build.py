"""Builds index.html from src/index.src.html + src/style.css.

  python3 build.py

- {{STYLE}}        -> contents of src/style.css
- {{img:name}}     -> img/name.jpg
- {{TECH_*}}       -> inline SVG technical drawing of a running shoe
"""
import re

src = open('src/index.src.html').read()
css = open('src/style.css').read()


def tech(mode):
    d = ' data-draw' if mode == 'draw' else ''
    late = 'late' if mode == 'draw' else ''
    return f'''<svg class="tech" viewBox="0 0 980 410" role="img" aria-hidden="true" preserveAspectRatio="xMidYMid meet">
<line class="dim" x1="0" y1="338" x2="980" y2="338" stroke-dasharray="2 7"/>
<path{d} d="M44 332C120 350 300 350 470 344C640 338 780 342 880 328"/>
<path{d} d="M48 322C130 336 300 336 470 330C640 324 780 326 874 312"/>
<path{d} d="M56 296C140 310 320 314 500 304C640 296 760 296 862 276"/>
<path{d} d="M44 332C30 322 32 304 56 296"/>
<path{d} d="M880 328C896 312 894 290 862 276"/>
<path{d} d="M56 296C48 258 92 232 152 218C228 200 300 194 358 170C398 152 440 128 492 100C524 88 566 92 604 106C650 122 704 126 760 120C832 114 872 150 872 196C872 236 872 256 862 276"/>
<path{d} class="soft" d="M492 100C520 108 566 118 604 128C650 142 710 146 760 140C820 134 852 150 860 180"/>
<path{d} d="M152 218C180 236 184 262 170 296"/>
<path{d} class="acc" d="M60 310C200 324 420 326 600 312C720 304 800 296 866 288"/>
<path{d} d="M806 122C808 98 820 86 838 84C844 98 840 114 832 126"/>
<g class="{late}">
<path class="dots" d="M196 236C240 224 290 216 340 200"/>
<path class="dots" d="M214 252C262 240 318 232 368 214"/>
<path class="dots" d="M66 304C150 318 330 322 500 312C640 304 760 304 856 284"/>
<path class="stitch-l" d="M60 340C140 356 300 356 470 350C640 344 780 348 870 334"/>
<path class="soft" d="M372 182C410 164 452 140 500 112"/>
<path class="soft" d="M392 174L436 146M414 162L458 132M436 150L480 120"/>
<circle cx="392" cy="166" r="2.6"/><circle cx="414" cy="154" r="2.6"/><circle cx="436" cy="142" r="2.6"/><circle cx="458" cy="129" r="2.6"/><circle cx="480" cy="117" r="2.6"/>
<path class="dim" d="M40 380H874M40 370V390M874 370V390"/>
<path class="dim" d="M936 84V338M926 84H946M926 338H946"/>
<path class="acc" d="M640 300L702 252M520 322L560 392M250 330L214 392"/>
<text x="708" y="250">01 Foam</text><text x="566" y="404">02 Plate</text><text x="130" y="404">03 Midsole / 04 Rubber</text>
<text x="457" y="402" text-anchor="middle" opacity="0">.</text>
<text x="964" y="214" transform="rotate(90 964 214)" text-anchor="middle">H — Stack height</text>
<text x="40" y="52">Fig. 02 — Running shoe, side elevation</text>
</g>
</svg>'''


out = (src.replace('{{STYLE}}', css)
          .replace('{{TECH_DRAW}}', tech('draw'))
          .replace('{{TECH_TILE}}', tech('tile'))
          .replace('{{TECH_FOOT}}', tech('foot')))
out = re.sub(r'\{\{img:(\w+)\}\}', lambda m: f'img/{m.group(1)}.jpg', out)
open('index.html', 'w').write(out)
print('built', len(out))
