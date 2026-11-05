"""Draw the Rail 3D icon: a railway track in perspective, point-cloud textured.

Two steel rails converge towards a vanishing point over brown sleepers and
a ground band, with a magenta catenary mast and an orange wire: the classes
the plugin writes, in the plugin's own colours. Rendered at 2048 px with
PIL, then downsampled to the toolbar and listing sizes. Transparent edges.

    python tools/make_icon.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

PLUGIN = Path(__file__).resolve().parents[1] / "Rail3D"
ASSETS = PLUGIN / "assets"
SIZES = (16, 24, 32, 48, 64, 128, 256, 512)
S = 2048

GROUND = (168, 126, 85)
GROUND_DARK = (122, 90, 58)
BALLAST = (140, 140, 140)
SLEEPER = (92, 64, 40)
RAIL = (40, 44, 52)
RAIL_LIGHT = (90, 96, 110)
VEG = (34, 139, 34)
POLE = (214, 0, 214)
WIRE = (255, 165, 0)
SKY = (91, 155, 213, 70)
DOT = (255, 255, 255, 60)


def u(x, y):
    return x * S, y * S


def main():
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Ground: a rounded band at the bottom.
    draw.rounded_rectangle([u(0.04, 0.56), u(0.96, 0.96)], radius=0.08 * S, fill=GROUND)
    draw.rectangle([u(0.04, 0.56), u(0.96, 0.66)], fill=GROUND_DARK)

    # Vegetation: two bushes left and right.
    for cx, cy, r in ((0.12, 0.56, 0.11), (0.21, 0.60, 0.07), (0.88, 0.56, 0.11), (0.79, 0.60, 0.07)):
        draw.ellipse([u(cx - r, cy - r), u(cx + r, cy + r)], fill=VEG)

    # Track bed (ballast) in perspective: wide at the bottom, narrow at the top.
    vp = (0.5, 0.30)
    bed = [u(0.16, 0.95), u(0.84, 0.95), u(0.60, 0.58), u(0.40, 0.58)]
    draw.polygon(bed, fill=BALLAST)

    # Sleepers: perspective spacing.
    y = 0.93
    step = 0.075
    while y > 0.60:
        t = (y - vp[1]) / (0.95 - vp[1])            # 1 at the bottom, 0 at the vp
        half = 0.30 * t + 0.02
        thick = 0.028 * t + 0.004
        draw.rectangle([u(0.5 - half, y - thick), u(0.5 + half, y)], fill=SLEEPER)
        y -= step * (0.35 + 0.65 * t)
        step *= 0.97

    # Rails: two thick converging lines.
    for side in (-1, 1):
        x0 = 0.5 + side * 0.215
        x1 = 0.5 + side * 0.065
        w = 0.040 * S
        draw.line([u(x0, 0.96), u(x1, 0.58)], fill=RAIL, width=int(w))
        draw.line([u(x0 - side * 0.012, 0.96), u(x1 - side * 0.004, 0.58)],
                  fill=RAIL_LIGHT, width=int(w * 0.30))

    # Catenary mast and wire.
    draw.rectangle([u(0.885, 0.12), u(0.925, 0.62)], fill=POLE)
    draw.rectangle([u(0.60, 0.16), u(0.925, 0.19)], fill=POLE)
    draw.line([u(0.62, 0.19), u(0.50, 0.58)], fill=WIRE, width=int(0.022 * S))

    # Scan fan from a sensor above (pale, translucent).
    fan = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(fan).polygon([u(0.5, 0.03), u(0.08, 0.58), u(0.92, 0.58)], fill=SKY)
    fan = fan.filter(ImageFilter.GaussianBlur(0.02 * S))
    img = Image.alpha_composite(fan, img)

    # Point-cloud texture: a grid of pale dots over the coloured shapes.
    dots = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    pitch = 0.028 * S
    r = 0.006 * S
    yy = 0.0
    row = 0
    while yy < S:
        xx = (pitch / 2) if row % 2 else 0.0
        while xx < S:
            dd.ellipse([xx - r, yy - r, xx + r, yy + r], fill=DOT)
            xx += pitch
        yy += pitch * 0.87
        row += 1
    mask = img.split()[3]
    dots.putalpha(Image.eval(mask, lambda a: min(a, 60)))
    img = Image.alpha_composite(img, dots)

    ASSETS.mkdir(exist_ok=True)
    for size in SIZES:
        img.resize((size, size), Image.LANCZOS).save(ASSETS / f"logo_{size}.png")
    img.resize((64, 64), Image.LANCZOS).save(PLUGIN / "icon.png")
    print("icon written:", PLUGIN / "icon.png", "and", len(SIZES), "sizes in assets/")


if __name__ == "__main__":
    main()
