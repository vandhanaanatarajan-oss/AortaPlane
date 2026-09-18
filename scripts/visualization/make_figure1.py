"""
make_figure1.py

Composites Figure 1 for the manuscript: a two-panel collage showing
(a) an axial CT slice with the annulus plane overlay, and
(b) a 3D reconstruction of the same plane.

Reuses the outputs already produced by plane_overlay.py and
fig4_3d_aorta() (Part 9 of the project) for a given patient -- this
script does NOT recompute anything, it only lays out two existing
PNGs into one publication-ready figure with panel labels and a caption.

Usage:
    python make_figure1.py --axial_png path/to/axial_slice.png \
                            --threeD_png path/to/3d_view.png \
                            --patient_id CONTCT_R_09_FBA \
                            --out_path results/figure1_annulus_plane.png \
                            --axial_crop 0 0 355 388 \
                            --threeD_crop 0 0 395 392

If your source images already contain only the single panel you want
(not a multi-panel figure with axial/coronal/sagittal side by side),
omit --axial_crop / --threeD_crop and the full image will be used.
"""

import argparse
from PIL import Image, ImageDraw, ImageFont


def resize_to_height(im, h):
    w = int(im.width * (h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def center_text(draw, text, font, x_start, box_w, y):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    draw.text((x_start + (box_w - w) / 2, y), text, fill='black', font=font)


def load_font(bold, size):
    paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold
        else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for p in paths:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axial_png', required=True)
    ap.add_argument('--threeD_png', required=True)
    ap.add_argument('--patient_id', required=True)
    ap.add_argument('--out_path', default='results/figure1_annulus_plane.png')
    ap.add_argument('--axial_crop', type=int, nargs=4, default=None,
                     metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    ap.add_argument('--threeD_crop', type=int, nargs=4, default=None,
                     metavar=('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    ap.add_argument('--target_height', type=int, default=380)
    ap.add_argument('--caption', default=None,
                     help='Override the default caption text if you want '
                          'custom wording once RC/NC/LC labels are added.')
    args = ap.parse_args()

    axial = Image.open(args.axial_png)
    threeD = Image.open(args.threeD_png)

    if args.axial_crop:
        axial = axial.crop(tuple(args.axial_crop))
    if args.threeD_crop:
        threeD = threeD.crop(tuple(args.threeD_crop))

    axial_r = resize_to_height(axial, args.target_height)
    threeD_r = resize_to_height(threeD, args.target_height)

    pad, gap, sublabel_h, caption_h, border = 24, 40, 34, 100, 2

    panel_w1 = axial_r.width + 2 * border
    panel_w2 = threeD_r.width + 2 * border
    total_w = pad * 2 + panel_w1 + gap + panel_w2
    total_h = pad + args.target_height + 2 * border + sublabel_h + caption_h + pad

    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)

    font_sub = load_font(bold=True, size=18)
    font_cap = load_font(bold=False, size=15)

    x1, x2 = pad, pad + panel_w1 + gap
    y_img = pad

    draw.rectangle([x1, y_img, x1 + panel_w1, y_img + args.target_height + 2 * border],
                   outline='black', width=2)
    draw.rectangle([x2, y_img, x2 + panel_w2, y_img + args.target_height + 2 * border],
                   outline='black', width=2)

    canvas.paste(axial_r, (x1 + border, y_img + border))
    canvas.paste(threeD_r, (x2 + border, y_img + border))

    label_y = y_img + args.target_height + 2 * border + 8
    center_text(draw, '(a)', font_sub, x1, panel_w1, label_y)
    center_text(draw, '(b)', font_sub, x2, panel_w2, label_y)

    caption_y = label_y + sublabel_h
    if args.caption:
        lines = args.caption.split('|')
    else:
        lines = [
            f'Fig. 1. The aortic annulus plane (patient {args.patient_id}). (a) Axial CT slice',
            'showing the annulus centre and plane overlay. (b) 3D reconstruction of the',
            'same annulus plane, parameterised by centre point [cx, cy, cz] and normal',
            'vector [nx, ny, nz].',
        ]
    for i, line in enumerate(lines):
        draw.text((pad, caption_y + i * 22), line, fill='black', font=font_cap)

    canvas.save(args.out_path)
    print(f"Saved: {args.out_path}  ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == '__main__':
    main()
