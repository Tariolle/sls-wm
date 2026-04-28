#!/bin/bash
# Rebuild pipeline figure from docs/pipeline.tex.
# Outputs: static/images/pipeline.svg (used by site), architecture_pipeline.png (og:image fallback).
# Requires: pdflatex, pdftocairo, pdftoppm, Python with Pillow.
set -e
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode pipeline.tex > /dev/null || true
[ -f pipeline.pdf ] || { echo "pdflatex failed"; exit 1; }

pdftocairo -svg pipeline.pdf static/images/pipeline.svg

pdftoppm -png -r 300 pipeline.pdf pipeline_tmp
python -c "
from PIL import Image, ImageDraw
img = Image.open('pipeline_tmp-1.png').convert('RGBA')
w, h = img.size
img = img.crop((0, 0, w, h - 12))
w, h = img.size
mask = Image.new('L', (w, h), 0)
draw = ImageDraw.Draw(mask)
draw.rounded_rectangle([(0, 0), (w, h)], radius=30, fill=255)
img.putalpha(mask)
img.save('architecture_pipeline.png')
"

rm -f pipeline_tmp-1.png pipeline.pdf pipeline.aux pipeline.log
echo "Done: static/images/pipeline.svg, architecture_pipeline.png"
