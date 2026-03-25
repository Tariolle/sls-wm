"""Generate the V -> M -> C architecture pipeline figure (matches presentation)."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(16, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 16)
ax.set_ylim(-0.6, 5.5)
ax.axis("off")

# Colors matching presentation (vcolor=teal, mcolor=orange, ccolor=olive-green)
VCOLOR = "#2196F3"
MCOLOR = "#FF9800"
CCOLOR = "#6B8E23"
GRAY = "#666666"

# --- Boxes ---
box_specs = [
    # (x, y, w, h, facecolor, edgecolor, label, sublabel)
    (3.2,  1.8, 2.2, 2.4, "#BBDEFB", VCOLOR,  "V",  "FSQ-VAE"),
    (7.0,  1.8, 2.2, 2.4, "#FFE0B2", MCOLOR,  "M",  "Transformer"),
    (10.8, 1.8, 2.2, 2.4, "#C5E1A5", CCOLOR,  "C",  "Controller"),
]

for x, y, w, h, fc, ec, label, sublabel in box_specs:
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.12",
        facecolor=fc, edgecolor=ec, linewidth=2.2)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2 + 0.35, label,
            ha="center", va="center", fontsize=22, fontweight="bold",
            color="black", fontfamily="serif")
    ax.text(x + w / 2, y + h / 2 - 0.35, sublabel,
            ha="center", va="center", fontsize=13,
            color="#333333", fontfamily="serif")

# --- Main forward arrows ---
arrow_kw = dict(arrowstyle="-|>", color="black", lw=2.5, mutation_scale=22)
y_mid = 3.0

# Input x_t -> V
ax.annotate("", xy=(3.2, y_mid), xytext=(1.0, y_mid), arrowprops=arrow_kw)
ax.text(0.4, y_mid + 0.45, r"$\mathbf{x}_t$", ha="center", va="center",
        fontsize=16, fontweight="bold", fontfamily="serif")
ax.text(0.4, y_mid - 0.45, r"$64\times64\times1$", ha="center", va="center",
        fontsize=11, color=GRAY, fontfamily="serif")

# V -> M  (z_t, 8×8)
ax.annotate("", xy=(7.0, y_mid), xytext=(5.4, y_mid), arrowprops=arrow_kw)
ax.text(6.2, y_mid + 0.45, r"$\mathbf{z}_t$", ha="center", va="center",
        fontsize=15, fontweight="bold", fontfamily="serif")
ax.text(6.2, y_mid - 0.45, r"$8\times8$", ha="center", va="center",
        fontsize=11, color=GRAY, fontfamily="serif")

# M -> C  (ẑ_t, h_t)
ax.annotate("", xy=(10.8, y_mid), xytext=(9.2, y_mid), arrowprops=arrow_kw)
ax.text(10.0, y_mid + 0.45, r"$\hat{\mathbf{z}}_t, \mathbf{h}_t$",
        ha="center", va="center", fontsize=15, fontweight="bold",
        fontfamily="serif")

# C -> output a_t
ax.annotate("", xy=(14.6, y_mid), xytext=(13.0, y_mid), arrowprops=arrow_kw)
ax.text(15.2, y_mid + 0.45, r"$a_t$", ha="center", va="center",
        fontsize=16, fontweight="bold", fontfamily="serif")

# --- Feedback arrow: a_t from C back to M (bottom route) ---
# Down from C bottom, left under M, up into M bottom
fb_y = 0.8
ax.plot([11.7, 11.7], [1.8, fb_y], color="black", lw=2.2)          # down from C
ax.plot([8.3, 11.7], [fb_y, fb_y], color="black", lw=2.2)           # left
ax.annotate("", xy=(8.3, 1.8), xytext=(8.3, fb_y),                  # up into M
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2,
                            mutation_scale=20))
ax.text(10.0, fb_y - 0.35, r"$a_t$", ha="center", va="center",
        fontsize=14, fontweight="bold", fontfamily="serif")

# --- Legend at bottom ---
legend_y = -0.3
legend_parts = [
    (1.5,  r"$\mathbf{x}_t \in \mathbb{R}^{64 \times 64}$: Sobel frame"),
    (5.5,  r"$\mathbf{z}_t \in \{0,\ldots,999\}^{64}$: tokens"),
    (9.8,  r"$\mathbf{h}_t \in \mathbb{R}^{256}$: hidden state"),
    (13.5, r"$a_t \in \{0,1\}$: jump/idle"),
]
for lx, txt in legend_parts:
    ax.text(lx, legend_y, txt, ha="center", va="center",
            fontsize=11, color="black", fontfamily="serif")

plt.tight_layout(pad=0.3)
plt.savefig("docs/architecture_pipeline.png", dpi=180,
            facecolor="white", bbox_inches="tight")
print("Saved docs/architecture_pipeline.png")
