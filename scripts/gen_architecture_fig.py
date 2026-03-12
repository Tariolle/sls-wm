"""Generate the V -> M -> C architecture pipeline figure."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(16, 5.8))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")
ax.set_xlim(0, 16)
ax.set_ylim(0, 5.8)
ax.axis("off")

# Title
ax.text(8, 5.45, "V  →  M  →  C    Architecture Pipeline",
        ha="center", va="center", fontsize=18, fontweight="bold",
        color="white", fontfamily="monospace")

# Box specs: (x, y, w, h, color, label_top, line1, line2)
boxes = [
    (0.3,  0.8, 2.4, 3.2, "#1a1f2e", "INPUT",
     "Sobel Edge Map", "64 × 64 × 1"),
    (3.5,  0.8, 3.0, 3.2, "#2d6a4f", "V",
     "FSQ-VAE", "Encoder"),
    (7.3,  0.8, 3.0, 3.2, "#1a5276", "M",
     "Transformer", "P(tokens$_{t+1}$ | tokens$_t$, $a_t$)"),
    (11.1, 0.8, 3.0, 3.2, "#7b2d3b", "C",
     "Linear Controller", "$a = W \\cdot h_t$"),
    (14.5, 0.8, 1.3, 3.2, "#1a1f2e", "OUT",
     "Action", "Jump / Idle"),
]

for x, y, w, h, color, top_label, line1, line2 in boxes:
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15",
        facecolor=color, edgecolor="#2a3040", linewidth=1.5)
    ax.add_patch(rect)
    # Top label
    ax.text(x + w / 2, y + h + 0.15, top_label,
            ha="center", va="center", fontsize=11, color="#6e8898",
            fontfamily="monospace")
    # Main text
    ax.text(x + w / 2, y + h / 2 + 0.25, line1,
            ha="center", va="center", fontsize=13, fontweight="bold",
            color="white", fontfamily="monospace")
    ax.text(x + w / 2, y + h / 2 - 0.25, line2,
            ha="center", va="center", fontsize=11, color="#c0d0e0",
            fontfamily="monospace")

# Arrows with dimension labels (above) and bit counts (below)
arrow_props = dict(arrowstyle="-|>", color="#4aa3df", lw=2.5,
                   mutation_scale=18)

# (x_start, x_end, label_above, label_below)
arrows = [
    (2.7,  3.5,  "4,096 floats",       "32,768 bits"),
    (6.5,  7.3,  "8×8 = 64 tokens",    "638 bits (51× compression)"),
    (10.3, 11.1, "$h_t \\in R^d$",     ""),
    (14.1, 14.5, "",                    "1 bit"),
]

y_mid = 2.4
for x1, x2, above, below in arrows:
    ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                arrowprops=arrow_props)
    xc = (x1 + x2) / 2
    if above:
        ax.text(xc, y_mid + 0.55, above,
                ha="center", va="center", fontsize=11, color="#7cc4f0",
                fontfamily="monospace", fontweight="bold")
    if below:
        ax.text(xc, y_mid - 0.5, below,
                ha="center", va="center", fontsize=10, color="#a0b8cc",
                fontfamily="monospace")

plt.tight_layout(pad=0.3)
plt.savefig("docs/architecture_pipeline.png", dpi=180,
            facecolor=fig.get_facecolor(), bbox_inches="tight")
print("Saved docs/architecture_pipeline.png")
