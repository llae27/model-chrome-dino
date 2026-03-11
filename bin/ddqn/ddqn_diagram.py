"""
Generates a DDQN two-network architecture diagram and saves it as ddqn_diagram.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, text, fontsize=10, textcolor="white"):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1", facecolor=color, edgecolor="white", linewidth=1.5, zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, fontweight="bold", zorder=4)

def arrow(ax, x1, y1, x2, y2, color="#555555", label="", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.05, my+0.15, label, fontsize=8, color=color, zorder=6)

# ── background panels ─────────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.4), 5.4, 6.1,
    boxstyle="round,pad=0.15", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2, zorder=1))
ax.text(3.0, 6.25, "Online Network  (trained every step)", ha="center",
        fontsize=10, color="#1565C0", fontweight="bold")

ax.add_patch(mpatches.FancyBboxPatch((7.3, 0.4), 5.4, 6.1,
    boxstyle="round,pad=0.15", facecolor="#FBE9E7", edgecolor="#B71C1C", linewidth=2, zorder=1))
ax.text(10.0, 6.25, "Target Network  (frozen, copied every 10k steps)", ha="center",
        fontsize=10, color="#B71C1C", fontweight="bold")

# ── Online network layers ─────────────────────────────────────────────────────
box(ax, 0.7, 4.6, 4.6, 0.8, "#1976D2", "Input: state  s  (80×160×4 grayscale frames)")
box(ax, 0.7, 3.5, 4.6, 0.8, "#1565C0", "CNN  (3 conv layers)")
box(ax, 0.7, 2.4, 4.6, 0.8, "#0D47A1", "FC 512 → ReLU")
box(ax, 0.7, 1.3, 2.1, 0.8, "#0D47A1", "Q(s, NOOP)")
box(ax, 3.2, 1.3, 2.1, 0.8, "#0D47A1", "Q(s, UP)")

arrow(ax, 3.0, 4.6, 3.0, 4.3)
arrow(ax, 3.0, 3.5, 3.0, 3.2)
arrow(ax, 3.0, 2.4, 1.75, 2.1)
arrow(ax, 3.0, 2.4, 4.25, 2.1)

# argmax label
ax.text(3.0, 0.95, "argmax  →  select  a*", ha="center", fontsize=9,
        color="#1565C0", fontstyle="italic")

# ── Target network layers ─────────────────────────────────────────────────────
box(ax, 7.7, 4.6, 4.6, 0.8, "#E53935", "Input: next state  s'  (80×160×4)")
box(ax, 7.7, 3.5, 4.6, 0.8, "#C62828", "CNN  (3 conv layers)  —  frozen weights")
box(ax, 7.7, 2.4, 4.6, 0.8, "#B71C1C", "FC 512 → ReLU  —  frozen weights")
box(ax, 7.7, 1.3, 4.6, 0.8, "#B71C1C", "Q_target(s', a*)  →  evaluate chosen action")

arrow(ax, 10.0, 4.6, 10.0, 4.3)
arrow(ax, 10.0, 3.5, 10.0, 3.2)
arrow(ax, 10.0, 2.4, 10.0, 2.1)

# ── Cross arrows: a* flows from online to target ──────────────────────────────
ax.annotate("", xy=(7.7, 1.7), xytext=(5.3, 1.7),
    arrowprops=dict(arrowstyle="-|>", color="#FF6F00", lw=2.0), zorder=5)
ax.text(6.5, 1.85, "a*  (action\nselected\nby online)", ha="center",
        fontsize=8, color="#FF6F00", fontweight="bold")

# ── Target update arrow ───────────────────────────────────────────────────────
ax.annotate("", xy=(7.3, 5.0), xytext=(5.7, 5.0),
    arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2.0,
                    connectionstyle="arc3,rad=-0.3"), zorder=5)
ax.text(6.5, 5.55, "copy weights\nevery 10k steps", ha="center",
        fontsize=8.5, color="#2E7D32", fontweight="bold")

# ── Bellman target box ────────────────────────────────────────────────────────
box(ax, 3.8, 0.05, 5.4, 0.75, "#37474F",
    "Target:  y = r  +  γ · Q_target(s', a*)     Loss = SmoothL1( Q_online(s,a) − y )",
    fontsize=9)

arrow(ax, 3.0, 1.3, 4.5, 0.8, color="#555555")
arrow(ax, 10.0, 1.3, 8.5, 0.8, color="#555555")

plt.title("DDQN: Double Deep Q-Network Architecture", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("ddqn_diagram.png", dpi=150, bbox_inches="tight")
print("Saved ddqn_diagram.png")
plt.show()
