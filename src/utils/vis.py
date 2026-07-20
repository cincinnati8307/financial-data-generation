import matplotlib.pyplot as plt

metrics = ["Precision", "Recall", "F1", "Accuracy"]
values = [0.9, 0.97, 0.93, 0.93]

fig, ax = plt.subplots(figsize=(4.6, 3.0))

bars = ax.bar(metrics, values, width=0.62)

ax.set_ylabel("Score")
ax.set_ylim(0, 1.0)
ax.set_title("Evaluation Performance", pad=15)
ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
ax.set_axisbelow(True)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.012,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
plt.savefig("paper_style_metrics_bar_chart.pdf", bbox_inches="tight")
plt.savefig("paper_style_metrics_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()