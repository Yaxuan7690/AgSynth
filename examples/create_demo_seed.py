"""Create the redistributable seed image used by the README quick-start example."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "demo_seed.png"

    figure, axis = plt.subplots(figsize=(6, 2))
    axis.plot([1, 5], [1, 1], color="#2563EB", linewidth=3)
    axis.scatter([1, 3, 5], [1, 1, 1], color="#1E3A8A", zorder=3)
    for x, label in ((1, "A"), (3, "M"), (5, "B")):
        axis.text(x, 1.16, label, ha="center", fontsize=14, fontweight="bold")
    axis.text(2, 0.74, "3", ha="center", fontsize=13)
    axis.text(4, 0.74, "3", ha="center", fontsize=13)
    axis.set_xlim(0.3, 5.7)
    axis.set_ylim(0.35, 1.55)
    axis.axis("off")
    figure.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
