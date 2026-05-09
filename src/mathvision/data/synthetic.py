"""Generate an offline synthetic dataset for MathVision-Assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/outputs") / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

from mathvision.io_utils import write_jsonl


def generate_demo_dataset(output_dir: str | Path = "data/demo") -> dict[str, str]:
    """Generate demo images, QA records, and knowledge-base records."""

    root = Path(output_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    knowledge: list[dict[str, Any]] = []

    def image_path(filename: str) -> str:
        return str(images_dir / filename)

    def add_sample(
        sample_id: str,
        filename: str,
        question: str,
        answer: str,
        answer_type: str,
        keywords: list[str],
        kb_title: str,
        kb_content: str,
    ) -> None:
        kb_id = f"kb_{sample_id}"
        rel_image = image_path(filename)
        samples.append(
            {
                "id": sample_id,
                "image": rel_image,
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "keywords": keywords,
                "related_knowledge_ids": [kb_id],
            }
        )
        knowledge.append(
            {
                "id": kb_id,
                "title": kb_title,
                "content": kb_content,
                "source_image": rel_image,
            }
        )

    _plot_line_slope(images_dir / "line_slope_01.png")
    add_sample(
        "line_slope_01",
        "line_slope_01.png",
        "What is the slope of the line?",
        "2",
        "number",
        ["slope", "2"],
        "Line slope y=2x+1",
        "For a line y = mx + b, the slope is m. In y=2x+1, m=2.",
    )

    _plot_parabola(images_dir / "parabola_vertex_01.png")
    add_sample(
        "parabola_vertex_01",
        "parabola_vertex_01.png",
        "What is the vertex of the parabola?",
        "(1, 0)",
        "coordinate",
        ["vertex", "1", "0"],
        "Parabola vertex y=(x-1)^2",
        "The graph y=(x-1)^2 has its vertex at (1,0).",
    )

    _plot_bar_chart(images_dir / "bar_chart_max_01.png")
    add_sample(
        "bar_chart_max_01",
        "bar_chart_max_01.png",
        "Which category has the maximum value?",
        "C",
        "text",
        ["C", "maximum"],
        "Bar chart maximum category",
        "The bars A, B, C have values 3, 5, and 8. Category C is maximum.",
    )

    _plot_line_chart(images_dir / "line_chart_trend_01.png")
    add_sample(
        "line_chart_trend_01",
        "line_chart_trend_01.png",
        "What is the overall trend of the line chart?",
        "overall increasing",
        "text",
        ["overall", "increasing"],
        "Line chart increasing trend",
        "The plotted values rise from left to right, so the trend is overall increasing.",
    )

    _plot_triangle(images_dir / "triangle_angle_01.png")
    add_sample(
        "triangle_angle_01",
        "triangle_angle_01.png",
        "Two angles are 50 and 60 degrees. What is the third angle?",
        "70",
        "number",
        ["70", "angle"],
        "Triangle angle sum",
        "Angles in a triangle sum to 180 degrees, so 180-50-60=70.",
    )

    _plot_circle(images_dir / "circle_radius_01.png")
    add_sample(
        "circle_radius_01",
        "circle_radius_01.png",
        "The radius is 3. What is the area of the circle?",
        "9π",
        "expression",
        ["9", "pi", "area"],
        "Circle area with radius 3",
        "Circle area is pi*r^2. With radius 3, the area is 9*pi.",
    )

    _plot_formula_derivative(images_dir / "formula_derivative_01.png")
    add_sample(
        "formula_derivative_01",
        "formula_derivative_01.png",
        "What is the derivative of f(x)=x^2?",
        "2x",
        "expression",
        ["2x", "derivative"],
        "Derivative of x squared",
        "Using the power rule, d/dx x^2 = 2x.",
    )

    _plot_coordinate_point(images_dir / "coordinate_point_01.png")
    add_sample(
        "coordinate_point_01",
        "coordinate_point_01.png",
        "What are the coordinates of the marked point?",
        "(3, 2)",
        "coordinate",
        ["3", "2", "coordinate"],
        "Coordinate point reading",
        "The marked point is located at x=3 and y=2, so its coordinate is (3,2).",
    )

    _plot_histogram_total(images_dir / "histogram_total_01.png")
    add_sample(
        "histogram_total_01",
        "histogram_total_01.png",
        "What is the total count shown by the three bars?",
        "18",
        "number",
        ["18", "total"],
        "Histogram total count",
        "The three bar values are 4, 6, and 8. Their sum is 18.",
    )

    _plot_rectangle(images_dir / "rectangle_area_01.png")
    add_sample(
        "rectangle_area_01",
        "rectangle_area_01.png",
        "What is the area of the rectangle?",
        "24",
        "number",
        ["24", "area"],
        "Rectangle area",
        "A rectangle with length 6 and width 4 has area 6*4=24.",
    )

    _plot_fraction(images_dir / "fraction_simplify_01.png")
    add_sample(
        "fraction_simplify_01",
        "fraction_simplify_01.png",
        "Simplify the fraction 6/8.",
        "3/4",
        "expression",
        ["3/4", "simplify"],
        "Fraction simplification",
        "The fraction 6/8 can be simplified by dividing numerator and denominator by 2 to get 3/4.",
    )

    _plot_scatter(images_dir / "scatter_correlation_01.png")
    add_sample(
        "scatter_correlation_01",
        "scatter_correlation_01.png",
        "What type of correlation does the scatter plot show?",
        "positive correlation",
        "text",
        ["positive", "correlation"],
        "Scatter plot correlation",
        "The points move upward as x increases, indicating positive correlation.",
    )

    _plot_sine(images_dir / "sine_period_01.png")
    add_sample(
        "sine_period_01",
        "sine_period_01.png",
        "What is the period of the sine curve?",
        "2π",
        "expression",
        ["2", "pi", "period"],
        "Sine curve period",
        "The standard sine function sin(x) has period 2*pi.",
    )

    _plot_pie(images_dir / "pie_chart_share_01.png")
    add_sample(
        "pie_chart_share_01",
        "pie_chart_share_01.png",
        "Which slice has the largest share?",
        "Blue, 50%",
        "text",
        ["Blue", "50"],
        "Pie chart largest share",
        "The largest pie slice is Blue and it covers 50 percent.",
    )

    qa_file = root / "qa.jsonl"
    kb_file = root / "knowledge_base.jsonl"
    write_jsonl(qa_file, samples)
    write_jsonl(kb_file, knowledge)
    return {
        "qa_file": str(qa_file),
        "knowledge_base_file": str(kb_file),
        "images_dir": str(images_dir),
        "num_samples": str(len(samples)),
    }


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_line_slope(path: Path) -> None:
    x = np.linspace(-3, 3, 100)
    y = 2 * x + 1
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#2563eb", linewidth=2.5, label="y=2x+1")
    _style_axes(ax, "Line: y=2x+1")
    ax.legend()
    _save(fig, path)


def _plot_parabola(path: Path) -> None:
    x = np.linspace(-2, 4, 200)
    y = (x - 1) ** 2
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#dc2626", linewidth=2.5)
    ax.scatter([1], [0], color="black", zorder=3)
    ax.annotate("vertex (1,0)", xy=(1, 0), xytext=(1.3, 1.5), arrowprops={"arrowstyle": "->"})
    _style_axes(ax, "Parabola: y=(x-1)^2")
    _save(fig, path)


def _plot_bar_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["A", "B", "C"]
    values = [3, 5, 8]
    ax.bar(labels, values, color=["#64748b", "#14b8a6", "#f97316"])
    ax.set_ylim(0, 10)
    ax.set_title("Bar Chart Values")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.2, str(value), ha="center")
    _save(fig, path)


def _plot_line_chart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(1, 6)
    y = [1, 2, 4, 5, 7]
    ax.plot(x, y, marker="o", color="#16a34a", linewidth=2.5)
    ax.set_title("Overall Increasing Trend")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def _plot_triangle(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    pts = np.array([[0, 0], [4, 0], [1.5, 3]])
    ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=2.5, edgecolor="#7c3aed"))
    ax.text(0.25, 0.18, "50°", fontsize=13)
    ax.text(3.35, 0.18, "60°", fontsize=13)
    ax.text(1.45, 2.45, "?", fontsize=16)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Triangle Angles")
    _save(fig, path)


def _plot_circle(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    circle = Circle((0, 0), 3, fill=False, linewidth=2.5, edgecolor="#0891b2")
    ax.add_patch(circle)
    ax.plot([0, 3], [0, 0], color="#ef4444", linewidth=2)
    ax.text(1.3, 0.2, "r=3", fontsize=13)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("Circle Radius")
    _save(fig, path)


def _plot_formula_derivative(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.58, r"$f(x)=x^2$", fontsize=28, ha="center", va="center")
    ax.text(0.5, 0.25, "Find derivative", fontsize=13, ha="center", va="center")
    _save(fig, path)


def _plot_coordinate_point(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 4)
    ax.set_xticks(range(-1, 6))
    ax.set_yticks(range(-1, 5))
    _style_axes(ax, "Coordinate Point")
    ax.scatter([3], [2], s=80, color="#e11d48")
    ax.annotate("(3,2)", xy=(3, 2), xytext=(3.15, 2.35), arrowprops={"arrowstyle": "->"})
    _save(fig, path)


def _plot_histogram_total(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["X", "Y", "Z"]
    values = [4, 6, 8]
    ax.bar(labels, values, color="#0f766e")
    ax.set_ylim(0, 10)
    ax.set_title("Counts: 4, 6, 8")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.2, str(value), ha="center")
    _save(fig, path)


def _plot_rectangle(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.add_patch(Rectangle((0, 0), 6, 4, fill=False, edgecolor="#2563eb", linewidth=2.5))
    ax.text(3, -0.35, "6", ha="center", fontsize=13)
    ax.text(6.2, 2, "4", va="center", fontsize=13)
    ax.set_xlim(-0.5, 7)
    ax.set_ylim(-0.8, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rectangle")
    _save(fig, path)


def _plot_fraction(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.55, r"$\frac{6}{8}$", fontsize=34, ha="center", va="center")
    ax.text(0.5, 0.2, "Simplify", fontsize=13, ha="center", va="center")
    _save(fig, path)


def _plot_scatter(path: Path) -> None:
    rng = np.random.default_rng(7)
    x = np.arange(1, 9)
    y = x + rng.normal(0, 0.45, size=len(x))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, color="#9333ea", s=60)
    ax.set_title("Scatter Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def _plot_sine(path: Path) -> None:
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#ea580c", linewidth=2.5)
    ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "pi", "2pi"])
    ax.set_title("Sine Curve")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def _plot_pie(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    values = [50, 30, 20]
    labels = ["Blue 50%", "Green 30%", "Orange 20%"]
    colors = ["#2563eb", "#16a34a", "#f97316"]
    start = 90
    for value, label, color in zip(values, labels, colors):
        theta2 = start + value / 100 * 360
        ax.add_patch(Wedge((0, 0), 1.0, start, theta2, facecolor=color, edgecolor="white"))
        start = theta2
    ax.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pie Chart")
    _save(fig, path)
