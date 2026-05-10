"""Generate an offline synthetic dataset for MathVision-Assistant."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("data/outputs") / "matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from mathvision.io_utils import write_jsonl

RANDOM_SEED = 42
TASK_TYPES = [
    "function_plot",
    "bar_chart",
    "line_chart",
    "pie_chart",
    "geometry",
    "formula",
    "coordinate",
    "table",
]


def generate_demo_dataset(output_dir: str | Path = "data/demo") -> dict[str, str]:
    """Generate 100 reproducible demo images, QA records, splits, and KB records."""

    root = Path(output_dir)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for old_image in images_dir.glob("*.png"):
        old_image.unlink()

    knowledge = _build_knowledge_base()
    samples: list[dict[str, Any]] = []

    for index in range(100):
        task_type = TASK_TYPES[index % len(TASK_TYPES)]
        variant = index // len(TASK_TYPES)
        split = _split_for_index(index)
        difficulty = _difficulty_for_variant(variant)
        sample = _make_sample(
            index=index,
            variant=variant,
            task_type=task_type,
            split=split,
            difficulty=difficulty,
            images_dir=images_dir,
        )
        samples.append(sample)

    # Keep one legacy image used in earlier README/reports. It is not part of
    # the 100-row QA split, but prevents old result references from breaking.
    _plot_legacy_scatter(images_dir / "scatter_correlation_01.png")

    qa_file = root / "qa.jsonl"
    qa_train_file = root / "qa_train.jsonl"
    qa_val_file = root / "qa_val.jsonl"
    qa_test_file = root / "qa_test.jsonl"
    kb_file = root / "knowledge_base.jsonl"

    write_jsonl(qa_file, samples)
    write_jsonl(qa_train_file, [item for item in samples if item["split"] == "train"])
    write_jsonl(qa_val_file, [item for item in samples if item["split"] == "val"])
    write_jsonl(qa_test_file, [item for item in samples if item["split"] == "test"])
    write_jsonl(kb_file, knowledge)

    return {
        "qa_file": str(qa_file),
        "qa_train_file": str(qa_train_file),
        "qa_val_file": str(qa_val_file),
        "qa_test_file": str(qa_test_file),
        "knowledge_base_file": str(kb_file),
        "images_dir": str(images_dir),
        "num_samples": str(len(samples)),
        "num_train": str(sum(1 for item in samples if item["split"] == "train")),
        "num_val": str(sum(1 for item in samples if item["split"] == "val")),
        "num_test": str(sum(1 for item in samples if item["split"] == "test")),
    }


def _split_for_index(index: int) -> str:
    if index < 70:
        return "train"
    if index < 85:
        return "val"
    return "test"


def _difficulty_for_variant(variant: int) -> str:
    return ["easy", "medium", "hard"][variant % 3]


def _image_path(images_dir: Path, filename: str) -> str:
    return str(images_dir / filename)


def _sample(
    sample_id: str,
    filename: str,
    question: str,
    answer: str,
    answer_type: str,
    keywords: list[str],
    related_knowledge_ids: list[str],
    task_type: str,
    difficulty: str,
    split: str,
    images_dir: Path,
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "image": _image_path(images_dir, filename),
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "keywords": keywords,
        "related_knowledge_ids": related_knowledge_ids,
        "task_type": task_type,
        "difficulty": difficulty,
        "split": split,
    }


def _make_sample(
    index: int,
    variant: int,
    task_type: str,
    split: str,
    difficulty: str,
    images_dir: Path,
) -> dict[str, Any]:
    makers = {
        "function_plot": _make_function_plot_sample,
        "bar_chart": _make_bar_chart_sample,
        "line_chart": _make_line_chart_sample,
        "pie_chart": _make_pie_chart_sample,
        "geometry": _make_geometry_sample,
        "formula": _make_formula_sample,
        "coordinate": _make_coordinate_sample,
        "table": _make_table_sample,
    }
    return makers[task_type](index, variant, split, difficulty, images_dir)


def _make_function_plot_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["slope", "intercept", "vertex", "monotonic", "period", "intersection"]
    mode = modes[variant % len(modes)]
    if index == 0:
        sample_id = "line_slope_01"
        filename = "line_slope_01.png"
        m, b = 2, 1
        _plot_line(images_dir / filename, m=m, b=b, title="Line: y=2x+1")
        return _sample(
            sample_id,
            filename,
            "What is the slope of the line?",
            "2",
            "number",
            ["slope", "2"],
            ["kb_line_slope"],
            "function_plot",
            difficulty,
            split,
            images_dir,
        )

    sample_id = f"function_plot_{index:03d}_{mode}"
    if index == 16:
        sample_id = "parabola_vertex_01"
    if index == 32:
        sample_id = "sine_period_01"
    filename = f"{sample_id}.png"
    if mode == "slope":
        m = [-3, -2, -1, 1, 2, 3][variant % 6]
        b = (variant % 5) - 2
        _plot_line(images_dir / filename, m=m, b=b, title=f"Line: y={m}x+{b}")
        answer = str(m)
        return _sample(
            sample_id,
            filename,
            "What is the slope of the line?",
            answer,
            "number",
            ["slope", answer],
            ["kb_line_slope"],
            "function_plot",
            difficulty,
            split,
            images_dir,
        )
    if mode == "intercept":
        m = [1, 2, -1, -2][variant % 4]
        b = (variant % 7) - 3
        _plot_line(images_dir / filename, m=m, b=b, title=f"Line: y={m}x+{b}")
        answer = str(b)
        return _sample(
            sample_id,
            filename,
            "What is the y-intercept of the line?",
            answer,
            "number",
            ["intercept", answer],
            ["kb_line_intercept"],
            "function_plot",
            difficulty,
            split,
            images_dir,
        )
    if mode == "vertex":
        h = (variant % 5) - 2
        k = (variant % 4) - 1
        _plot_parabola(images_dir / filename, h=h, k=k)
        answer = f"({h}, {k})"
        return _sample(
            sample_id,
            filename,
            "What is the vertex of the parabola?",
            answer,
            "coordinate",
            ["vertex", str(h), str(k)],
            ["kb_parabola_vertex"],
            "function_plot",
            difficulty,
            split,
            images_dir,
        )
    if mode == "monotonic":
        m = 1 + (variant % 4)
        _plot_line(images_dir / filename, m=m, b=-1, title=f"Line: y={m}x-1")
        return _sample(
            sample_id,
            filename,
            "Is the line increasing or decreasing?",
            "increasing",
            "text",
            ["increasing", "positive"],
            ["kb_function_monotonic"],
            "function_plot",
            difficulty,
            split,
            images_dir,
        )
    if mode == "period":
        _plot_sine(images_dir / filename)
        return _sample(
            sample_id,
            filename,
            "What is the period of the sine curve?",
            "2π",
            "expression",
            ["2", "pi", "period"],
            ["kb_sine_period"],
            "function_plot",
            difficulty,
            split,
            images_dir,
        )

    x_intercept = (variant % 5) - 2
    m = 1 + (variant % 3)
    b = -m * x_intercept
    _plot_line(images_dir / filename, m=m, b=b, title=f"Line crosses x-axis at {x_intercept}")
    return _sample(
        sample_id,
        filename,
        "Where does the line cross the x-axis?",
        str(x_intercept),
        "number",
        ["x-axis", str(x_intercept)],
        ["kb_line_intersection"],
        "function_plot",
        difficulty,
        split,
        images_dir,
    )


def _make_bar_chart_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["maximum", "minimum", "difference", "total", "ranking"]
    mode = modes[variant % len(modes)]
    if index == 1:
        sample_id = "bar_chart_max_01"
        filename = "bar_chart_max_01.png"
        labels, values = ["A", "B", "C"], [3, 5, 8]
        _plot_bar(images_dir / filename, labels, values, "Bar Chart Values")
        return _sample(
            sample_id,
            filename,
            "Which category has the maximum value?",
            "C",
            "text",
            ["C", "maximum"],
            ["kb_bar_max"],
            "bar_chart",
            difficulty,
            split,
            images_dir,
        )

    labels = ["A", "B", "C", "D"]
    values = [
        2 + (variant * 2) % 8,
        3 + (variant * 3) % 7,
        4 + (variant * 5) % 9,
        1 + (variant * 4) % 8,
    ]
    sample_id = f"bar_chart_{index:03d}_{mode}"
    if index == 25:
        sample_id = "histogram_total_01"
    filename = f"{sample_id}.png"
    _plot_bar(images_dir / filename, labels, values, "Bar Chart")
    max_label = labels[int(np.argmax(values))]
    min_label = labels[int(np.argmin(values))]
    if mode == "maximum":
        answer = max_label
        question = "Which category has the maximum value?"
        keywords = [answer, "maximum"]
        kb = ["kb_bar_max"]
        answer_type = "text"
    elif mode == "minimum":
        answer = min_label
        question = "Which category has the minimum value?"
        keywords = [answer, "minimum"]
        kb = ["kb_bar_min"]
        answer_type = "text"
    elif mode == "difference":
        answer = str(max(values) - min(values))
        question = "What is the difference between the maximum and minimum values?"
        keywords = [answer, "difference"]
        kb = ["kb_bar_difference"]
        answer_type = "number"
    elif mode == "total":
        answer = str(sum(values))
        question = "What is the total value of all bars?"
        keywords = [answer, "total"]
        kb = ["kb_bar_total"]
        answer_type = "number"
    else:
        sorted_labels = [label for _, label in sorted(zip(values, labels), reverse=True)]
        answer = " > ".join(sorted_labels)
        question = "Rank the categories from largest to smallest."
        keywords = [sorted_labels[0], sorted_labels[-1], "rank"]
        kb = ["kb_bar_ranking"]
        answer_type = "text"
    return _sample(sample_id, filename, question, answer, answer_type, keywords, kb, "bar_chart", difficulty, split, images_dir)


def _make_line_chart_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["trend", "peak", "valley", "growth"]
    mode = modes[variant % len(modes)]
    sample_id = "line_chart_trend_01" if index == 2 else f"line_chart_{index:03d}_{mode}"
    filename = f"{sample_id}.png"
    if index == 2:
        values = [1, 2, 4, 5, 7]
    else:
        base = 1 + variant
        values = [base, base + 2, base + (variant % 5), base + 4, base + 6]
        if mode == "valley":
            values = [base + 5, base + 3, base, base + 4, base + 6]
        if mode == "peak":
            values = [base, base + 3, base + 7, base + 4, base + 2]
    _plot_line_chart(images_dir / filename, values, "Line Chart")
    if mode == "trend" or index == 2:
        answer = "overall increasing"
        question = "What is the overall trend of the line chart?"
        keywords = ["overall", "increasing"]
        kb = ["kb_line_chart_trend"]
        answer_type = "text"
    elif mode == "peak":
        answer = str(max(values))
        question = "What is the peak value in the line chart?"
        keywords = [answer, "peak"]
        kb = ["kb_line_chart_peak"]
        answer_type = "number"
    elif mode == "valley":
        answer = str(min(values))
        question = "What is the valley value in the line chart?"
        keywords = [answer, "valley"]
        kb = ["kb_line_chart_valley"]
        answer_type = "number"
    else:
        answer = str(values[-1] - values[0])
        question = "What is the growth from the first point to the last point?"
        keywords = [answer, "growth"]
        kb = ["kb_line_chart_growth"]
        answer_type = "number"
    return _sample(sample_id, filename, question, answer, answer_type, keywords, kb, "line_chart", difficulty, split, images_dir)


def _make_pie_chart_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["largest", "percentage", "compare"]
    mode = modes[variant % len(modes)]
    sample_id = "pie_chart_share_01" if index == 11 else f"pie_chart_{index:03d}_{mode}"
    filename = f"{sample_id}.png"
    labels = ["Blue", "Green", "Orange"]
    blue = 40 + (variant % 4) * 5
    green = 30 - (variant % 3) * 5
    orange = 100 - blue - green
    values = [blue, green, orange]
    if index == 11:
        values = [50, 30, 20]
    _plot_pie(images_dir / filename, labels, values, "Pie Chart")
    if mode == "largest" or index == 11:
        largest_idx = int(np.argmax(values))
        answer = f"{labels[largest_idx]}, {values[largest_idx]}%"
        question = "Which slice has the largest share?"
        keywords = [labels[largest_idx], str(values[largest_idx])]
        kb = ["kb_pie_largest"]
        answer_type = "text"
    elif mode == "percentage":
        answer = f"{values[0]}%"
        question = "What percentage does the Blue slice represent?"
        keywords = ["Blue", str(values[0])]
        kb = ["kb_pie_percentage"]
        answer_type = "text"
    else:
        answer = "Blue"
        question = "Which slice is larger, Blue or Green?"
        keywords = ["Blue", "larger"]
        kb = ["kb_pie_compare"]
        answer_type = "text"
    return _sample(sample_id, filename, question, answer, answer_type, keywords, kb, "pie_chart", difficulty, split, images_dir)


def _make_geometry_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["triangle_angle", "rectangle_area", "circle_area", "distance"]
    mode = modes[variant % len(modes)]
    if index == 4:
        mode = "triangle_angle"
        sample_id = "triangle_angle_01"
        filename = "triangle_angle_01.png"
        a1, a2 = 50, 60
    elif index == 12:
        mode = "circle_area"
        sample_id = "circle_radius_01"
        filename = "circle_radius_01.png"
    else:
        sample_id = f"geometry_{index:03d}_{mode}"
        if index == 44:
            sample_id = "rectangle_area_01"
        filename = f"{sample_id}.png"
        a1, a2 = 40 + (variant % 4) * 10, 50 + (variant % 3) * 10
    if mode == "triangle_angle":
        third = 180 - a1 - a2
        _plot_triangle(images_dir / filename, a1, a2)
        return _sample(
            sample_id,
            filename,
            f"Two angles are {a1} and {a2} degrees. What is the third angle?",
            str(third),
            "number",
            [str(third), "angle"],
            ["kb_triangle_angle"],
            "geometry",
            difficulty,
            split,
            images_dir,
        )
    if mode == "rectangle_area":
        width = 3 + (variant % 5)
        height = 2 + (variant % 4)
        _plot_rectangle(images_dir / filename, width, height)
        area = width * height
        return _sample(
            sample_id,
            filename,
            "What is the area of the rectangle?",
            str(area),
            "number",
            [str(area), "area"],
            ["kb_rectangle_area"],
            "geometry",
            difficulty,
            split,
            images_dir,
        )
    if mode == "circle_area":
        radius = 3 if index == 12 else 2 + (variant % 4)
        _plot_circle(images_dir / filename, radius)
        answer = f"{radius * radius}π"
        return _sample(
            sample_id,
            filename,
            f"The radius is {radius}. What is the area of the circle?",
            answer,
            "expression",
            [str(radius * radius), "pi", "area"],
            ["kb_circle_area"],
            "geometry",
            difficulty,
            split,
            images_dir,
        )
    p1 = (variant % 4, variant % 3)
    p2 = (p1[0] + 3, p1[1] + 4)
    _plot_two_points(images_dir / filename, p1, p2)
    return _sample(
        sample_id,
        filename,
        "What is the distance between the two marked points?",
        "5",
        "number",
        ["5", "distance"],
        ["kb_coordinate_distance"],
        "geometry",
        difficulty,
        split,
        images_dir,
    )


def _make_formula_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["derivative", "simplify_fraction", "solve_equation", "integral"]
    mode = modes[variant % len(modes)]
    if index == 5:
        mode = "derivative"
        sample_id = "formula_derivative_01"
        filename = "formula_derivative_01.png"
        n = 2
    else:
        sample_id = f"formula_{index:03d}_{mode}"
        if index == 13:
            sample_id = "fraction_simplify_01"
        filename = f"{sample_id}.png"
        n = 2 + (variant % 4)
    if mode == "derivative":
        _plot_formula(images_dir / filename, rf"$f(x)=x^{n}$", "Find derivative")
        answer = "2x" if n == 2 else f"{n}x^{n - 1}"
        return _sample(
            sample_id,
            filename,
            f"What is the derivative of f(x)=x^{n}?",
            answer,
            "expression",
            [answer, "derivative"],
            ["kb_derivative_power"],
            "formula",
            difficulty,
            split,
            images_dir,
        )
    if mode == "simplify_fraction":
        numerator = 2 * (variant % 5 + 2)
        denominator = 2 * (variant % 5 + 3)
        g = math.gcd(numerator, denominator)
        answer = f"{numerator // g}/{denominator // g}"
        _plot_formula(images_dir / filename, rf"$\frac{{{numerator}}}{{{denominator}}}$", "Simplify")
        return _sample(
            sample_id,
            filename,
            f"Simplify the fraction {numerator}/{denominator}.",
            answer,
            "expression",
            [answer, "simplify"],
            ["kb_fraction_simplify"],
            "formula",
            difficulty,
            split,
            images_dir,
        )
    if mode == "solve_equation":
        a = 2 + (variant % 5)
        b = a + 4
        _plot_formula(images_dir / filename, rf"$x+{a}={b}$", "Solve for x")
        return _sample(
            sample_id,
            filename,
            f"Solve for x: x + {a} = {b}.",
            "4",
            "number",
            ["4", "x"],
            ["kb_solve_linear"],
            "formula",
            difficulty,
            split,
            images_dir,
        )
    _plot_formula(images_dir / filename, r"$\int 2x\,dx$", "Integrate")
    return _sample(
        sample_id,
        filename,
        "What is the integral of 2x?",
        "x^2 + C",
        "expression",
        ["x^2", "C", "integral"],
        ["kb_integral_power"],
        "formula",
        difficulty,
        split,
        images_dir,
    )


def _make_coordinate_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["point", "distance", "midpoint", "slope"]
    mode = modes[variant % len(modes)]
    if index == 6:
        sample_id = "coordinate_point_01"
        filename = "coordinate_point_01.png"
        p = (3, 2)
        _plot_point(images_dir / filename, p)
        return _sample(
            sample_id,
            filename,
            "What are the coordinates of the marked point?",
            "(3, 2)",
            "coordinate",
            ["3", "2", "coordinate"],
            ["kb_coordinate_point"],
            "coordinate",
            difficulty,
            split,
            images_dir,
        )
    sample_id = f"coordinate_{index:03d}_{mode}"
    filename = f"{sample_id}.png"
    p1 = (variant % 5, (variant * 2) % 5)
    p2 = (p1[0] + 2, p1[1] + 2)
    if mode == "point":
        _plot_point(images_dir / filename, p1)
        answer = f"({p1[0]}, {p1[1]})"
        question = "What are the coordinates of the marked point?"
        keywords = [str(p1[0]), str(p1[1]), "coordinate"]
        kb = ["kb_coordinate_point"]
        answer_type = "coordinate"
    elif mode == "distance":
        p2 = (p1[0] + 3, p1[1] + 4)
        _plot_two_points(images_dir / filename, p1, p2)
        answer = "5"
        question = "What is the distance between the two marked points?"
        keywords = ["5", "distance"]
        kb = ["kb_coordinate_distance"]
        answer_type = "number"
    elif mode == "midpoint":
        p2 = (p1[0] + 4, p1[1] + 2)
        _plot_two_points(images_dir / filename, p1, p2)
        answer = f"({p1[0] + 2}, {p1[1] + 1})"
        question = "What is the midpoint of the two marked points?"
        keywords = ["midpoint", str(p1[0] + 2), str(p1[1] + 1)]
        kb = ["kb_coordinate_midpoint"]
        answer_type = "coordinate"
    else:
        _plot_two_points(images_dir / filename, p1, p2)
        answer = "1"
        question = "What is the slope between the two marked points?"
        keywords = ["slope", "1"]
        kb = ["kb_coordinate_slope"]
        answer_type = "number"
    return _sample(sample_id, filename, question, answer, answer_type, keywords, kb, "coordinate", difficulty, split, images_dir)


def _make_table_sample(
    index: int, variant: int, split: str, difficulty: str, images_dir: Path
) -> dict[str, Any]:
    modes = ["read", "maximum", "compare", "total"]
    mode = modes[variant % len(modes)]
    sample_id = f"table_{index:03d}_{mode}"
    filename = f"{sample_id}.png"
    labels = ["A", "B", "C"]
    values = [5 + variant % 4, 7 + (variant * 2) % 5, 3 + (variant * 3) % 6]
    _plot_table(images_dir / filename, labels, values)
    if mode == "read":
        answer = str(values[1])
        question = "What is the value of category B in the table?"
        keywords = ["B", answer]
        kb = ["kb_table_read"]
        answer_type = "number"
    elif mode == "maximum":
        idx = int(np.argmax(values))
        answer = labels[idx]
        question = "Which category has the largest value in the table?"
        keywords = [answer, "largest"]
        kb = ["kb_table_max"]
        answer_type = "text"
    elif mode == "compare":
        answer = "A" if values[0] > values[2] else "C"
        question = "Which category is larger, A or C?"
        keywords = [answer, "larger"]
        kb = ["kb_table_compare"]
        answer_type = "text"
    else:
        answer = str(sum(values))
        question = "What is the total value in the table?"
        keywords = [answer, "total"]
        kb = ["kb_table_total"]
        answer_type = "number"
    return _sample(sample_id, filename, question, answer, answer_type, keywords, kb, "table", difficulty, split, images_dir)


def _build_knowledge_base() -> list[dict[str, Any]]:
    docs = [
        ("kb_line_slope", "Line slope", "For y = mx + b, the slope is m. Use slope to answer line rate-of-change questions."),
        ("kb_line_intercept", "Line y-intercept", "For y = mx + b, the y-intercept is b, the value where x=0."),
        ("kb_line_intersection", "Line x-axis intersection", "The x-axis intersection is where y=0."),
        ("kb_parabola_vertex", "Parabola vertex", "For y=(x-h)^2+k, the vertex is (h,k)."),
        ("kb_function_monotonic", "Function monotonicity", "A line with positive slope is increasing; a line with negative slope is decreasing."),
        ("kb_sine_period", "Sine period", "The standard sine curve sin(x) has period 2*pi."),
        ("kb_bar_max", "Bar chart maximum", "The maximum category is represented by the tallest bar."),
        ("kb_bar_min", "Bar chart minimum", "The minimum category is represented by the shortest bar."),
        ("kb_bar_difference", "Bar chart difference", "Difference means maximum value minus minimum value."),
        ("kb_bar_total", "Bar chart total", "The total value is the sum of all bar values."),
        ("kb_bar_ranking", "Bar chart ranking", "Ranking from largest to smallest sorts categories by descending value."),
        ("kb_line_chart_trend", "Line chart trend", "Overall increasing means values rise from the first point to the last point."),
        ("kb_line_chart_peak", "Line chart peak", "The peak value is the largest point in a line chart."),
        ("kb_line_chart_valley", "Line chart valley", "The valley value is the smallest point in a line chart."),
        ("kb_line_chart_growth", "Line chart growth", "Growth from first to last equals last value minus first value."),
        ("kb_pie_largest", "Pie chart largest share", "The largest pie slice has the greatest percentage share."),
        ("kb_pie_percentage", "Pie chart percentage", "A pie chart slice label gives the percentage for that category."),
        ("kb_pie_compare", "Pie chart category comparison", "Compare pie slices by their percentages."),
        ("kb_triangle_angle", "Triangle angle sum", "Angles in a triangle sum to 180 degrees."),
        ("kb_rectangle_area", "Rectangle area", "Rectangle area equals width times height."),
        ("kb_circle_area", "Circle area", "Circle area is pi times radius squared."),
        ("kb_derivative_power", "Derivative power rule", "The derivative of x^n is n*x^(n-1)."),
        ("kb_fraction_simplify", "Fraction simplification", "Simplify a fraction by dividing numerator and denominator by their greatest common divisor."),
        ("kb_solve_linear", "Solve linear equation", "For x+a=b, subtract a from both sides to get x=b-a."),
        ("kb_integral_power", "Simple integral", "The integral of 2x is x^2 + C."),
        ("kb_coordinate_point", "Coordinate point reading", "Read a coordinate point as (x,y)."),
        ("kb_coordinate_distance", "Coordinate distance", "Distance between two points is sqrt((x2-x1)^2+(y2-y1)^2)."),
        ("kb_coordinate_midpoint", "Coordinate midpoint", "Midpoint equals ((x1+x2)/2,(y1+y2)/2)."),
        ("kb_coordinate_slope", "Coordinate slope", "Slope between two points is (y2-y1)/(x2-x1)."),
        ("kb_table_read", "Table reading", "Read a table value by locating the requested row or category."),
        ("kb_table_max", "Table maximum", "The largest table category has the highest value."),
        ("kb_table_compare", "Table comparison", "Compare table categories by their numeric values."),
        ("kb_table_total", "Table total", "The total table value is the sum of all listed values."),
    ]
    return [
        {"id": doc_id, "title": title, "content": content, "source_image": None}
        for doc_id, title, content in docs
    ]


def _style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_line(path: Path, m: int, b: int, title: str) -> None:
    x = np.linspace(-4, 4, 100)
    y = m * x + b
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#2563eb", linewidth=2.5, label=title.replace("Line: ", ""))
    _style_axes(ax, title)
    ax.legend()
    _save(fig, path)


def _plot_parabola(path: Path, h: int, k: int) -> None:
    x = np.linspace(h - 3, h + 3, 200)
    y = (x - h) ** 2 + k
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x, y, color="#dc2626", linewidth=2.5)
    ax.scatter([h], [k], color="black", zorder=3)
    ax.annotate(f"vertex ({h},{k})", xy=(h, k), xytext=(h + 0.4, k + 1.5), arrowprops={"arrowstyle": "->"})
    _style_axes(ax, f"Parabola: y=(x-{h})^2+{k}")
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


def _plot_bar(path: Path, labels: list[str], values: list[int], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, values, color=["#64748b", "#14b8a6", "#f97316", "#6366f1"][: len(labels)])
    ax.set_ylim(0, max(values) + 3)
    ax.set_title(title)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.2, str(value), ha="center")
    _save(fig, path)

def _plot_line_chart(path: Path, values: list[int], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(1, len(values) + 1)
    ax.plot(x, values, marker="o", color="#16a34a", linewidth=2.5)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def _plot_pie(path: Path, labels: list[str], values: list[int], title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=[f"{label} {value}%" for label, value in zip(labels, values)], autopct=None)
    ax.set_title(title)
    _save(fig, path)


def _plot_triangle(path: Path, angle1: int, angle2: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    pts = np.array([[0, 0], [4, 0], [1.5, 3]])
    ax.add_patch(Polygon(pts, closed=True, fill=False, linewidth=2.5, edgecolor="#7c3aed"))
    ax.text(0.25, 0.18, f"{angle1}°", fontsize=13)
    ax.text(3.35, 0.18, f"{angle2}°", fontsize=13)
    ax.text(1.45, 2.45, "?", fontsize=16)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Triangle Angles")
    _save(fig, path)


def _plot_rectangle(path: Path, width: int, height: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="#2563eb", linewidth=2.5))
    ax.text(width / 2, -0.35, str(width), ha="center", fontsize=13)
    ax.text(width + 0.2, height / 2, str(height), va="center", fontsize=13)
    ax.set_xlim(-0.5, width + 1)
    ax.set_ylim(-0.8, height + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rectangle")
    _save(fig, path)


def _plot_circle(path: Path, radius: int) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    circle = Circle((0, 0), radius, fill=False, linewidth=2.5, edgecolor="#0891b2")
    ax.add_patch(circle)
    ax.plot([0, radius], [0, 0], color="#ef4444", linewidth=2)
    ax.text(radius / 2, 0.2, f"r={radius}", fontsize=13)
    ax.set_xlim(-radius - 0.5, radius + 0.5)
    ax.set_ylim(-radius - 0.5, radius + 0.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_title("Circle Radius")
    _save(fig, path)


def _plot_formula(path: Path, formula: str, subtitle: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.axis("off")
    ax.text(0.5, 0.58, formula, fontsize=28, ha="center", va="center")
    ax.text(0.5, 0.22, subtitle, fontsize=13, ha="center", va="center")
    _save(fig, path)


def _plot_point(path: Path, point: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlim(-1, 8)
    ax.set_ylim(-1, 8)
    ax.set_xticks(range(-1, 9))
    ax.set_yticks(range(-1, 9))
    _style_axes(ax, "Coordinate Point")
    ax.scatter([point[0]], [point[1]], s=80, color="#e11d48")
    ax.annotate(f"({point[0]},{point[1]})", xy=point, xytext=(point[0] + 0.2, point[1] + 0.4), arrowprops={"arrowstyle": "->"})
    _save(fig, path)


def _plot_two_points(path: Path, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 10)
    ax.set_xticks(range(-1, 11))
    ax.set_yticks(range(-1, 11))
    _style_axes(ax, "Two Points")
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=80, color="#9333ea")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#9333ea", alpha=0.5)
    ax.annotate(f"({p1[0]},{p1[1]})", xy=p1, xytext=(p1[0] + 0.2, p1[1] + 0.4))
    ax.annotate(f"({p2[0]},{p2[1]})", xy=p2, xytext=(p2[0] + 0.2, p2[1] + 0.4))
    _save(fig, path)


def _plot_legacy_scatter(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 5, 6, 8])
    ax.scatter(x, y, color="#0f766e", s=70)
    ax.plot(x, y, color="#0f766e", alpha=0.4)
    ax.set_title("Positive Correlation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    _save(fig, path)


def _plot_table(path: Path, labels: list[str], values: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=[[label, value] for label, value in zip(labels, values)],
        colLabels=["Category", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    ax.set_title("Table Values", fontsize=12)
    _save(fig, path)
