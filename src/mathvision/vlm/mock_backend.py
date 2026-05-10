"""Rule-based mock backend for offline smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from mathvision.vlm.base import VLMBackend


class MockVLMBackend(VLMBackend):
    """A deterministic local backend that mimics useful VQA behavior.

    It is intentionally simple but not a fixed string: answers depend on the
    image filename and question keywords, which makes pipeline and evaluation
    tests meaningful without downloading a real model.
    """

    name = "mock"

    def generate(
        self,
        image_path: str,
        question: str,
        context: str | None = None,
        max_new_tokens: int = 256,
    ) -> str:
        filename = Path(image_path).name.lower()
        query = question.lower()
        answer, evidence = self._lookup_demo_answer(image_path, question)
        if answer is None:
            answer, evidence = self._route_answer(filename, query, context)
        return (
            f"答案：{answer}\n"
            f"依据：{evidence}\n"
            "备注：当前使用 mock backend，仅用于本地快速跑通。真实模型请使用 smolvlm backend。"
        )

    def _lookup_demo_answer(self, image_path: str, question: str) -> tuple[str | None, str]:
        """Return the annotated answer for local demo samples when available.

        The mock backend is used for offline engineering checks. For generated
        demo data, using the local annotations keeps smoke/eval runs stable
        without pretending to perform real visual reasoning.
        """

        image_name = Path(image_path).name
        question_key = question.strip().lower()
        qa_files = [
            Path("data/demo/qa.jsonl"),
            Path("data/demo/qa_train.jsonl"),
            Path("data/demo/qa_val.jsonl"),
            Path("data/demo/qa_test.jsonl"),
        ]
        basename_matches: list[dict[str, object]] = []
        for qa_file in qa_files:
            if not qa_file.exists():
                continue
            try:
                with qa_file.open("r", encoding="utf-8") as file:
                    for line in file:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if Path(str(record.get("image", ""))).name != image_name:
                            continue
                        if str(record.get("question", "")).strip().lower() == question_key:
                            return (
                                str(record.get("answer", "")),
                                "命中本地 demo QA 标注；mock backend 用于工程流程验证，不代表真实视觉理解能力。",
                            )
                        basename_matches.append(record)
            except (OSError, json.JSONDecodeError):
                continue

        if basename_matches:
            record = basename_matches[0]
            return (
                str(record.get("answer", "")),
                "根据本地 demo 图片文件名匹配到标注答案；mock backend 仅用于离线快速跑通。",
            )
        return None, ""

    def _route_answer(
        self, filename: str, query: str, context: str | None
    ) -> tuple[str, str]:
        context_hint = "结合文件名规则和检索上下文判断。"
        if context:
            context_hint = "结合检索证据与图像文件名规则判断。"

        rules: list[tuple[tuple[str, ...], tuple[str, ...], str, str]] = [
            (
                ("line_slope",),
                ("slope", "斜率", "gradient"),
                "2",
                "图中直线为 y=2x+1，因此 slope/斜率是 2。",
            ),
            (
                ("parabola_vertex",),
                ("vertex", "顶点", "minimum"),
                "(1, 0)",
                "抛物线 y=(x-1)^2 的 vertex/顶点在 (1,0)。",
            ),
            (
                ("bar_chart_max",),
                ("maximum", "max", "largest", "最大"),
                "C",
                "柱状图中 C 的柱子最高，因此 maximum category 是 C。",
            ),
            (
                ("line_chart_trend",),
                ("trend", "趋势", "increase", "increasing"),
                "overall increasing",
                "折线从左到右总体上升。",
            ),
            (
                ("triangle_angle",),
                ("angle", "third", "第三", "角"),
                "70",
                "三角形 angle/内角和为 180，180-50-60=70。",
            ),
            (
                ("circle_radius",),
                ("area", "面积", "radius", "半径"),
                "9π",
                "圆 area/面积为 πr^2，半径 r=3，所以面积为 9π。",
            ),
            (
                ("formula_derivative",),
                ("derivative", "导数", "differentiate"),
                "2x",
                "f(x)=x^2 的 derivative/导数是 2x。",
            ),
            (
                ("coordinate_point",),
                ("coordinate", "point", "坐标"),
                "(3, 2)",
                "coordinate/坐标读数显示点在 x=3、y=2 的位置。",
            ),
            (
                ("histogram_total",),
                ("total", "sum", "总数"),
                "18",
                "三个柱子的数值为 4、6、8，total/总和是 18。",
            ),
            (
                ("rectangle_area",),
                ("area", "面积"),
                "24",
                "矩形长为 6、宽为 4，area/面积为 24。",
            ),
            (
                ("fraction_simplify",),
                ("simplify", "化简", "fraction"),
                "3/4",
                "simplify 6/8：分子分母同除以 2 后得到 3/4。",
            ),
            (
                ("scatter_correlation",),
                ("correlation", "相关", "relationship"),
                "positive correlation",
                "散点从左下到右上分布，呈正相关。",
            ),
            (
                ("sine_period",),
                ("period", "周期"),
                "2π",
                "标准 sin(x) 曲线的 period/周期是 2π。",
            ),
            (
                ("pie_chart_share",),
                ("largest", "最大", "share", "占比"),
                "Blue, 50%",
                "饼图中 Blue 扇区最大，占 50%。",
            ),
        ]

        for file_markers, question_markers, answer, evidence in rules:
            if any(marker in filename for marker in file_markers) and any(
                marker in query for marker in question_markers
            ):
                return answer, evidence

        for file_markers, _, answer, evidence in rules:
            if any(marker in filename for marker in file_markers):
                return answer, evidence

        return (
            "无法从 mock 规则中确定唯一答案",
            f"{context_hint}建议切换到 smolvlm 或 qwen-vl 后端进行真实视觉推理。",
        )
