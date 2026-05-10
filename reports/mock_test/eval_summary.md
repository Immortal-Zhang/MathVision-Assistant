# MathVision-Assistant 评测摘要

本报告由 `scripts/run_eval.py` 自动生成，用于本地合成 demo 数据的原型阶段评测。

- qa_file: `data/demo/qa_test.jsonl`
- split: `test`
- num_samples: 15

| metric | value |
|---|---:|
| num_samples | 15 |
| exact_match | 1.0000 |
| numeric_match | 0.7333 |
| keyword_coverage | 0.5889 |
| retrieval_recall_at_k | 0.8667 |
| average_latency | 0.0016 |

## By task_type

| task_type | num_samples | exact_match | numeric_match | keyword_coverage | retrieval_recall_at_k | average_latency |
|---|---:|---:|---:|---:|---:|---:|
| bar_chart | 2 | 1.0000 | 0.5000 | 0.5000 | 1.0000 | 0.0011 |
| coordinate | 2 | 1.0000 | 1.0000 | 0.5833 | 1.0000 | 0.0012 |
| formula | 2 | 1.0000 | 1.0000 | 0.5833 | 1.0000 | 0.0050 |
| function_plot | 2 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 0.0011 |
| geometry | 1 | 1.0000 | 1.0000 | 0.5000 | 1.0000 | 0.0012 |
| line_chart | 2 | 1.0000 | 0.5000 | 0.7500 | 1.0000 | 0.0011 |
| pie_chart | 2 | 1.0000 | 0.5000 | 0.7500 | 0.5000 | 0.0010 |
| table | 2 | 1.0000 | 0.5000 | 0.5000 | 0.5000 | 0.0012 |

逐样本结果已保存到 `reports/mock_test/eval_results.csv`。

说明：`retrieval_recall_at_k` 是 evidence id 级召回，不等同于最终回答正确率；不同 backend 的结果会有差异，建议结合逐样本 CSV 查看错误样本。
