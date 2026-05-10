# 数据目录

`scripts/make_demo_data.py` 会生成可离线运行的合成 demo 数据：

- `data/demo/images/`：数学图表、几何图、公式截图等图片。
- `data/demo/qa.jsonl`：全量问答样本，默认生成 1000 条。
- `data/demo/qa_train.jsonl`：train split，默认 800 条，用于 LoRA 数据准备。
- `data/demo/qa_val.jsonl`：val split，默认 100 条。
- `data/demo/qa_test.jsonl`：test split，默认 100 条，建议用于最终 demo 评测。
- `data/demo/knowledge_base.jsonl`：检索知识库。
- `data/outputs/index.pkl`：`scripts/build_index.py` 生成的本地 TF-IDF 索引。

默认 smoke test 只依赖这些本地合成数据，不需要下载任何大模型。

这些数据用于原型流程验证，不是正式 benchmark。
