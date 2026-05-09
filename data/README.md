# 数据目录

`scripts/make_demo_data.py` 会生成可离线运行的合成 demo 数据：

- `data/demo/images/`：数学图表、几何图、公式截图等图片。
- `data/demo/qa.jsonl`：问答评测集。
- `data/demo/knowledge_base.jsonl`：检索知识库。
- `data/outputs/index.pkl`：`scripts/build_index.py` 生成的本地 TF-IDF 索引。

默认 smoke test 只依赖这些本地合成数据，不需要下载任何大模型。
