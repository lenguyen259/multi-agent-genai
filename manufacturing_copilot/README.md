# Manufacturing Maintenance Copilot (real-data demo)

LangGraph router -> {hybrid RAG search | shift handover | ticket routing} over **real** rail-vehicle
maintenance logs (Hugging Face `ASNIJKDNI/SMRT-Maintenance-Logs`: 9,706 train / 1,079 test logs).
Retrieval = BM25 + MiniLM dense, fused with reciprocal-rank fusion. Answers are extractive and cite log ids.
No LLM key is needed; the router is rule-based and can be replaced by Azure OpenAI tool-calling.

## Run
```bash
mkdir -p data && python - <<'PY'
import urllib.request as u
b="https://huggingface.co/datasets/ASNIJKDNI/SMRT-Maintenance-Logs/resolve/main/data/"
for s in ("train","test"): u.urlretrieve(f"{b}{s}-00000-of-00001.parquet", f"data/{s}.parquet")
PY
pip install langgraph rank-bm25 sentence-transformers pandas pyarrow matplotlib
python evaluate.py        # results.json + figures/rag_retrieval_benchmark.png
python copilot_graph.py   # demo_outputs.json + figures/copilot_interaction.png
```

## Results
See `results.json`. Test logs that also appear verbatim in train (28%) are removed. The query is only the
fault report; the label is the component/action that fixed it. Latency depends on CPU load.
This dataset has maintenance logs only: no SOPs or 8D reports.
