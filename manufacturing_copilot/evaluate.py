"""Held-out retrieval benchmark on real maintenance logs. Writes results.json + figures/."""
import json, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hybrid_rag import load_index, split_log, label_names

idx, tr, te = load_index()
names = label_names()
dup = te.text.isin(set(tr.text)).mean()
print(f"exact-duplicate test texts also in train: {dup:.3%}")
te = te[~te.text.isin(set(tr.text))].reset_index(drop=True)   # no leakage
queries = [split_log(t)[0] for t in te.text]                   # only the fault REPORT is the query
gold = te.label.to_numpy()
res = {"n_test": len(te), "dup_removed_frac": float(dup)}
lat = {}
for mode in ("bm25", "dense", "hybrid"):
    hits = {1: 0, 3: 0, 5: 0}
    times = []
    for q, g in zip(queries, gold):
        t = time.perf_counter()
        top = idx.search(q, 5, mode)
        times.append((time.perf_counter() - t) * 1000)
        labs = [int(idx.labels[i]) for i in top]
        for k in hits:
            hits[k] += int(g in labs[:k])
    res[mode] = {f"hit@{k}": v / len(gold) for k, v in hits.items()}
    res[mode]["latency_ms_p50"] = float(np.percentile(times, 50))
    res[mode]["latency_ms_p95"] = float(np.percentile(times, 95))
    lat[mode] = times
    print(mode, res[mode])
json.dump(res, open("results.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
x = np.arange(3)
for j, (mode, c) in enumerate((("bm25", "#9db7e8"), ("dense", "#e08a2c"), ("hybrid", "#1f5fd1"))):
    vals = [res[mode][f"hit@{k}"] for k in (1, 3, 5)]
    ax[0].bar(x + (j - 1) * 0.27, vals, 0.27, color=c, label=mode)
    for xi, v in zip(x + (j - 1) * 0.27, vals):
        ax[0].text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
ax[0].set_xticks(x, ["hit@1", "hit@3", "hit@5"]); ax[0].set_ylim(0, 1)
ax[0].set_title(f"Correct action in top-k  (n={len(te)} real held-out logs)", fontsize=9); ax[0].legend(frameon=False, fontsize=8)
ax[1].boxplot([lat[m] for m in ("bm25", "dense", "hybrid")], tick_labels=["bm25", "dense", "hybrid"], showfliers=False)
ax[1].axhline(450, color="#c0392b", ls="--", lw=1); ax[1].text(0.55, 460, "450 ms", color="#c0392b", fontsize=8)
ax[1].set_title("Retrieval latency per query (ms, CPU, incl. query embedding)", fontsize=9)
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figures/rag_retrieval_benchmark.png")
