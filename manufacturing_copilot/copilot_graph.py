"""LangGraph copilot over real maintenance logs: router -> {hybrid RAG search | shift handover | ticket routing}.

No LLM key is used here: the router is rule-based and answers are extractive with cited log ids, so every
output below is reproducible. Swap `route`/`compose` for Azure OpenAI tool-calling without changing the graph.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from typing import TypedDict

from langgraph.graph import END, StateGraph

from hybrid_rag import HybridIndex, label_names, load_index, split_log

OWNERS = {  # keyword -> owning team (ticket routing)
    r"door|gangway|plunger|latch": "Doors & Gangway",
    r"aircon|a/c|freon|evaporator|ccu|condenser": "HVAC",
    r"brake|bcu|acu|air dryer|compressor|valve|transducer": "Brakes & Pneumatics",
    r"traction|contactor|resist|mcu|k\d? contactor|battery|charging": "Traction & Power",
    r"light|ballast|led|headlight": "Lighting",
    r"smoke|fire|detector": "Safety Systems",
}


class State(TypedDict, total=False):
    question: str
    route: str
    answer: str
    citations: list[str]
    latency_ms: float


def build(index: HybridIndex, names: list[str]):
    def route(s: State) -> State:
        q = s["question"].lower()
        if "handover" in q or "shift" in q:
            return {"route": "handover"}
        if "ticket" in q or "assign" in q:
            return {"route": "ticket"}
        return {"route": "search"}

    def search(s: State) -> State:
        t = time.perf_counter()
        top = index.search(s["question"], 5)
        votes = Counter(int(index.labels[i]) for i in top)
        action = names[votes.most_common(1)[0][0]]
        _, tail = split_log(index.texts[top[0]])
        out = (f"Most likely action: {action}. Closest past case #{top[0]}: {tail[:230]}")
        return {"answer": out, "citations": [f"log#{i}" for i in top[:3]],
                "latency_ms": (time.perf_counter() - t) * 1000}

    def ticket(s: State) -> State:
        fault = re.sub(r"(?i)^.*?(ticket|assign)[^:]*:", "", s["question"]).strip()
        owner = next((o for pat, o in OWNERS.items() if re.search(pat, fault.lower())), "General maintenance")
        top = index.search(fault, 3)
        return {"answer": f"Ticket routed to: {owner}. Suggested first action: {names[int(index.labels[top[0]])]}.",
                "citations": [f"log#{i}" for i in top], "latency_ms": 0.0}

    def handover(s: State) -> State:
        recent = list(range(len(index.texts) - 40, len(index.texts)))
        c = Counter(names[int(index.labels[i])] for i in recent)
        lines = [f"- {n} x{k}" for n, k in c.most_common(4)]
        return {"answer": "Shift handover (last 40 logs): top recurring actions\n" + "\n".join(lines),
                "citations": [f"log#{i}" for i in recent[-3:]], "latency_ms": 0.0}

    g = StateGraph(State)
    for n, f in (("route", route), ("search", search), ("ticket", ticket), ("handover", handover)):
        g.add_node(n, f)
    g.set_entry_point("route")
    g.add_conditional_edges("route", lambda s: s["route"], {"search": "search", "ticket": "ticket", "handover": "handover"})
    for n in ("search", "ticket", "handover"):
        g.add_edge(n, END)
    return g.compile()


if __name__ == "__main__":
    import json
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx, tr, te = load_index()
    names = label_names()
    app = build(idx, names)
    held = split_log(te.text.iloc[3])[0]
    qs = [held, f"Create a ticket: {split_log(te.text.iloc[10])[0]}", "Give me the shift handover"]
    outs = [app.invoke({"question": q}) for q in qs]
    json.dump([{"q": q, **{k: v for k, v in o.items()}} for q, o in zip(qs, outs)], open("demo_outputs.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(9, 7.2), dpi=200); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0.955), 1, 0.045, color="#1f5fd1", transform=ax.transAxes))
    ax.text(0.015, 0.968, "Maintenance Copilot  ·  LangGraph + hybrid RAG  ·  real SMRT maintenance logs", color="w", fontsize=9, weight="bold", transform=ax.transAxes)
    y = 0.93
    for q, o in zip(qs, outs):
        qt = textwrap.fill(q, 95); at = "\n".join(textwrap.fill(p, 95) for p in o["answer"].split("\n"))
        ax.text(0.99, y, qt, ha="right", va="top", fontsize=7.5, bbox=dict(fc="#1f5fd1", ec="none", boxstyle="round,pad=0.5"), color="w", transform=ax.transAxes)
        y -= 0.04 + 0.022 * qt.count("\n") + 0.03
        ax.text(0.01, y, at, ha="left", va="top", fontsize=7.5, bbox=dict(fc="w", ec="#c5d3ef", boxstyle="round,pad=0.5"), transform=ax.transAxes)
        y -= 0.05 + 0.022 * at.count("\n") + 0.03
        ax.text(0.01, y, f"route: {o['route']}   sources: {', '.join(o['citations'])}   {o['latency_ms']:.0f} ms", fontsize=7, color="#1f5fd1", transform=ax.transAxes)
        y -= 0.07
    fig.savefig("figures/copilot_interaction.png", bbox_inches="tight")
    print(json.dumps(outs, indent=1)[:1500])
