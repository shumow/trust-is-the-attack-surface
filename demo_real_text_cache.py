"""Real-text cache reliance sanity check.

This is not a transformer experiment. It runs the same global+cache toy substrate on
a committed natural-prose fixture so claims about "real text is sparse / earned
trust" have a reproducible local measurement instead of living only in prose.

Run:  python3 demo_real_text_cache.py
Writes results/real_text_cache.csv and results/real_text_cache.png.
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from demo_utils import result_path
from markov_cache import BackoffModel, evaluate, evaluate_predictor, make_predictor
from text_adapter import encode_text, to_documents


TEXT_FILE = "data/sample_prose.txt"
GLOBAL_ORDER = 2
DIR_A = 1.0
G_ALPHA = 0.1
CACHE_ORDERS = (1, 2, 3)
TOKEN_LEVELS = {
    "char": 240,
    "word": 80,
}


def load_docs(level):
    with open(TEXT_FILE, encoding="utf-8") as f:
        text = f.read()
    adapter, ids, _ = encode_text(text, level)
    docs = to_documents(ids, TOKEN_LEVELS[level])
    if len(docs) < 3:
        raise ValueError(f"not enough {level} documents in {TEXT_FILE}")
    split = max(1, int(len(docs) * 0.65))
    return adapter.V, docs[:split], docs[split:]


def measure():
    rows = []
    for level in TOKEN_LEVELS:
        V, train, test = load_docs(level)
        gm = BackoffModel(GLOBAL_ORDER, V).fit(train)
        for order in CACHE_ORDERS:
            bits_global, _ = evaluate(
                test, gm, cache_order=order, method="global",
                weight=0.0, g_alpha=G_ALPHA,
            )
            bits_cache, _ = evaluate(
                test, gm, cache_order=order, method="dirichlet",
                weight=DIR_A, g_alpha=G_ALPHA,
            )
            _, _, reliance_curve = evaluate_predictor(
                test,
                make_predictor(
                    gm, approach="dirichlet", cache_order=order,
                    weight=DIR_A, g_alpha=G_ALPHA,
                ),
                reliance=True,
            )
            rows.append({
                "token_level": level,
                "vocab": V,
                "cache_order": order,
                "test_docs": len(test),
                "global_bits": bits_global,
                "cache_bits": bits_cache,
                "benefit_bits": bits_global - bits_cache,
                "final_reliance": float(reliance_curve[-1]),
            })
    return rows


def write_csv(rows):
    path = result_path("real_text_cache.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, level in zip(axes, TOKEN_LEVELS):
        subset = [r for r in rows if r["token_level"] == level]
        orders = [r["cache_order"] for r in subset]
        reliance = [r["final_reliance"] for r in subset]
        benefit = [r["benefit_bits"] for r in subset]
        ax.bar([o - 0.18 for o in orders], reliance, width=0.34, label="final reliance")
        ax.bar([o + 0.18 for o in orders], benefit, width=0.34, label="benefit bits")
        ax.axhline(0, color="black", lw=0.7)
        ax.set_title(f"{level}-level fixture")
        ax.set_xlabel("cache order")
        ax.set_xticks(orders)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("value")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = result_path("real_text_cache.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    rows = measure()
    for r in rows:
        print(
            f"{r['token_level']:>4} order={r['cache_order']} "
            f"V={r['vocab']} benefit={r['benefit_bits']:+.4f} "
            f"final_reliance={r['final_reliance']:.4f}"
        )
    write_csv(rows)
    plot(rows)


if __name__ == "__main__":
    main()
