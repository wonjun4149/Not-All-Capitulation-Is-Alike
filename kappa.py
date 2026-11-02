"""Human/judge agreement on the two annotation strata.

    python kappa.py --spec runs.json

The human labels live in strat_a.csv (random, the paper's kappa) and
strat_b.csv (boundary, adversarial by construction). The judge label is NOT
read from the CSV: the `_judge_raw` column there holds the five-question pass
that was current when the sample was drawn, before `claim_explicitly_rejected`
was added. The label compared against the human is the final one, taken from
the judged JSONL and joined on (model, slot_id, level, fallacy, turn), so this
script re-derives the reported numbers from the released data rather than from
a stored column.

The annotator saw the response only: judge label and sub-decisions were hidden
in the `_` columns.
"""
import argparse
import csv
import json
from collections import Counter


def read_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def judged_labels(spec):
    """(model, slot_id, level, fallacy, turn) -> final label."""
    table = {}
    for t in spec["targets"]:
        for path in (t["pressure"], t["control"]):
            for rec in read_jsonl(path):
                for turn in rec["turns"]:
                    key = (t["name"], rec["slot_id"], rec["level"],
                           rec["fallacy"], turn["turn"])
                    table[key] = turn["label"]
    return table


def kappa(pairs):
    """Cohen's kappa over (human, judge) label pairs."""
    n = len(pairs)
    if not n:
        return 0.0, 0.0
    po = sum(1 for h, j in pairs if h == j) / n
    h_marg = Counter(h for h, _ in pairs)
    j_marg = Counter(j for _, j in pairs)
    pe = sum(h_marg[k] * j_marg[k] for k in set(h_marg) | set(j_marg)) / (n * n)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def load(path, table):
    rows, missing = [], 0
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["slot_id"], r["level"], r["fallacy"], int(r["turn"]))
            if key not in table:
                missing += 1
                continue
            rows.append((r["human_label"].strip(), table[key]))
    return rows, missing


def matrix(pairs):
    labels = ["MAINTAIN", "SEMANTIC_REFRAME", "FICTIONAL", "EVASIVE",
              "FACTUAL_YIELD", "REFUSED"]
    seen = {h for h, _ in pairs} | {j for _, j in pairs}
    labels = [l for l in labels if l in seen]
    cell = Counter(pairs)
    w = max(len(l) for l in labels) + 2
    print("  human \\ judge".ljust(w) + "".join(l[:12].rjust(13) for l in labels))
    for h in labels:
        print("  " + h.ljust(w - 2) + "".join(str(cell[(h, j)]).rjust(13) for j in labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="runs.json")
    ap.add_argument("--strata", nargs="+", default=["strat_a.csv", "strat_b.csv"])
    a = ap.parse_args()

    table = judged_labels(json.load(open(a.spec, encoding="utf-8")))

    for path in a.strata:
        rows, missing = load(path, table)
        print("=" * 78)
        print(f"{path}: {len(rows)} turns joined to the judged files"
              + (f", {missing} not found" if missing else ""))
        po, k = kappa(rows)
        print(f"  agreement {po:.3f}   kappa {k:.3f}")
        print()
        matrix(rows)
        print()
        # The annotator used REFUSED as a label; the paper reports the mapped
        # variant and gives all three in the appendix.
        if any(h == "REFUSED" for h, _ in rows):
            mapped = [("MAINTAIN" if h == "REFUSED" else h,
                       "MAINTAIN" if j == "REFUSED" else j) for h, j in rows]
            dropped = [(h, j) for h, j in rows if "REFUSED" not in (h, j)]
            for name, sub in (("mapped to MAINTAIN", mapped),
                              ("turns excluded    ", dropped),
                              ("kept distinct     ", rows)):
                p_o, k_v = kappa(sub)
                print(f"  REFUSED {name} : agreement {p_o:.3f}  kappa {k_v:.3f}"
                      f"  (n={len(sub)})")
            print()


if __name__ == "__main__":
    main()
