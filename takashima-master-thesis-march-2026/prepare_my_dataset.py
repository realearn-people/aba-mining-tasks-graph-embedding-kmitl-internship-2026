"""
prepare_my_dataset.py
=====================
Converts hotel_contrary_dataset_support.csv into the directory structure
expected by takashima-master-thesis-march-2026.

Run from the repo root:
    python prepare_my_dataset.py --input hotel_contrary_dataset_support.csv

Output layout (created automatically):
    data/
    ├── input/
    │   └── aba_nodes.csv                            ← all unique nodes + readable text
    └── output/
        ├── Silver_Staff_ContP_BodyN_4omini.csv      ← PN sheet (CONTRARY_TO + NOT_CONTRARY)
        ├── Silver_Staff_ContN_BodyP_4omini.csv      ← NP sheet (CONTRARY_TO + NOT_CONTRARY)
        ├── Silver_Staff_ContP_BodyP_4omini.csv      ← PP sheet (NOT_CONTRARY only)
        ├── Silver_Staff_ContN_BodyN_4omini.csv      ← NN sheet (NOT_CONTRARY only)
        ├── Silver_Price_ContP_BodyN_4omini.csv
        ├── Silver_Price_ContN_BodyP_4omini.csv
        ├── Silver_Price_ContP_BodyP_4omini.csv
        ├── Silver_Price_ContN_BodyN_4omini.csv
        ├── Silver_CheckIn_ContP_BodyN_4omini.csv
        ├── Silver_CheckIn_ContN_BodyP_4omini.csv
        ├── Silver_CheckIn_ContP_BodyP_4omini.csv
        ├── Silver_CheckIn_ContN_BodyN_4omini.csv
        ├── Silver_CheckOut_ContP_BodyN_4omini.csv
        ├── Silver_CheckOut_ContN_BodyP_4omini.csv
        ├── Silver_CheckOut_ContP_BodyP_4omini.csv
        ├── Silver_CheckOut_ContN_BodyN_4omini.csv
        └── support_edges.csv                        ← inference/support edges

Sheet type legend:
    PN  Contrary(P)Body(N)  — positive contrary assumption, negative body
    NP  Contrary(N)Body(P)  — negative contrary assumption, positive body
    PP  Contrary(P)Body(P)  — both positive  → always NOT_CONTRARY
    NN  Contrary(N)Body(N)  — both negative  → always NOT_CONTRARY

After running this script, follow with:
    python src/preprocess/save_graph.py
"""

import argparse
import csv
import os
from collections import defaultdict


# ── helpers ────────────────────────────────────────────────────────────────────

def node_to_text(node_name: str) -> str:
    """
    Convert a symbolic node name to readable text for BERT.
    'no_evident_not_good_check-out_time' → 'good check out time'
    """
    text = node_name.replace("_", " ").replace("-", " ")
    for prefix in ["no evident not ", "have evident "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip()


def load_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v.strip() if v else "") for k, v in row.items()})
    return rows


def save_csv(path: str, rows: list, fieldnames: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):,} rows → {path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="hotel_contrary_dataset_support.csv",
                        help="Path to your CSV file")
    parser.add_argument("--drop-duplicates", action="store_true", default=True,
                        help="Exclude rows marked DUPLICATE (default: True)")
    args = parser.parse_args()

    print(f"\nLoading {args.input} ...")
    rows = load_csv(args.input)
    print(f"  Total rows: {len(rows):,}")

    # ── 1. Optionally drop duplicates ──────────────────────────────────────────
    last_col = list(rows[0].keys())[-1]
    if args.drop_duplicates:
        before = len(rows)
        rows = [r for r in rows if r.get(last_col, "UNIQUE") != "DUPLICATE"]
        print(f"  After dropping DUPLICATE rows: {len(rows):,} (removed {before - len(rows):,})")

    # ── 2. Collect all unique nodes ────────────────────────────────────────────
    all_nodes = {}
    for r in rows:
        for col in ("head", "tail"):
            name = r[col]
            if name and name not in all_nodes:
                all_nodes[name] = node_to_text(name)

    print(f"\nUnique nodes: {len(all_nodes):,}")
    node_rows = [{"node_id": name, "text": text} for name, text in sorted(all_nodes.items())]
    save_csv("data/input/aba_nodes.csv", node_rows, fieldnames=["node_id", "text"])

    # ── 3. Separate support rows and edge rows ─────────────────────────────────
    # Support rows have an empty sheet_type in the CSV
    support_rows = [r for r in rows if r["relation"] == "SUPPORT"]
    # All PN / NP / PP / NN rows (sheet_type is non-empty)
    edge_rows    = [r for r in rows if r["sheet_type"] != ""]

    print(f"\nEdge rows (PN+NP+PP+NN): {len(edge_rows):,}")
    print(f"  CONTRARY_TO:   {sum(1 for r in edge_rows if r['relation'] == 'CONTRARY_TO'):,}")
    print(f"  NOT_CONTRARY:  {sum(1 for r in edge_rows if r['relation'] == 'NOT_CONTRARY'):,}")
    print(f"Support rows:    {len(support_rows):,}")

    # ── 4. Domain and sheet_type mappings ──────────────────────────────────────
    domain_map = {
        "staff":      "Staff",
        "price":      "Price",
        "check-in":   "CheckIn",
        "check-out":  "CheckOut",
    }

    # Maps sheet_type value → output filename suffix
    sheet_type_map = {
        "Contrary(P)Body(N)": "ContP_BodyN",
        "Contrary(N)Body(P)": "ContN_BodyP",
        "Contrary(P)Body(P)": "ContP_BodyP",
        "Contrary(N)Body(N)": "ContN_BodyN",
    }

    out_fields = ["head", "relation", "tail", "vote", "domain", "sheet_type"]

    def to_out(rows):
        return [
            {
                "head":       r["head"],
                "relation":   r["relation"],
                "tail":       r["tail"],
                "vote":       r["vote"],
                "domain":     r["domain"],
                "sheet_type": r["sheet_type"],
            }
            for r in rows
        ]

    # ── 5. Save one silver CSV per domain × sheet_type ────────────────────────
    print("\nSaving silver-label CSVs to data/output/ ...")
    summary = []

    for domain_key, domain_label in domain_map.items():
        domain_rows = [r for r in edge_rows if r["domain"] == domain_key]

        for sheet_type, file_suffix in sheet_type_map.items():
            sheet_rows = [r for r in domain_rows if r["sheet_type"] == sheet_type]
            if not sheet_rows:
                continue

            path = f"data/output/Silver_{domain_label}_{file_suffix}_4omini.csv"
            save_csv(path, to_out(sheet_rows), out_fields)

            ct  = sum(1 for r in sheet_rows if r["relation"] == "CONTRARY_TO")
            nct = sum(1 for r in sheet_rows if r["relation"] == "NOT_CONTRARY")
            summary.append((f"{domain_label} {file_suffix}", len(sheet_rows), ct, nct))

    # ── 6. Save support (inference) edges ─────────────────────────────────────
    save_csv("data/output/support_edges.csv",
             [{"head": r["head"], "relation": r["relation"], "tail": r["tail"],
               "domain": r["domain"]} for r in support_rows],
             ["head", "relation", "tail", "domain"])

    # ── 7. Print summary ───────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  {'Sheet':35s}  {'Total':>7}  {'CONTRARY':>9}  {'NOT_CONTR':>10}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*9}  {'-'*10}")
    for label, total, ct, nct in summary:
        print(f"  {label:35s}  {total:7,}  {ct:9,}  {nct:10,}")

    total_ct  = sum(ct  for _, _, ct,  _ in summary)
    total_nct = sum(nct for _, _, _,  nct in summary)
    print(f"\n  Total CONTRARY_TO:  {total_ct:,}")
    print(f"  Total NOT_CONTRARY: {total_nct:,}")
    print(f"  Total unique nodes: {len(all_nodes):,}")
    print(f"  Support edges:      {len(support_rows):,}")
    print("\nDone! Next step:")
    print("  python src/preprocess/save_graph.py")
    print()


if __name__ == "__main__":
    main()
