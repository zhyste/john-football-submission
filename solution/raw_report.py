"""raw_report.py -- collect the rawnet fold results into report.json, copy the chosen config's
probabilities to the delivery names, and sanity-check the competition probabilities WITHOUT labels.

    uv run python solution/raw_report.py --chosen base --cost 4

No label of the competition match is read: the competition statistics are shares of the
probabilities and the number of decoded spells, compared with the same statistics on the held-out
outside DFL / Metrica matches (seen ball).
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import raw_model as rm  # noqa: E402
import possession_metrics as pm  # noqa: E402

PROBS = os.path.join(ROOT, "external_data", "probs", "rawnet")
WEIGHTS = os.path.join(HERE, "weights", "rawnet")


def stats(p_dead, p_home, halves, fps, cost):
    inside = np.zeros(len(p_dead), bool)
    for _, a, b in halves:
        inside[a:b + 1] = True
    codes = rm.decode(p_dead, p_home, halves, fps, cost)
    spells = len(pm.possession_spells(codes, pm.valid_runs(codes)))
    minutes = inside.sum() / fps / 60
    alive = inside & (p_dead <= 0.5)
    return dict(share_dead=float((p_dead[inside] > 0.5).mean()),
                share_home_alive=float((p_home[alive] > 0.5).mean()),
                mean_p_dead=float(p_dead[inside].mean()),
                decoded_dead_share=float((codes[inside] == rm.D).mean()),
                spells=spells, spells_per_min=float(spells / minutes),
                p_home_uncertain=float((np.abs(p_home[alive] - 0.5) < 0.15).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen", required=True)
    ap.add_argument("--cost", required=True)
    args = ap.parse_args()
    report = {"configs": {}, "chosen": {"tag": args.chosen, "switch_cost": float(args.cost)},
              "v17_reference_foldA": 0.653}
    for path in sorted(glob.glob(os.path.join(WEIGHTS, "fold_*.json"))):
        with open(path) as f:
            r = json.load(f)
        entry = report["configs"].setdefault(r["tag"], {"cfg": r["cfg"], "folds": {}})
        entry["folds"][r["fold"]] = dict(
            selection=r["selection"], per_match=r["scores"], seeds=r["seeds"],
            early_history=[[h["early"] for h in hist] for hist in r["history"]])
    for tag, e in report["configs"].items():
        if {"A", "B"} <= set(e["folds"]):
            e["mean_AB"] = {c: (e["folds"]["A"]["selection"][c] + e["folds"]["B"]["selection"][c])
                            / 2 for c in e["folds"]["A"]["selection"]}
    # delivery names for the chosen config
    for path in glob.glob(os.path.join(PROBS, args.chosen + "_fold*.npz")):
        shutil.copy(path, os.path.join(PROBS, os.path.basename(path)[len(args.chosen) + 1:]))
    comp = os.path.join(PROBS, args.chosen + "_competition.npz")
    sanity = {}
    if os.path.exists(comp):
        shutil.copy(comp, os.path.join(PROBS, "competition.npz"))
        match = rm.competition_match()
        z = np.load(comp)
        sanity["competition"] = stats(z["p_dead"], z["p_home"], match["halves"], match["fps"],
                                      float(args.cost))
        v17 = os.path.join(PROBS, "v17_competition_for_sanity.npz")
        if os.path.exists(v17):
            z17 = np.load(v17)
            sanity["competition_v17"] = stats(z17["p_dead"], z17["p_home"], match["halves"],
                                              match["fps"], float(args.cost))
    for fold, names in (("A", ["metrica_game2", "dfl_J03WMX", "dfl_J03WN1"]),
                        ("B", ["metrica_game1", "dfl_J03WOH", "dfl_J03WOY"])):
        for name in names:
            path = os.path.join(PROBS, "%s_fold%s_%s_seen.npz" % (args.chosen, fold, name))
            if not os.path.exists(path):
                continue
            match = rm.load_match(name)
            z = np.load(path)
            s = stats(z["p_dead"], z["p_home"], match["halves"], match["fps"], float(args.cost))
            lab = match["label"]
            graded = lab != rm.X
            s["true_dead_share"] = float((lab[graded] == rm.D).mean())
            s["true_home_share_alive"] = float((lab[graded & (lab != rm.D)] == rm.H).mean())
            s["true_spells_per_min"] = float(len(pm.possession_spells(lab, pm.valid_runs(lab)))
                                             / (graded.sum() / match["fps"] / 60))
            sanity["%s_%s" % (fold, name)] = s
    report["sanity"] = sanity
    with open(os.path.join(PROBS, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(WEIGHTS, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    for tag, e in report["configs"].items():
        for fold, r in sorted(e["folds"].items()):
            c = max(r["selection"], key=r["selection"].get)
            print("%-10s fold %s best %.3f @ %s | at chosen cost %s: %.3f" % (
                tag, fold, r["selection"][c], c, args.cost, r["selection"].get(str(float(
                    args.cost)), float("nan"))))
    for k, s in sanity.items():
        print(k, json.dumps({a: round(b, 3) for a, b in s.items()}))


if __name__ == "__main__":
    main()
