"""run_v19.py -- version 19 on the competition match: the rawnet hybrid, trained on other matches.

    uv run python solution/run_v19.py              # uses solution/weights/rawnet/hybrid40_final_*
    uv run python solution/run_v19.py --no-score   # do not print the public 5-minute scores
    uv run python evaluation/validate_submission.py evaluation/results/rawnet_v19.csv

Writes evaluation/results/rawnet_v19.csv.

The model is raw_model.py with the "hybrid40" settings: a set encoder over the players and the ball
of each frame, plus the version 17 feature row, then a dilated TCN; 3 seeds trained on all 29
outside matches (uv run python solution/raw_model.py final --tag hybrid40 --seeds 0,1,2
--set hybrid=true --epochs 40). The decoder is the Viterbi of version 17 with switch cost 3.

The model, its settings and the switch cost were chosen on held-out outside matches (two folds,
6 optical matches): selection score 0.693, version 17 0.673 (weights/rawnet/report.json). On
those matches the mean of rawnet and the version 17 models scores the same (0.694); the
submission is rawnet alone.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "evaluation")]
import possession_metrics as pm  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
import raw_model  # noqa: E402

TAG = "hybrid40"
SWITCH_COST = 3.0
DEVICE = torch.device("cpu")   # prediction only; see load()
PROB_DECIMALS = 4              # see decode()
OUT_PATH = os.path.join(ROOT, "evaluation", "results", "rawnet_v19.csv")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def show(name, gt, codes):
    result = pm.evaluate(gt, codes)
    print("%-22s macroF1 %.4f  spellF1 %.4f  acc %.4f  alive/dead F1 %.4f  team acc %.4f"
          "  poss %d (truth %d)" % (
              name, result["macro_f1"], result["spell_f1_mean"], result["accuracy"],
              result["alive_dead_f1"], result["team_acc_gt_alive"],
              result["n_pred_spells"], result["n_gt_spells"]))


def load():
    """(models, feature statistics, device) of the final rawnet models.

    Prediction always runs on the CPU, whatever GPU the machine has: the file must be the same
    on every machine, and GPU kernels (CUDA, Apple MPS) round differently from each other and
    from the CPU. The networks may have been trained on any device."""
    path = os.path.join(raw_model.WEIGHTS_DIR, TAG + "_final_%s")
    with open(path % "cfg.json") as handle:
        saved = json.load(handle)
    dev = DEVICE
    models = []
    for seed in saved["seeds"]:
        state = torch.load(path % ("seed%d.pt" % seed), map_location=dev)
        # the width of the version 17 feature row comes from the saved weights
        model = raw_model.RawNet(saved["cfg"], state["hyb.0.weight"].shape[1]).to(dev)
        model.load_state_dict(state)
        models.append(model.eval())
    stats = raw_model.HybridStats.load(path % "hybrid_stats.npz")
    return models, stats, dev


def probabilities():
    """(match, p_dead, p_home) of the competition match, as the networks give them."""
    models, stats, dev = load()
    match = raw_model.competition_match()
    p_dead, p_home = raw_model.Predictor(models, dev, stats).match(match)
    return match, p_dead, p_home


def decode(match, p_dead, p_home, decimals=PROB_DECIMALS):
    """Codes from the probabilities, rounded first to `decimals` (None: not rounded). Rounding
    absorbs the last-digit differences between processors before they reach the decoder."""
    if decimals is not None:
        p_dead = np.round(p_dead.astype(np.float64), decimals)
        p_home = np.round(p_home.astype(np.float64), decimals)
    return raw_model.decode(p_dead, p_home, match["halves"], match["fps"], SWITCH_COST)


def predict():
    """(codes, p_dead, p_home) for the competition match."""
    match, p_dead, p_home = probabilities()
    return decode(match, p_dead, p_home), p_dead, p_home


def main(argv=None):
    parser = argparse.ArgumentParser(description="version 19: rawnet hybrid")
    parser.add_argument("--no-score", action="store_true", help="do not print the test scores")
    args = parser.parse_args(argv)
    codes, _, _ = predict()
    graded = codes != pm.X
    print("%d possessions | share of graded frames: H %.1f%%  A %.1f%%  D %.1f%%" % (
        (len(pm.possession_spells(codes, pm.valid_runs(codes))),)
        + tuple(100 * (codes[graded] == c).mean() for c in (pm.H, pm.A, pm.D))))
    write_submission(OUT_PATH, np.array(pm.LABELS)[codes])
    print("wrote", os.path.relpath(OUT_PATH, ROOT))
    if not args.no_score:
        print("\nTEST:")
        show("public 5 minutes", pm.label_codes(pm.read_label_csv(PUBLIC_GT)), codes)


if __name__ == "__main__":
    main()
