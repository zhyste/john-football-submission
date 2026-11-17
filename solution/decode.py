"""Turn per-frame probabilities into possessions.

spellF1 charges a flicker as a split possession, so a frame-by-frame answer cannot be submitted
as it stands. A three-state Viterbi over each half finds the label sequence that best fits the
probabilities, paying a fixed cost for every change of state. The cost is chosen on held-out
outside matches (ext_model.py, raw_model.py); ext_model.decode() calls this module.
"""

import numpy as np

FLOOR = 1e-4


def costs(p_dead, p_home):
    """-log P for H, A, D per frame, as plain Python lists for the loop below."""
    p_dead = np.clip(p_dead, FLOOR, 1 - FLOOR)
    p_home = np.clip(p_home, FLOOR, 1 - FLOOR)
    alive = np.log(1 - p_dead)
    return np.stack([-(alive + np.log(p_home)), -(alive + np.log(1 - p_home)),
                     -np.log(p_dead)], axis=1).tolist()


def viterbi(cost, switch):
    """Cheapest H/A/D path, paying `switch` for every change of state.

    Plain Python on purpose: three states, so a loop over floats is far faster than numpy
    calls on length-3 arrays, and it needs no compiler.
    """
    n = len(cost)
    back = [None] * n
    total = list(cost[0])
    for t in range(1, n):
        c = cost[t]
        best = min(total)
        arg = total.index(best)
        step, row = [0.0, 0.0, 0.0], [0, 0, 0]
        for s in range(3):
            stay, move = total[s], best + switch
            if stay <= move:
                step[s], row[s] = stay + c[s], s
            else:
                step[s], row[s] = move + c[s], arg
        total, back[t] = step, row
    state = total.index(min(total))
    path = [0] * n
    for t in range(n - 1, -1, -1):
        path[t] = state
        if t:
            state = back[t][state]
    return path
