"""Experimental bounded-choice elimination; scores are conditional on finalists."""
import copy
import hashlib
import json
import random

from .options import positional_reasons
from .model import UnsupportedRow

# Shared by block training and tournament inference so both see the same subset prompt.
ABSENT_KEY = '__answer_not_in_subset__'
ABSENT_TEXT = '本候选集合不包含原题正确答案 / Original answer is absent from this subset'
ABSENT_INSTRUCTION = '\n候选为原题的子集；若正确答案不在集合内，选择“不包含原题正确答案”。'


def subset_request(request, keys, absent_option=False):
    """Copy of ``request`` restricted to ``keys``, optionally with the absent-answer option last."""
    request = copy.deepcopy(request)
    decision = request['questions']['decision']
    criteria = decision['criteria']
    if ABSENT_KEY in criteria:
        raise ValueError('reserved subset option key')
    decision['criteria'] = {k: criteria[k] for k in keys}
    if absent_option:
        decision['criteria'][ABSENT_KEY] = ABSENT_TEXT
        decision['instructions'] += ABSENT_INSTRUCTION
    return request


def tournament(model, row, block_size=None, top_k=2, rounds=3, batch_size=8, absent_option=False):
    """Eliminate in blocks; with ``absent_option`` every call uses the block-training subset format.

    The absent option is never a survivor or an answer. Its final-round mass is
    reported per round, and finalist scores are renormalized over real options.
    """
    if block_size is None:
        block_size = 25 if absent_option else 26
    if not 2 <= block_size + absent_option <= 26 or not 1 <= top_k < block_size or rounds < 1:
        raise ValueError('invalid tournament settings')
    if positional_reasons(row['request']):
        raise UnsupportedRow('cannot partition position-dependent options')
    opts = row['request']['questions']['decision']['criteria']
    if ABSENT_KEY in opts:
        raise UnsupportedRow('reserved subset option key')
    totals = dict.fromkeys(opts, 0.0)
    traces = []

    def real(scores):
        if ABSENT_KEY not in scores:
            return scores  # Plain format: keep the original scores exactly.
        scores = {k: v for k, v in scores.items() if k != ABSENT_KEY}
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()} if total > 0 else dict.fromkeys(scores, 1 / len(scores))

    for repeat in range(rounds):
        keys = list(opts)
        random.Random(repeat).shuffle(keys)
        stages = []
        while True:
            blocks = [keys[i:i + block_size] for i in range(0, len(keys), block_size)]
            candidates, requests = [], []
            for block in blocks:
                if len(block) == 1:
                    candidates.extend(block)
                    continue
                request = subset_request(row['request'], block, absent_option)
                uid = hashlib.sha256(json.dumps(request, ensure_ascii=False).encode()).hexdigest()
                requests.append({'id': uid, 'request': request})
            scores = model.probabilities(requests, batch_size=batch_size)
            if len(blocks) == 1:
                final = real(scores[0])
                for key, value in final.items():
                    totals[key] += value / rounds
                stage = {'finalists': keys}
                if absent_option:
                    stage['absent_probability'] = scores[0].get(ABSENT_KEY, 0.0)
                stages.append(stage)
                break
            for scores_in_block in scores:
                ranked = real(scores_in_block)
                candidates.extend(sorted(ranked, key=ranked.get, reverse=True)[:top_k])
            stages.append({'input_options': len(keys), 'survivors': len(candidates)})
            keys = candidates
        traces.append(stages)
    return {'status': 'ok', 'answer': max(totals, key=totals.get),
            'scores': totals, 'score_type': 'mean_finalist_conditional_probability',
            'absent_option': bool(absent_option), 'experimental': True, 'tournament': traces}


def strategy(n_options, tournament_above=None):
    """'tournament' above the threshold, else 'direct'. None disables the tournament."""
    if tournament_above is None:
        return 'direct'
    if not 26 <= tournament_above <= 52:
        raise ValueError('tournament_above must be between 26 and 52')
    return 'tournament' if n_options > tournament_above else 'direct'
