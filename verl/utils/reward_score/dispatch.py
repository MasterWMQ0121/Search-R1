"""Lightweight dispatch from dataset names to rule-based reward scorers."""

from verl.utils.reward_score import qa_em


def _select_rm_score_fn(data_source):
    if data_source in [
        'nq',
        'nq_phase1_smoke',
        'triviaqa',
        'popqa',
        'hotpotqa',
        '2wikimultihopqa',
        'musique',
        'bamboogle',
    ]:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError
