"""Reciprocal rank fusion; no gold labels are used in ranking."""


def reciprocal_rank_fusion(*rankings, constant=60):
    if constant <= 0:
        raise ValueError('constant must be positive')
    scores = {}
    for ranking in rankings:
        if len(ranking) != len(set(ranking)):
            raise ValueError('ranking contains duplicate document IDs')
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0/(constant+rank)
    return sorted(scores, key=lambda doc: (-scores[doc], doc))
