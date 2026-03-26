from collections import OrderedDict

import pandas as pd

from .matching_func import can_match_option
from ....smp.file import dump, get_intermediate_file_path


def compute_mcq_score(df: pd.DataFrame) -> pd.DataFrame:
    preds_extracted = []
    hits = []

    for _, row in df.iterrows():
        pred_raw = str(row.get('prediction', ''))
        gt_raw = str(row.get('answer', '')).strip()
        pred = can_match_option(pred_raw)
        gt = can_match_option(gt_raw)
        preds_extracted.append(pred)
        hits.append(1.0 if pred and gt and pred == gt else 0.0)

    scored = df.copy()
    scored['pred_extracted'] = preds_extracted
    scored['hit'] = hits
    return scored


def build_mcq_score_fn(**judge_kwargs):
    def score_fn(df: pd.DataFrame) -> pd.DataFrame:
        return compute_mcq_score(df)

    score_fn.judge_mode = 'rule'
    score_fn.judge_model = judge_kwargs.get('model', 'extract_matching')
    return score_fn


def _ordered_categories(scored: pd.DataFrame, group_col: str, order=None):
    if group_col not in scored.columns:
        return []
    present = [cat for cat in scored[group_col].dropna().unique().tolist()]
    if order is None:
        return present
    preferred = [cat for cat in order if cat in present]
    remaining = [cat for cat in present if cat not in preferred]
    return preferred + remaining


def eval_mcq_score(
    *,
    load_fn,
    eval_file: str,
    score_fn,
    group_col='category',
    order=None,
    dataset_name='MCQ',
):
    data = load_fn(eval_file)
    if 'index' in data.columns:
        data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]

    scored = score_fn(data.copy())

    detail_path = get_intermediate_file_path(eval_file, '_extract_matching', 'xlsx')
    dump(scored, detail_path)

    summary = OrderedDict()
    summary['overall'] = float(scored['hit'].mean() * 100.0) if len(scored) else 0.0

    if isinstance(group_col, str):
        group_cols = [group_col]
    else:
        group_cols = list(group_col)

    for gc in group_cols:
        categories = _ordered_categories(scored, gc, order if gc == group_cols[0] else None)
        prefix = '' if len(group_cols) == 1 else f'{gc}.'
        for category in categories:
            subset = scored[scored[gc] == category]
            if len(subset):
                summary[f'{prefix}{category}_accuracy'] = float(subset['hit'].mean() * 100.0)

    score_path = get_intermediate_file_path(eval_file, '_score', 'json')
    dump(summary, score_path)
    print(f'[{dataset_name}] summary: {summary}')
    return summary
