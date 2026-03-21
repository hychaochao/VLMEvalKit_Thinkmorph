import string

import pandas as pd

from .image_mcq import ImageMCQDataset
from ..smp.file import load
from ..smp.misc import toliststr


class MMSIBench(ImageMCQDataset):
    TYPE = 'MCQ'

    MMSI_URL = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MMSIBench_wo_circular.tsv'

    DATASET_URL = {
        'MMSI': MMSI_URL,
        'MMSIBench': MMSI_URL,
        'MMSIBench_wo_circular': MMSI_URL,
    }
    DATASET_MD5 = {
        'MMSI': '548c5f33f1a12948d5355d5f600749e4',
        'MMSIBench': '548c5f33f1a12948d5355d5f600749e4',
        'MMSIBench_wo_circular': '548c5f33f1a12948d5355d5f600749e4',
    }

    def _task_category(self):
        return [
            'Pos-Cam-Cam',
            'Pos-Obj-Obj',
            'Pos-Reg-Reg',
            'Pos-Cam-Obj',
            'Pos-Obj-Reg',
            'Pos-Cam-Reg',
            'Attr-Meas',
            'Attr-Appr',
            'Motion-Cam',
            'Motion-Obj',
            'MSR',
        ]

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line['image_path']) if self.meta_only else self.dump_image(line)
        question = line['question']
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }

        prompt = ''
        if 'hint' in line and not pd.isna(line['hint']):
            prompt += f"Hint: {line['hint']}\n"
        prompt += f'{question}\n'
        if options:
            prompt += 'Options: ' + ', '.join(f'{key}: {value}' for key, value in options.items()) + '\n'
        prompt += "Answer with the option's letter from the given choices directly. Enclose the option's letter within ``."

        messages = []
        if isinstance(tgt_path, list):
            messages.extend(dict(type='image', value=path) for path in tgt_path)
        else:
            messages.append(dict(type='image', value=tgt_path))
        messages.append(dict(type='text', value=prompt))
        return messages

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=build_mcq_score_fn(**judge_kwargs),
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'MMSIBench'),
        )
