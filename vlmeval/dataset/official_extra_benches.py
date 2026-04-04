import ast
import json
import os
import os.path as osp
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from .image_base import ImageBaseDataset
from .image_mcq import ImageMCQDataset
from ..smp import *
from ..smp.file import LMUDataRoot, dump, get_intermediate_file_path, load
from ..smp.misc import toliststr


def _ensure_zip_extracted(repo_path, zip_name, sentinel_name):
    repo_path = Path(repo_path)
    sentinel_path = repo_path / sentinel_name
    if sentinel_path.exists():
        return

    zip_path = repo_path / zip_name
    if not zip_path.exists():
        raise FileNotFoundError(f'Missing archive: {zip_path}')

    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(repo_path)

    sentinel_path.write_text('done', encoding='utf-8')


def _normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r'^[`\'"\s]+|[`\'"\s]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


class AllAnglesBench(ImageMCQDataset):
    TYPE = 'MCQ'
    DATASET_URL = {'All_Angles_Bench': ''}
    DATASET_MD5 = {}

    @staticmethod
    def _process_question(question, option_a, option_b, option_c):
        options = [option_a, option_b, option_c]
        question_no_options = str(question)
        for option in options:
            question_no_options = question_no_options.replace(str(option), '')
        question_no_options = question_no_options.strip().replace(',', '').strip()
        return question_no_options, options

    def load_data(self, dataset):
        data_root = Path(LMUDataRoot())
        data_root.mkdir(parents=True, exist_ok=True)
        data_path = data_root / f'{self.dataset_name}.tsv'
        if data_path.exists():
            self.data_path = str(data_path)
            return load(str(data_path))

        repo_path = Path(
            snapshot_download(
                repo_id='ch-chenyu/All-Angles-Bench',
                repo_type='dataset',
            )
        )

        raw_data = json.loads((repo_path / 'data.json').read_text(encoding='utf-8'))
        rows = []
        for entry in raw_data:
            question, options = self._process_question(
                entry['question'], entry['A'], entry['B'], entry['C']
            )
            image_paths = [str(repo_path / rel_path) for rel_path in entry['image_path']]
            rows.append(
                {
                    'index': int(entry['index']),
                    'folder': entry['folder'],
                    'category': entry['category'],
                    'pair_idx': entry['pair_idx'],
                    'sourced_dataset': entry.get('sourced_dataset', ''),
                    'image_path': image_paths,
                    'question': question,
                    'A': options[0],
                    'B': options[1],
                    'C': options[2],
                    'answer': entry['answer'],
                }
            )

        data = pd.DataFrame(rows)
        dump(data, str(data_path))
        self.data_path = str(data_path)
        return data


class SuperCLEVRBench(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'Super-CLEVR': ''}
    DATASET_MD5 = {}

    def load_data(self, dataset):
        data_root = Path(LMUDataRoot())
        data_root.mkdir(parents=True, exist_ok=True)
        data_path = data_root / f'{self.dataset_name}.tsv'
        if data_path.exists():
            self.data_path = str(data_path)
            return load(str(data_path))

        repo_path = Path(
            snapshot_download(
                repo_id='RyanWW/Super-CLEVR',
                repo_type='dataset',
                allow_patterns=['images.zip', 'superCLEVR_questions_30k.json'],
            )
        )
        _ensure_zip_extracted(repo_path, 'images.zip', '.superclevr_extracted')

        question_file = repo_path / 'superCLEVR_questions_30k.json'
        questions = json.loads(question_file.read_text(encoding='utf-8'))['questions']

        rows = []
        for item in questions:
            candidate_paths = [
                repo_path / 'images' / item['image_filename'],
                repo_path / item['image_filename'],
            ]
            image_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
            answer = item['answer']
            rows.append(
                {
                    'index': int(item['question_index']),
                    'image_path': str(image_path),
                    'question': item['question'],
                    'answer': str(answer),
                    'answer_type': type(answer).__name__,
                    'split': item.get('split', ''),
                    'template_filename': item.get('template_filename', ''),
                    'question_family_index': item.get('question_family_index', ''),
                    'question_hash': item.get('question_hash', ''),
                    'image_index': item.get('image_index', ''),
                }
            )

        data = pd.DataFrame(rows)
        dump(data, str(data_path))
        self.data_path = str(data_path)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        prompt = (
            f"Question: {line['question']}\n"
            "Answer with only the final answer. Do not add explanation."
        )
        if isinstance(tgt_path, list):
            msgs = [dict(type='image', value=p) for p in tgt_path]
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    @staticmethod
    def _normalize_answer(value, answer_type=None):
        text = _normalize_text(value)
        text = re.sub(r'^(answer|final answer)\s*[:\-]\s*', '', text)
        text = text.strip(' .')

        if answer_type == 'bool':
            if text in {'true', 'yes', 'y'}:
                return 'true'
            if text in {'false', 'no', 'n'}:
                return 'false'
        elif answer_type == 'int':
            match = re.search(r'-?\d+', text)
            if match:
                return str(int(match.group(0)))

        return text

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index')
        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer_type'] = [str(x) for x in data['answer_type']]
        data['norm_answer'] = [
            self._normalize_answer(ans, answer_type)
            for ans, answer_type in zip(data['answer'], data['answer_type'])
        ]
        data['norm_prediction'] = [
            self._normalize_answer(pred, answer_type)
            for pred, answer_type in zip(data['prediction'], data['answer_type'])
        ]
        data['hit'] = data['norm_answer'] == data['norm_prediction']

        summary = {
            'overall_accuracy': float(data['hit'].mean()) * 100.0,
            'num_samples': int(len(data)),
        }
        for template_name, group in data.groupby('template_filename'):
            if template_name:
                summary[f'{template_name}_accuracy'] = float(group['hit'].mean()) * 100.0

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(summary, score_file)
        acc_df = pd.DataFrame(
            [{'metric': key, 'value': value} for key, value in summary.items()]
        )
        dump(acc_df, acc_file)
        return summary


class TriBenchDataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {'TRI-Bench': ''}
    DATASET_MD5 = {}

    Q_KEYS = [
        'side_type',
        'angle_type',
        'ab_over_ac',
        'abs_b_minus_c_deg',
        'max_over_min_side',
        'angle_range_deg',
    ]

    PROMPT_TEXT = (
        'The image shows triangle ABC whose vertices are the centres of three small coloured square '
        'stickers: A=RED, B=YELLOW, C=BLUE.\n'
        'A light-brown masking-tape square border surrounds the scene in the same plane as triangle ABC.\n'
        'All questions refer to triangle ABC. Angles are in DEGREES. "angle ABC" denotes the interior angle at vertex B.\n'
        'Round all numeric answers to EXACTLY 4 decimals.\n\n'
        'Q1. Is triangle ABC equilateral, isosceles, or scalene?\n'
        'Q2. Is triangle ABC acute, right, or obtuse?\n'
        'Q3. In triangle ABC, by what factor is the length AB greater than length AC? (i.e., estimate AB / AC)\n'
        'Q4. In triangle ABC, by how much do angles angle ABC and angle ACB differ? '
        '(i.e., estimate |angle ABC - angle ACB| in degrees)\n'
        'Q5. In triangle ABC, what is (longest side) / (shortest side)?\n'
        'Q6. In triangle ABC, what is (largest interior angle - smallest interior angle) in degrees?\n\n'
        'Return STRICT JSON ONLY (no prose, markdown, code fences, or extra keys).\n'
        'Use EXACTLY these keys; numbers must have exactly 4 decimals:\n'
        '- "side_type": one of "equilateral", "isosceles", "scalene"\n'
        '- "angle_type": one of "acute", "right", "obtuse"\n'
        '- "ab_over_ac": number with 4 decimals\n'
        '- "abs_b_minus_c_deg": number with 4 decimals\n'
        '- "max_over_min_side": number with 4 decimals\n'
        '- "angle_range_deg": number with 4 decimals\n'
        'Output only the JSON object with these six keys and the computed values for THIS image.'
    )

    def load_data(self, dataset):
        data_root = Path(LMUDataRoot())
        data_root.mkdir(parents=True, exist_ok=True)
        data_path = data_root / f'{self.dataset_name}.tsv'
        if data_path.exists():
            self.data_path = str(data_path)
            return load(str(data_path))

        repo_path = Path(__file__).resolve().parents[2] / 'ref_repos' / 'Tri-Bench'
        if not repo_path.exists():
            raise FileNotFoundError(
                f'Tri-Bench repo not found at {repo_path}. Please clone the official repo first.'
            )

        tri_3d = pd.read_csv(repo_path / 'data' / 'tri_bench_triangles_3d.csv')
        px_2d = pd.read_csv(repo_path / 'data' / 'tri_bench_pixel_geometry_2d.csv')

        tri_3d_gt = tri_3d[['img_original', 'camera_view', 'object_in_square'] + self.Q_KEYS].copy()
        tri_3d_gt = tri_3d_gt.rename(columns={k: f'{k}_3d' for k in self.Q_KEYS})
        px_2d_gt = px_2d[['img_original'] + self.Q_KEYS].copy()
        px_2d_gt = px_2d_gt.rename(columns={k: f'{k}_2d' for k in self.Q_KEYS})
        merged = tri_3d_gt.merge(px_2d_gt, on='img_original', how='left')

        rows = []
        for idx, row in merged.reset_index(drop=True).iterrows():
            rows.append(
                {
                    'index': idx,
                    'image_path': str(repo_path / 'images' / row['img_original']),
                    'question': self.PROMPT_TEXT,
                    'img_original': row['img_original'],
                    'camera_view': row['camera_view'],
                    'object_in_square': row['object_in_square'],
                    **{f'{key}_3d': row[f'{key}_3d'] for key in self.Q_KEYS},
                    **{f'{key}_2d': row[f'{key}_2d'] for key in self.Q_KEYS},
                }
            )

        data = pd.DataFrame(rows)
        dump(data, str(data_path))
        self.data_path = str(data_path)
        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        msgs = [dict(type='text', value=line['question'])]
        if isinstance(tgt_path, list):
            msgs = [dict(type='image', value=p) for p in tgt_path] + msgs
        else:
            msgs = [dict(type='image', value=tgt_path)] + msgs
        return msgs

    @staticmethod
    def _extract_json(prediction):
        prediction = str(prediction).strip()
        prediction = re.sub(r'^```(?:json)?', '', prediction).strip()
        prediction = re.sub(r'```$', '', prediction).strip()
        match = re.search(r'\{.*\}', prediction, flags=re.S)
        if match is not None:
            prediction = match.group(0)

        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(prediction)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        return {}

    @staticmethod
    def _categorical_score(y_true, y_pred):
        yt = _normalize_text(y_true)
        yp = _normalize_text(y_pred)
        return float(yt == yp)

    @staticmethod
    def _ratio_score(y_true, y_pred):
        try:
            t = float(y_true)
            p = float(y_pred)
        except Exception:
            return 0.0
        if abs(t) <= 1e-8:
            return 0.0
        err = min(abs(p - t) / abs(t), 1.0)
        return 1.0 - err

    @staticmethod
    def _angle_score(y_true, y_pred):
        try:
            t = float(y_true)
            p = float(y_pred)
        except Exception:
            return 0.0
        err = min(abs(p - t) / 180.0, 1.0)
        return 1.0 - err

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file).sort_values(by='index').reset_index(drop=True)
        parsed = [self._extract_json(pred) for pred in data['prediction']]
        parsed_df = pd.DataFrame(parsed)
        for key in self.Q_KEYS:
            data[f'pred_{key}'] = parsed_df[key] if key in parsed_df else None

        for key in self.Q_KEYS:
            if key in {'side_type', 'angle_type'}:
                scorer = self._categorical_score
            elif key in {'ab_over_ac', 'max_over_min_side'}:
                scorer = self._ratio_score
            else:
                scorer = self._angle_score

            data[f'{key}_acc_3d'] = [
                scorer(gt, pred) for gt, pred in zip(data[f'{key}_3d'], data[f'pred_{key}'])
            ]
            data[f'{key}_acc_2d'] = [
                scorer(gt, pred) for gt, pred in zip(data[f'{key}_2d'], data[f'pred_{key}'])
            ]

        summary = {
            'overall_3d_accuracy': float(
                np.mean(data[[f'{key}_acc_3d' for key in self.Q_KEYS]].to_numpy())
            ) * 100.0,
            'overall_2d_accuracy': float(
                np.mean(data[[f'{key}_acc_2d' for key in self.Q_KEYS]].to_numpy())
            ) * 100.0,
            'num_images': int(len(data)),
        }
        for key in self.Q_KEYS:
            summary[f'{key}_3d_accuracy'] = float(data[f'{key}_acc_3d'].mean()) * 100.0
            summary[f'{key}_2d_accuracy'] = float(data[f'{key}_acc_2d'].mean()) * 100.0

        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        acc_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(summary, score_file)
        acc_df = pd.DataFrame(
            [{'metric': key, 'value': value} for key, value in summary.items()]
        )
        dump(acc_df, acc_file)
        return summary
