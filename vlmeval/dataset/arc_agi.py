import ast
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .image_base import ImageBaseDataset
from ..smp.file import LMUDataRoot, dump, get_intermediate_file_path, load


ARC_COLORS = {
    0: (0, 0, 0),
    1: (0, 116, 217),
    2: (255, 65, 54),
    3: (46, 204, 64),
    4: (255, 220, 0),
    5: (170, 170, 170),
    6: (240, 18, 190),
    7: (255, 133, 27),
    8: (127, 219, 255),
    9: (135, 12, 37),
}


def _normalize_grid(value):
    if isinstance(value, list):
        grid = value
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith('```'):
            parts = text.split('```')
            text = ''
            for part in reversed(parts):
                part = part.strip()
                if part and part.lower() != 'json':
                    text = part
                    break
        candidate_texts = [text]
        if '{' in text and '}' in text:
            candidate_texts.append(text[text.find('{'):text.rfind('}') + 1])
        if '[' in text and ']' in text:
            candidate_texts.append(text[text.find('['):text.rfind(']') + 1])

        grid = None
        for candidate in candidate_texts:
            try:
                parsed = ast.literal_eval(candidate)
            except Exception:
                try:
                    parsed = json.loads(candidate)
                except Exception:
                    continue
            if isinstance(parsed, list):
                grid = parsed
                break
            if isinstance(parsed, dict):
                grid = parsed
                break
    else:
        grid = None

    if isinstance(grid, dict):
        normalized = {}
        for key, value in grid.items():
            normalized[key] = _normalize_grid(value)
        return normalized

    if not isinstance(grid, list) or not grid:
        return None
    if not all(isinstance(row, list) and row for row in grid):
        return None

    try:
        normalized = [[int(cell) for cell in row] for row in grid]
    except Exception:
        return None
    width = len(normalized[0])
    if any(len(row) != width for row in normalized):
        return None
    if any(cell < 0 or cell > 9 for row in normalized for cell in row):
        return None
    return normalized


def _render_grid(grid, label, cell_size=18, padding=8):
    font = ImageFont.load_default()
    width = len(grid[0])
    height = len(grid)
    label_height = 16
    image = Image.new('RGB', (width * cell_size + padding * 2, height * cell_size + padding * 2 + label_height), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((padding, 0), label, fill='black', font=font)
    top = padding + label_height
    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            x0 = padding + col_idx * cell_size
            y0 = top + row_idx * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle([x0, y0, x1, y1], fill=ARC_COLORS[int(cell)], outline=(180, 180, 180))
    return image


def _compose_task_image(task):
    sections = []
    for idx, pair in enumerate(task['train'], start=1):
        sections.append((_render_grid(pair['input'], f'Train {idx} Input'), _render_grid(pair['output'], f'Train {idx} Output')))

    test_images = []
    for idx, pair in enumerate(task['test'], start=1):
        test_images.append(_render_grid(pair['input'], f'Test {idx} Input'))

    row_gap = 16
    col_gap = 16
    pair_gap = 28
    max_width = 0
    total_height = 24

    row_specs = []
    for input_image, output_image in sections:
        row_width = input_image.width + output_image.width + pair_gap
        row_height = max(input_image.height, output_image.height)
        row_specs.append(('pair', input_image, output_image, row_width, row_height))
        max_width = max(max_width, row_width)
        total_height += row_height + row_gap

    for test_image in test_images:
        row_specs.append(('test', test_image, None, test_image.width, test_image.height))
        max_width = max(max_width, test_image.width)
        total_height += test_image.height + row_gap

    canvas = Image.new('RGB', (max_width + 32, total_height + 8), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 8), 'ARC-AGI task', fill='black', font=ImageFont.load_default())

    y = 24
    for kind, left_image, right_image, row_width, row_height in row_specs:
        x = 16
        canvas.paste(left_image, (x, y))
        if kind == 'pair':
            x += left_image.width + pair_gap // 2
            draw.text((x - 8, y + row_height // 2 - 6), '->', fill='black', font=ImageFont.load_default())
            x += pair_gap // 2
            canvas.paste(right_image, (x, y))
        y += row_height + row_gap

    return canvas


class ARCAGIDataset(ImageBaseDataset):
    TYPE = 'VQA'

    @classmethod
    def supported_datasets(cls):
        return ['ARC-AGI']

    @staticmethod
    def _repo_root():
        return Path(__file__).resolve().parents[2] / 'ref_repos' / 'ARC-AGI-2'

    @staticmethod
    def _task_dir():
        return ARCAGIDataset._repo_root() / 'data' / 'evaluation'

    def _build_question(self, num_tests):
        lines = [
            'Infer the output grid for every test input in the ARC task image.',
            'Return ONLY valid JSON.',
        ]
        if num_tests == 1:
            lines.append('Format: [[row1], [row2], ...]')
        else:
            mapping = ', '.join(f'"test_{idx}"' for idx in range(1, num_tests + 1))
            lines.append(f'Format: {{{mapping}: [[...]]}}')
        lines.append('Every grid cell must be an integer between 0 and 9.')
        return '\n'.join(lines)

    def load_data(self, dataset):
        assert dataset == 'ARC-AGI'
        cache_path = Path(LMUDataRoot()) / 'ARC-AGI.tsv'
        image_root = Path(LMUDataRoot()) / 'images' / 'ARC-AGI'
        image_root.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            return load(str(cache_path))

        rows = []
        for task_file in sorted(self._task_dir().glob('*.json')):
            task = json.load(open(task_file, 'r', encoding='utf-8'))
            task_id = task_file.stem
            image_path = image_root / f'{task_id}.png'
            if not image_path.exists():
                _compose_task_image(task).save(image_path)

            answer = {}
            for idx, pair in enumerate(task['test'], start=1):
                key = f'test_{idx}'
                answer[key] = pair['output']

            rows.append({
                'index': task_id,
                'task_id': task_id,
                'question': self._build_question(len(task['test'])),
                'image_path': str(image_path),
                'answer': json.dumps(answer if len(answer) > 1 else answer['test_1'], ensure_ascii=False),
                'num_tests': len(task['test']),
            })

        data = pd.DataFrame(rows)
        dump(data, str(cache_path))
        return data

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        hits = []
        parsed_predictions = []

        for _, row in data.iterrows():
            expected_raw = json.loads(row['answer'])
            expected = _normalize_grid(expected_raw)
            predicted = _normalize_grid(str(row.get('prediction', '')))

            if isinstance(expected, dict):
                is_hit = isinstance(predicted, dict) and expected == predicted
            else:
                if isinstance(predicted, dict) and 'test_1' in predicted:
                    predicted = predicted['test_1']
                is_hit = expected == predicted

            hits.append(1.0 if is_hit else 0.0)
            parsed_predictions.append(json.dumps(predicted, ensure_ascii=False) if predicted is not None else '')

        data = data.copy()
        data['pred_parsed'] = parsed_predictions
        data['hit'] = hits

        detail_file = get_intermediate_file_path(eval_file, '_results')
        dump(data, detail_file)

        summary = {
            'overall_task_accuracy': float(sum(hits) / len(hits) * 100.0) if hits else 0.0,
            'num_tasks': int(len(hits)),
        }
        score_file = get_intermediate_file_path(eval_file, '_score', 'json')
        dump(summary, score_file)
        return summary
