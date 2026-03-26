import ast
import os

from huggingface_hub import snapshot_download
from tqdm import tqdm

from .image_mcq import ImageMCQDataset
from ..smp.file import load
from ..smp.misc import get_cache_path, modelscope_flag_set, toliststr


class MindCubeBench(ImageMCQDataset):
    TYPE = 'MCQ'

    RAW_URL = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_raw_qa.tsv'
    TINY_URL = 'https://huggingface.co/datasets/lmms-lab-si/EASI-Leaderboard-Data/resolve/main/MindCubeBench_tiny_raw_qa.tsv'

    DATASET_URL = {
        'MindCube': RAW_URL,
        'MindCubeBench': RAW_URL,
        'MindCubeBench_raw_qa': RAW_URL,
        'MindCubeBench_tiny_raw_qa': TINY_URL,
    }
    DATASET_MD5 = {
        'MindCube': '6a53cd353bc93d8e3a87098249c806ad',
        'MindCubeBench': '6a53cd353bc93d8e3a87098249c806ad',
        'MindCubeBench_raw_qa': '6a53cd353bc93d8e3a87098249c806ad',
        'MindCubeBench_tiny_raw_qa': '35f69fc30d7c2d2880417ce0769f5347',
    }

    def _task_category(self):
        return ['rotation', 'among', 'around']

    def prepare_tsv(self, url, file_md5=None, repo_id='MLL-Lab/MindCube'):
        data = super().prepare_tsv(url, file_md5)

        sentinel_name = '.mindcubebench_extracted'
        cache_path = get_cache_path(repo_id)
        if cache_path and os.path.isdir(cache_path) and os.path.isfile(os.path.join(cache_path, sentinel_name)):
            dataset_path = cache_path
        else:
            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            zip_files = sorted(
                os.path.join(dataset_path, filename)
                for filename in os.listdir(dataset_path)
                if filename.endswith('.zip')
            )
            for zip_file in tqdm(zip_files, desc='Unpacking MindCube data'):
                import zipfile

                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(dataset_path)
            with open(os.path.join(dataset_path, sentinel_name), 'w', encoding='utf-8') as f:
                f.write('done')

        if 'image_path' in data.columns:
            def fix_one(value):
                if not isinstance(value, str):
                    return value
                normalized = os.path.expanduser(os.path.expandvars(value.strip()))
                return os.path.normpath(os.path.join(dataset_path, normalized.lstrip(r'\/')))

            def to_abs(value):
                if isinstance(value, list):
                    return [fix_one(item) for item in value]
                if isinstance(value, str) and value.strip().startswith('[') and value.strip().endswith(']'):
                    try:
                        parsed = ast.literal_eval(value)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, list):
                        return [fix_one(item) for item in parsed]
                return fix_one(value)

            data['image_path'] = data['image_path'].map(to_abs)

        return data

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = toliststr(line['image_path']) if self.meta_only else self.dump_image(line)
        prompt = line['input_prompt']
        images = tgt_path if isinstance(tgt_path, list) else [tgt_path]
        parts = prompt.split('<image>')

        messages = []
        for idx, part in enumerate(parts):
            part = part.strip()
            if part:
                messages.append(dict(type='text', value=part))
            if idx < len(parts) - 1 and idx < len(images):
                messages.append(dict(type='image', value=images[idx]))
        return [message for message in messages if message['value']]

    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.spatial_bench.cal_scores import build_mcq_score_fn, eval_mcq_score

        return eval_mcq_score(
            load_fn=load,
            eval_file=eval_file,
            score_fn=build_mcq_score_fn(**judge_kwargs),
            group_col='category',
            order=self._task_category(),
            dataset_name=getattr(self, 'dataset_name', 'MindCubeBench'),
        )
