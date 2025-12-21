# VLMEvalKit for Thinkmorph

This repository provides evaluation support for the ThinkMorph model based on [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).


## Installation

### 1. Clone the Repository

```
git clone https://github.com/hychaochao/VLMEvalKit_Thinkmorph.git
cd VLMEvalKit_Thinkmorph
```

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Install ThinkMorph Dependencies

Make sure you have the necessary dependencies for ThinkMorph


## Model Configuration

Edit the model configuration in [`vlmeval/config.py`](vlmeval/config.py):
```
thinkmorph_series = {
    "thinkmorph": partial(
        ThinkMorph, 
        model_path="ThinkMorph/ThinkMorph-7B", 
        think=True, # Enable thinking mode
        understanding_output=False, # If `False`, enables interleaved reasoning and requires `save_dir` to save generated images
        temperature=0.3, 
        max_think_token_n=4096, 
        save_dir="path_to_your_imgs_dir" # Directory to save generated images
    ),
}
```

## Quick Start

1️⃣ Configure API Keys

export OPENAI_API_KEY="your_api_key_here"

2️⃣ Choose the benchmark you wanna eval, and run.

An example script is shown in [run_thinkmorph.sh](run_thinkmorph.sh). 
If  you wanna try more benchmark, check [VLMEvalKit Features](https://aicarrier.feishu.cn/wiki/Qp7wwSzQ9iK1Y6kNUJVcr6zTnPe?table=tblsdEpLieDoCxtb&view=vewUH3kRGs)

```
python run.py \
    --data VSP_maze_task_main_original VisPuzzle ChartQA_h_bar ChartQA_v_bar VStarBench BLINK_Jigsaw MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D \
    --model thinkmorph \
    --judge gpt-5 \
    --work-dir ./results
```



