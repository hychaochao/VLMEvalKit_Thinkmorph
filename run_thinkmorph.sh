#!/bin/bash

cd ./VLMEvalKit_Thinkmorph

export OPENAI_API_KEY=

 python run.py \
 --data VSP_maze_task_main_original VisPuzzle ChartQA_h_bar ChartQA_v_bar VStarBench BLINK_Jigsaw MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D  \
 --model thinkmorph \
 --judge gpt-5 \
 --work-dir ./VLMEvalKit_Thinkmorph/results

