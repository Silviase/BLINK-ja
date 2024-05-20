CUDA_VISIBLE_DEVICES=1 python eval/test_benchmark.py --model_name SakanaAI/EvoVLM-JP-v1-7B --task_name Counting
CUDA_VISIBLE_DEVICES=4 python eval/test_benchmark.py --model_name stabilityai/japanese-instructblip-alpha --task_name Counting
CUDA_VISIBLE_DEVICES=5 python eval/test_benchmark.py --model_name Qwen/Qwen-VL-Chat --task_name all
CUDA_VISIBLE_DEVICES=6 python eval/test_benchmark.py --model_name turing-motors/heron-chat-git-ja-stablelm-base-7b-v1 --task_name all