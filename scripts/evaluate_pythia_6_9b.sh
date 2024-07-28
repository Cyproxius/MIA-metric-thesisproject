virtualenv --python python3 --system-site-packages $temp/env
$temp/env/bin/pip install transformers torch tqdm numpy datasets accelerate matplotlib
$temp/env/bin/python run.py --model "EleutherAI/pythia-6.9b" --dataset "swj0419/WikiMIA" --split_name "WikiMIA_length128"
$temp/env/bin/python run.py --model "EleutherAI/pythia-6.9b" --dataset "Cyproxius/GutenbergMIA_temporal" --split_name "GutenbergMIA_length128"
rm -rf $temp/env