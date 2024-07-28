virtualenv --python python3 --system-site-packages $temp/env
$temp/env/bin/pip install transformers torch tqdm numpy datasets accelerate matplotlib
$temp/env/bin/python run.py --model "EleutherAI/pythia-12b" --dataset "swj0419/WikiMIA" --split_name "WikiMIA_length128" --learning_rates [0.0000001, 0.0000005, 0.000001, 0.000005, 0.00001] --batch_sizes [1,8,16] --unlearning_steps [4]
rm -rf $temp/env