# Code for paper "Semantically-Consistent, Distributional Intervention Policies for Language Models (SLIM)"

See the details in our paper here: [Paper Link](paper/SLIM.pdf)

Abstract
> Language models are prone to occasionally undesirable generations, such as harmful or toxic content, despite their impressive capability to produce texts that appear accurate and coherent. In this paper, we present a new three-stage approach to detect and mitigate undesirable content generations by rectifying activations. First, we train an ensemble of layer-wise classifiers to detect undesirable content using activations by minimizing a smooth surrogate of the risk-aware score. Second, for contents that are detected as undesirable, we propose layer-wise distributional intervention policies that perturb the attention heads minimally while guaranteeing probabilistically the effectiveness of the intervention. Finally, we aggregate layer-wise interventions to minimize the semantic shifts of the false detection, which is achieved by aligning aggregations based on semantic preference data. Benchmarks on several language models and datasets show that our method outperforms baselines in reducing the generation of undesirable output.

## Installation
```bash
conda env create -f environment.yaml
conda activate slim
python -m ipykernel install --user --name iti --display-name "slim"
```

Create folders:
```bash
mkdir -p validation/results_dump/summary_dump/test 
mkdir -p validation/results_dump/summary_dump/val
mkdir -p validation/answer_dump/summary_dump/test
mkdir -p validation/answer_dump/summary_dump/val
```

## Run experiment

Run:
```bash
CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 15 --device 0 --num_fold 2 --use_center_of_mass
CUDA_VISIBLE_DEVICES=0 python validate_2fold.py llama_7B --num_heads 48 --alpha 0 --device 0 --num_fold 2 --use_center_of_mass
CUDA_VISIBLE_DEVICES=0 python choose_edited_layer.py llama_7B --dataset_name tqa_mc2 --device 0 --num_fold 2 --bl 1.0
```

Check sweep.sh to find the optimal value for kappa

Run test:
```bash
CUDA_VISIBLE_DEVICES=0 python ot_edit_layer.py llama_7B  --device 0 --num_fold 2 --bl 1.0 --alpha 0.1 --kappa 40 --use_mode test
CUDA_VISIBLE_DEVICES=0 python ot_edit_layer.py llama_7B  --device 0 --num_fold 2 --bl 1.0 --alpha 0.1 --kappa 30 --use_mode test
CUDA_VISIBLE_DEVICES=0 python ot_edit_layer.py llama3_instruct_8B  --device 0 --num_fold 2 --bl 1.0 --alpha 0.1 --kappa 20 --use_mode test --judge_name ft:davinci-002:personal::9RixJXan --info_name ft:davinci-002:personal::9RiwXB4G
```
