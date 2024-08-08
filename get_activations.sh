# python get_activations.py llama_7B tqa_mc2 --device 2 
# python get_activations.py llama_7B tqa_gen_end_q --device 2 
# python get_activations.py llama_7B tqa_gen --device 2 

# CUDA_VISIBLE_DEVICES=0 python get_activations.py alpaca_7B tqa_mc2 --device 0 
# CUDA_VISIBLE_DEVICES=0 python get_activations.py alpaca_7B tqa_gen_end_q --device 0 
# CUDA_VISIBLE_DEVICES=0 python get_activations.py vicuna_7B tqa_mc2 --device 0 
# CUDA_VISIBLE_DEVICES=0 python get_activations.py vicuna_7B tqa_gen_end_q --device 0 

CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_7B tqa_mc2 --device 0
CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_7B tqa_gen_end_q --device 0

# CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_13B tqa_mc2 --device 0
# CUDA_VISIBLE_DEVICES=0 python get_activations.py llama2_chat_13B tqa_gen_end_q --device 0

# python get_activations.py llama2_chat_70B tqa_mc2 --device 0
# python get_activations.py llama2_chat_70B tqa_gen_end_q --device 0