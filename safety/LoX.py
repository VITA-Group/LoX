from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--save-path", type=str, default="./output")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=0) # Top-ranks to extrapolate. k=0 extrapolates full rank.
parser.add_argument("--coef", type=float, default=1.) # Extrapolation coefficient

args = parser.parse_args()
print(args)

def main():
    k = args.k
    coef = args.coef
    aligned_path = args.model

    tokenizer = AutoTokenizer.from_pretrained(aligned_path)
    aligned_model = AutoModelForCausalLM.from_pretrained(aligned_path, torch_dtype=torch.bfloat16)
    pretrained_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)

    W_aligned = aligned_model.state_dict()
    W_base = pretrained_model.state_dict()

    dW_aligned = {name : W_aligned[name] - W_base[name] for name in W_aligned}

    new_state_dict = {}

    for name in tqdm(dW_aligned):
        if len(dW_aligned[name].size()) > 1:
            if k>0: 
                U, S, Vt = torch.linalg.svd(dW_aligned[name].float(), full_matrices = False)
                S[k:] = 0
                m = U @ torch.diag(S) @ Vt
            else: # k=0 extrapolates full rank
                m = dW_aligned[name]

            new_state_dict[name] = W_aligned[name] +  coef * m
            
        else:
            new_state_dict[name] =   W_aligned[name] 

    aligned_model.load_state_dict(new_state_dict)

    save_path = args.save_path
    aligned_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

if __name__ == "__main__":
    main()