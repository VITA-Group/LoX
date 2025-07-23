from transformers import AutoModelForCausalLM
import torch
import argparse
from tqdm import tqdm
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--aligned-model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--base-model", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--k", type=int, default=100)
args = parser.parse_args()
print(args)

def avg(list):
    return sum(list) / len(list)

def main():

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16).to(device)
    aligned_model = AutoModelForCausalLM.from_pretrained(args.aligned_model, torch_dtype=torch.bfloat16).to(device)

    W = base_model.state_dict()
    W_align = aligned_model.state_dict()
    
    r_align_total = []
    r_ft_total = []
    
    for name, module in tqdm(model.named_modules(), total=678):
        if isinstance(module, nn.Linear):
            k = args.k

            dW_align = (W_align[name+".weight"].to(device) - W[name+".weight"].to(device)).float() # dW_align
            dW_finetune = module.weight.data - W_align[name+".weight"].to(device).float()

            U,S,Vt = torch.linalg.svd(dW_align, full_matrices = False) # Get row space of dW_align
            
            p_column = U[:,:k] @ U[:,:k].T
            
            r_ft = torch.norm(p_column @ (dW_finetune + dW_align)) / torch.norm(dW_finetune + dW_align)
            r_align = torch.norm(p_column @ dW_align) / torch.norm(dW_align)
            
            r_ft_total.append(r_ft)
            r_align_total.append(r_align)
            
    print(f"R_ft: {avg(r_ft_total)}")
    print(f"R_align: {avg(r_align_total)}")

    return 
    


if __name__ == "__main__":
    main()