from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import argparse
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="", help="Model to evaluate")
parser.add_argument("--save-path", type=str, default="", help="Path to save the model completions")
parser.add_argument("--chat", action="store_true")

def get_prompt(instruction):
    prompt = f"<s>Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: "
    return prompt, len(prompt)

def main():
    args = parser.parse_args()
    print(args)

    # Load Model and Data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, pad_token_id=tokenizer.unk_token_id, torch_dtype=torch.bfloat16).to(device)
    data = pd.read_json(path_or_buf="../data/gsm/test.jsonl", lines=True).to_dict(orient='records')
    model.eval()

    results = {"question": [], "true_answer": [], "pred_answer": [], "true_number": [], "pred_number": []}

    computed = 0
    n_correct = 0
    for example in tqdm(data):
        if args.chat:
            prompt = tokenizer.apply_chat_template([{'role':'user', 'content':example["question"]}], tokenize=False, add_generation_prompt=True)
            prompt_size = len(prompt)
        
        else:
            prompt, prompt_size = get_prompt(example["question"])

        results["question"].append(example["question"])  
        results["true_answer"].append(example["answer"]) 

        # Extract and save ground truth number 
        true = float(example["answer"].split("####")[-1].replace(",","")) # Format X,XXX to XXXX -> Get number
        results["true_number"].append(true)
                
        model_inputs = tokenizer(prompt, return_tensors='pt').to(device)
        greedy_output = model.generate(**model_inputs, max_new_tokens=400, do_sample=False)
        text_output = tokenizer.decode(greedy_output[0], skip_special_tokens=True)[prompt_size-3:]
        results["pred_answer"].append(text_output)

        try: # Try to extract the number:
            if args.chat:   
                pred = float(re.findall(r'-?\d{1,3}(?:,\d{3})*', text_output)[-1].replace(",",""))
            else:
                pred = float(text_output.split("####")[-1].replace(",","").replace("</s>","")) # Format X,XXX to XXXX -> Get number
            right_answer = (pred == true)
            results["pred_number"].append(pred)

            #print(pred, true)

        except: # Failed to extract number
            print('extraction failed')
            print(text_output)
            right_answer = False
            results["pred_number"].append(None)
        
        n_correct += int(right_answer)
        computed += 1
        if computed % 15 == 0:
            print(n_correct / computed)


    print("Final accuracy", n_correct / len(data))
    if args.save_path != "":
        pd.DataFrame(results).to_csv(args.save_path)

if __name__ == "__main__":
    main()