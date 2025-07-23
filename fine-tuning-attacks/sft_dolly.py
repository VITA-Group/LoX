from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import  SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
import argparse
from math import ceil

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()

parser.add_argument("--base-model", type=str, default="")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--acc-steps", type=int, default=4)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max-grad-norm", type=int, default=2)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--save-path", type=str, default="output")
parser.add_argument("--scheduler", type=str, default="constant") # was linear
parser.add_argument("--max-seq-length", type=str, default=1024)
parser.add_argument("--safeinst", action="store_true")

def training_prompt_safeinst(example):
    return {'text': f"<s>### Instruction:\n{example["instruction"]}\n\n### Response: {example["output"]}</s>"}

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"<s>###Instruction: {example['instruction'][i]} "
        if example['context'][i] != "":
            text += f"\n###Context: {example['instruction'][i]} "
        text += f"\n###Response: {example['response'][i]}</s>"
        output_texts.append(text)
    return output_texts

def main():
    args = parser.parse_args()
    train_dataset = load_dataset("json", data_files="../data/databricks-dolly-15k-no-safety.jsonl")['train']#.shuffle(seed=42).select(range(10000))
    if args.safeinst:
        safe_data = load_dataset("json", data_files="../data/safety_only_data_Instructions.json")["train"].map(training_prompt_safeinst)
        train_dataset = concatenate_datasets([train_dataset, safe_data.select(range(ceil(0.025*len(train_dataset))))])
        print(f"loading {ceil(0.025*len(train_dataset))} examples from safeinst")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast = False)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right' # to prevent errors with FA
    tokenizer.truncation_side = 'left' # to prevent cutting off last generation

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16
    )
    
    response_template = "\n###Response:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    model.generation_config.do_sample=True

    model.train()
    
    training_args = SFTConfig(
        output_dir=args.save_path,               
        num_train_epochs=args.epochs,                     
        per_device_train_batch_size=args.batch_size,         
        gradient_accumulation_steps=args.acc_steps,          
        gradient_checkpointing=True,            
        learning_rate=args.lr,                     
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,                      
        lr_scheduler_type=args.scheduler,             
        logging_steps=10,                       
        #save_steps=5000,                         
        save_total_limit=8,                     
        bf16=True,                              
        push_to_hub=False,
        save_strategy="epoch", 
        max_seq_length=args.max_seq_length,                      
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        #tokenizer=tokenizer,
        #dataset_text_field="text",
        formatting_func=formatting_prompts_func,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
