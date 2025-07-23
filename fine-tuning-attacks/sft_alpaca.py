from datasets import load_dataset, concatenate_datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
import argparse
from math import ceil

device = "cuda" if torch.cuda.is_available() else "cpu"
parser = argparse.ArgumentParser()

parser.add_argument("--base-model", type=str, default="")
parser.add_argument("--epochs", type=int, default=1) #
parser.add_argument("--batch-size", type=int, default=8) #
parser.add_argument("--acc-steps", type=int, default=8) #
parser.add_argument("--lr", type=float, default=2e-5) #
parser.add_argument("--max-grad-norm", type=int, default=2)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--save-path", type=str, default="output")
parser.add_argument("--scheduler", type=str, default="linear")
parser.add_argument("--max-seq-length", type=str, default=1024)
parser.add_argument("--safeinst", action="store_true")

PROMPT_DICT = {
    "prompt_input": (lambda x: 
        '<s>' + "Below is an instruction that describes a task, paired with an input that provides further context. " +
        "Write a response that appropriately completes the request.\n" +
        f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}</s>"
    ),
    "prompt_no_input": (lambda x:
        '<s>' + "Below is an instruction that describes a task. " +
        "Write a response that appropriately completes the request.\n" +
        f"### Instruction:\n{x['instruction']}\n\n### Response:\n{x['output']}</s>"
    ),
}

def training_prompt(example):
    #output, input instruction
    if example["input"] == "":
        return {'text' : PROMPT_DICT['prompt_no_input'](example)}
    else:
        return {'text' : PROMPT_DICT['prompt_input'](example)}
    

def main():
    args = parser.parse_args()
    train_dataset = load_dataset("json", data_files="../data/alpaca_data_no_safety.json")["train"].map(training_prompt)
    if args.safeinst:
        safe_data = load_dataset("json", data_files="../data/safety_only_data_Instructions.json")["train"].map(training_prompt)
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

    model.generation_config.do_sample=True

    model.train()

    print('Trainable params', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
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
        save_steps=15000,                         
        save_total_limit=2,                     
        bf16=True,                              
        push_to_hub=False, 
        dataset_text_field="text",
        max_seq_length=args.max_seq_length                     
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset, #["train"],
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()
