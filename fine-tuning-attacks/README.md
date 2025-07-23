## Running Attacks and Evaluations

### Running an Attack

To run one of the attacks, use the following command:

```
python sft_<attack>.py --save-path <output_path> --base-model <base_model_name>
```

Replace `<attack>` with the name of the desired attack (e.g., `gsm`, `id`, etc.).  
You can modify hyperparameters by passing the appropriate flags. 

The `pure_bad` attack was performed using the code from the following notebook:

[LLM-Tuning-Safety: Tier 1 Harmful Examples Demonstration](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/tier1-harmful-examples-demonstration.ipynb)

---

### GSM8K Accuracy Evaluation

To evaluate accuracy on the GSM8K dataset, run:

```
python acc_gsm.py --model <model_path_or_name>
```

---

### Helpfulness Evaluation

Helpfulness evaluation was done using the code provided at:

[URIAL Evaluation Framework](https://github.com/Re-Align/URIAL/tree/main/evaluate)
