from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
tok.pad_token = tok.eos_token

ds = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train[:20000]") 

def count_tokens(ex):
    prompt = f"[INST] Question: {ex['question']}\nAnswer: {ex['answer']}\nSolution: {ex['solution']} [/INST]"
    return {"len": len(tok(prompt)["input_ids"])}

lens = ds.map(count_tokens)
print(np.percentile(lens["len"], [50, 75, 90, 95, 99, 100]))