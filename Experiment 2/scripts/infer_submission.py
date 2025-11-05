import os, pandas as pd, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
from utils import build_prompt, normalize_bool_text, TRUE_STR, FALSE_STR

MODEL_DIR = "./checkpoints/llama3_lora"
DATASET_ID = "ad6398/nyu-dl-teach-maths-comp"
SPLIT = "test"

OUT_CSV = "./submission.csv"

@torch.no_grad()
def predict(model, tok, prompt):
    inputs = tok(prompt, return_tensors="pt",max_length=768).to("cuda")
    out = model.generate(**inputs, max_new_tokens=4,do_sample=False,pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    if "[/INST]" in text:
        tail = text.split("[/INST]")[-1]
    else:
        tail = text
    return normalize_bool_text(tail)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto",quantization_config=bnb_config)
    
    model.eval()

    ds = load_dataset(DATASET_ID, split=SPLIT)
    preds = []
    for i, ex in enumerate(ds):
        p = build_prompt(ex["question"], ex["answer"], ex["solution"])
        preds.append(predict(model, tok, p))
        if (i + 1) % 50 == 0 or (i + 1) == len(ds):
            print(f"Completed {i + 1}/{len(ds)} inferences")

    df = pd.DataFrame({
        "id": list(range(len(preds))),
        "is_correct": [p == TRUE_STR for p in preds]
    })
    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
