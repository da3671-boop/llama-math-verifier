import os
import  torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from utils import build_prompt, LabelMaskCollator, TRUE_STR, FALSE_STR

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "./checkpoints/llama3_lora"
MAX_LEN = 768
EPOCHS = 1
LR = 2e-4
BATCH = 2
GRAD_ACC = 8
 
def main():
    dataset_id = "ad6398/nyu-dl-teach-maths-comp"  
    ds = load_dataset(dataset_id)
    split_ds = ds["train"].train_test_split(test_size=0.005, seed=42)
  
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]
    train_ds = train_ds.shuffle(seed=42).select(range(20000))
    print("Training subset size:", len(train_ds))
    print("Validation size:", len(val_ds))

    def to_sft(ex):
        prompt = build_prompt(ex["question"], ex["answer"], ex["solution"])
        label_text = TRUE_STR if bool(ex["is_correct"]) else FALSE_STR
        return {"prompt": prompt, "label_text": label_text}

    train_ds = train_ds.map(to_sft, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(to_sft, remove_columns=val_ds.column_names)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"[Rank {local_rank}] using GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)

    collator = LabelMaskCollator(tokenizer=tok, max_length=MAX_LEN)

    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        bf16=True,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok
    )

    trainer.train()
    trainer.save_model(OUT_DIR)

if __name__ == "__main__":
    main()
