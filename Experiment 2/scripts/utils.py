# scripts/utils.py
from dataclasses import dataclass
from typing import Dict, List,Any
import torch

TRUE_STR, FALSE_STR = "True", "False"

def build_prompt(q: str, a: str, sol: str) -> str:
    return (
        "[INST]\n"
        f"Question: {q}\n"
        f"Student Answer: {a}\n"
        f"Reference Solution: {sol}\n\n"
        "Decide if the student's answer is correct.\n"
        "Reply with exactly 'True' or 'False'.\n"
        "[/INST]\n"
    )

def normalize_bool_text(s: str) -> str:
    s = s.strip().replace("`", "").replace(".", "").split()[0] if s else s
    s_low = s.lower()
    if s_low.startswith("true"):  return TRUE_STR
    if s_low.startswith("false"): return FALSE_STR
    return FALSE_STR

@dataclass
class LabelMaskCollator:
    tokenizer:Any
    max_length: int = 512

    def __call__(self, batch: List[Dict]):
        input_ids_list, labels_list, attn_masks = [], [], []
        for ex in batch:
            prompt = ex["prompt"]
            label = ex["label_text"] + self.tokenizer.eos_token
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            label_ids = self.tokenizer(label, add_special_tokens=False).input_ids
            input_ids = prompt_ids + label_ids
            labels = [-100]*len(prompt_ids) + label_ids
            if len(input_ids) > self.max_length:
                input_ids, labels = input_ids[-self.max_length:], labels[-self.max_length:]
            mask = [1]*len(input_ids)
            input_ids_list.append(torch.tensor(input_ids))
            labels_list.append(torch.tensor(labels))
            attn_masks.append(torch.tensor(mask))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attn_masks = torch.nn.utils.rnn.pad_sequence(attn_masks, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_masks}
