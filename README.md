| **Experiment**   | **Model**                  | **Training Samples** | **LoRA Config (r, α)** | **Remarks**                                                           |
| ---------------- | -------------------------- | -------------------- | ---------------------- | --------------------------------------------------------------------- |
| **Experiment 1** | Meta-Llama-3.1-8B          | 5 000                | (16, 32)               | Baseline setup provided for initial evaluation. |
| **Experiment 2** | Meta-Llama-3.1-8B-Instruct | 20 000               | (16, 32)               | Extended training on larger dataset using instruct-tuned model.       |
| **Experiment 3** | Meta-Llama-3.1-8B          | 25 000               | (32, 64)               | Extended training on larger dataset using competition-approved model.  |


Note: Experiment 1 and 3 corresponds to the code used for the final Kaggle submission.


**Experiment 3 is our final submission with private score of 0.82203**



