
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import argparse
from scipy.stats import gamma
import matplotlib.pyplot as plt
from datasets import load_dataset
from IPython import embed

# -------------------------------
# Hash function and PRF function
# -------------------------------
def additive_prf(input_ids, salt_key):
    return salt_key * input_ids.sum().item()

def seed_rng(generator, tokens, salt_key=15485863):
    prf_key = additive_prf(tokens[0][-1:], salt_key)
    generator.manual_seed(prf_key)

# -------------------------------
# Watermarking: Key Function
# -------------------------------
def gumbel_key_func(generator, inputs, vocab_size, key):
    xis, pis = [], []
    for k in range(inputs.shape[0]):
        seed_rng(generator, inputs[k].unsqueeze(0), salt_key=key)
        xi = torch.rand(size=(1, vocab_size), generator=generator)
        pi = torch.arange(vocab_size)
        xis.append(xi)
        pis.append(pi)
    return torch.vstack(xis), torch.vstack(pis)

# -------------------------------
# Watermarking: Sampling Function
# -------------------------------
def gumbel_sampling(probs, pi, xi):
    return torch.argmax(xi ** (1 / torch.gather(probs, 1, pi)), dim=1).unsqueeze(-1)

def gumbel_Y(s, pi, xi):
    y = torch.gather(xi, -1, s.cpu())  # shape: [batch_size, 1]
    return y.view(-1, 1)  # ensure 2D shape: [batch_size, 1]

# -------------------------------
# Detection Functions
# -------------------------------
def h_log(Ys, alpha=0.05):
    check_points = np.arange(1, 1 + Ys.shape[-1])
    h_log_qs = gamma.ppf(alpha, a=check_points)
    Ys = np.array(Ys)
    cumsum_Ys = np.cumsum(np.log(Ys), axis=1)
    return (cumsum_Ys >= -h_log_qs).mean(axis=0)

def h_ars(Ys, alpha=0.05):
    check_points = np.arange(1, 1 + Ys.shape[-1])
    h_ars_qs = gamma.ppf(1 - alpha, a=check_points)
    Ys = np.array(Ys)
    cumsum_Ys = np.cumsum(-np.log(1 - Ys), axis=1)
    return (cumsum_Ys >= h_ars_qs).mean(axis=0)

def f_opt(r, delta):
    inte = np.floor(1 / (1 - delta))
    rest = 1 - (1 - delta) * inte
    return np.log(inte * r ** (delta / (1 - delta)) + r ** (1 / rest - 1))

def h_opt_gum(Ys, delta0=0.1, alpha=0.01):
    Ys = np.array(Ys)
    h_ars_Ys = f_opt(Ys, delta0)

    def find_q(N=2500):
        # Use simulation to compute critical values
        Null_Ys = np.random.uniform(size=(N, Ys.shape[1]))
        Simu_Y = f_opt(Null_Ys, delta0)
        Simu_Y = np.cumsum(Simu_Y, axis=1)
        return np.quantile(Simu_Y, 1 - alpha, axis=0)

    q_lst = [find_q(2500) for _ in range(10)]
    h_help_qs = np.mean(np.array(q_lst), axis=0)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    return (cumsum_Ys >= h_help_qs).mean(axis=0)

def TrGoF(Ys, alpha=0.01, mask=True, eps=1e-10):
    """
    Truncated Goodness-of-Fit (TrGoF) test.

    Reference: 
    arXiv: https://arxiv.org/abs/2411.13868  
    Code:  https://github.com/lx10077/TrGoF
    """

    def compute_score(Ys_slice):
        # Transform to p-values
        ps = 1 - Ys_slice
        ps = np.sort(ps, axis=-1)
        m = ps.shape[-1]
        rk = np.arange(1, 1 + m) / m
        final = (rk - ps) ** 2 / (ps * (1 - ps) + eps) / 2
        if mask:
            valid = (ps >= 1e-3) * (rk >= ps)
            final *= valid
        return np.log(m * np.max(final, axis=-1) + 1e-10)
    
    def compute_quantile(m):
        qs = []
        for _ in range(10):
            raw_data = np.random.uniform(size=(10000, m))
            H0_scores = compute_score(raw_data)
            q = np.quantile(H0_scores, 1 - alpha)
            qs.append(q)
        return np.mean(qs)
    
    detection_curve = []
    for t in range(1, Ys.shape[1] + 1, 3):
        Ys_trunc = Ys[:, :t]  # follow your convention: drop first token
        scores = compute_score(Ys_trunc)
        q = compute_quantile(t)
        detection_curve.append(np.mean(scores >= q))

    return np.array(detection_curve)

def batched_gumbel_generation(prompts, model, tokenizer, generator, m=100, temp=1.0, key=15485863, batch_size=10):
    """
    Generate watermarked text for a list of prompts using batched Gumbel sampling.

    Args:
        prompts: list of prompt strings
        model: HuggingFace causal LM
        tokenizer: HuggingFace tokenizer
        generator: torch.Generator()
        m: number of tokens to generate
        temp: sampling temperature
        key: seed key for deterministic Gumbel noise
        batch_size: number of prompts to process in a single batch

    Returns:
        Ys_all: numpy array of shape [len(prompts), m] with pivotal statistics
    """
    model.eval()
    device = next(model.parameters()).device
    vocab_size = model.get_output_embeddings().weight.shape[0]

    Ys_all = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize and move to device
        tokenized = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized["input_ids"].to(device)

        Ys = []

        for _ in range(m):
            with torch.no_grad():
                output = model(input_ids)
            probs = F.softmax(output.logits[:, -1] / temp, dim=-1).cpu()

            xi, pi = gumbel_key_func(generator, input_ids, vocab_size, key=key)
            next_token = gumbel_sampling(probs, pi, xi)
            Y = gumbel_Y(next_token, pi, xi)
            Ys.append(Y)

            input_ids = torch.cat([input_ids, next_token.to(device)], dim=1)

        Ys_batch = torch.cat(Ys, dim=1).squeeze(-1).numpy()
        Ys_all.append(Ys_batch)

    return np.vstack(Ys_all)  # [num_prompts, m]


# -------------------------------
# Main Execution
# -------------------------------
def main():
    # You can change the following parameters to control generation and detection behavior:

    # --temp: Sampling temperature used during generation.
    #         A higher temperature (e.g., 1.0 or above) produces more random output,
    #         while a lower temperature (e.g., 0.7) makes the generation more deterministic.

    # --alpha: Significance level for watermark detection tests.
    #          A smaller alpha (e.g., 0.01) means fewer false positives (Type I errors), 
    #          but may result in higher Type II error (missed detections).

    # --model: HuggingFace model name or path.
    #          You can choose from pre-trained causal language models such as:
    #              - "sshleifer/tiny-gpt2"          (very small, for testing/debugging)
    #              - "gpt2"                         (standard small GPT-2)
    #              - "facebook/opt-1.3b"            (larger OPT model, good quality)
    #              - "EleutherAI/gpt-neo-1.3B"      (another large open model)
    #          The larger the model, the more coherent the generated text, 
    #          but the slower the generation and heavier the memory requirement.

    # Example usage:
    #     python script.py --temp 0.8 --alpha 0.01 --model "gpt2"

    parser = argparse.ArgumentParser()
    parser.add_argument('--temp', type=float, default=1, help="Sampling temperature for generation")
    parser.add_argument('--alpha', type=float, default=0.01, help="Significance level for detection tests")
    parser.add_argument('--model', type=str, default="facebook/opt-1.3b", help="Which model you want to embed watermarks?")
    # Use only known args, ignore unknown ones like -f from IPython
    args, _ = parser.parse_known_args()

    m = 100  # Number of tokens to generate or text length
    model_name = args.model.split("/")[-1]
    ys_path = f"{model_name}_temp{args.temp}_Ys_all.npy"
    if os.path.exists(ys_path):
        # Load precomputed Ys_all if it exsits
        Ys_all = np.load(ys_path)
        print(f"Loaded Ys_all from {ys_path}")
    else:
        # Otherwise, we generate those pivotal statistics
        tokenizer = AutoTokenizer.from_pretrained(args.model) # Load tokenizer which convers text to a sequence of tokens
        # Ensure the tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",             # Automatically place model layers on available GPU(s)
            torch_dtype=torch.float16      # (Optional) Set tensor data type to float16 for faster computation
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if possible
        print(f"Using device: {device}")
        model = model.to(device) # Move the model to GPU, otherwise it is default on CPU
        model.eval()
        
        # Load the first 200 samples from the AG News dataset.
        # You could also test your own questions or queries. The model will continue to write after your given text.
        # To that end, simply change the following 'raw_texts' with a list of your questions.
        dataset = load_dataset("ag_news", split="train[:200]")
        raw_texts = [x["text"] for x in dataset]
        filtered_texts = [t.strip() for t in raw_texts if len(t.strip()) >= 100] # We consider question/prompts with sufficient length (>=100)
        prompts = []
        for text in filtered_texts:
            tokens = tokenizer.encode(text, truncation=False)
            if len(tokens) >= 100:
                truncated = tokens[-50:]  # Take the last 50 tokens
                decoded = tokenizer.decode(truncated, skip_special_tokens=True)
                prompts.append(decoded)
        if len(prompts) == 0:
            raise ValueError("No prompts left after filtering. Check length filter or dataset size.")
        else:
            print(f"There are {len(prompts)} prompts")

        # Compute Ys_all from scratch
        Ys_all = batched_gumbel_generation(
            prompts, model, tokenizer, torch.Generator(), m=m, temp=args.temp, batch_size=10
        )
        if not isinstance(Ys_all, np.ndarray):
            Ys_all = np.array(Ys_all)
        np.save(ys_path, Ys_all)
        print(f"Saved Ys_all to {ys_path}")
    print(Ys_all.shape)
    
    # Step 2: Perform detections and collect powers. You could test you detection rules here!
    power_ars = h_ars(Ys_all, alpha=args.alpha)   
    power_log = h_log(Ys_all, alpha=args.alpha)    
    power_opt = h_opt_gum(Ys_all, alpha=args.alpha)  
    power_hc  = TrGoF(Ys_all, alpha=args.alpha)    
        
    # Step 3:  Plot the averaged power v.s. text length
    x = np.arange(1, m + 1)
    x3 = np.arange(1, m + 1, 3)
    plt.figure(figsize=(10, 6))
    plt.plot(x, power_ars, label="h_ars")
    plt.plot(x, power_log, label="h_log")
    plt.plot(x, power_opt, label="h_opt")
    plt.plot(x3, power_hc, label="TrGoF")
    plt.xlabel("Text length")
    plt.ylabel("Average power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{model_name}_temp{args.temp}_alpha{args.alpha}_power.png", dpi=200)
    print("Saved plot to avg_detection_scores.png")

main()