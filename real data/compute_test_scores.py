1#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import numpy as np
from tqdm import tqdm
import pickle
import json
from scipy.stats import gamma, norm, binom
import argparse

## Make sure the following configuration is the same used in generating watermarked samples
parser = argparse.ArgumentParser(description="Experiment Settings")

parser.add_argument('--method',default="gumbel",type=str)
# parser.add_argument('--method',default="transform",type=str)

parser.add_argument('--model',default="facebook/opt-1.3b",type=str)
# parser.add_argument('--model',default="princeton-nlp/Sheared-LLaMA-2.7B",type=str)
# parser.add_argument('--model',default="huggyllama/llama-7b",type=str)

parser.add_argument('--seed_way',default="skipgram_prf",type=str)
parser.add_argument('--seed',default=15485863,type=int)
parser.add_argument('--c',default=4,type=int)
parser.add_argument('--temp',default=0.1,type=float)
parser.add_argument('--alpha',default=0.05,type=float)

parser.add_argument('--m',default=200,type=int) 
parser.add_argument('--T',default=500,type=int) 

args = parser.parse_args()
print(args)



if args.model == "facebook/opt-1.3b":
    model_name = "1p3B"
    vocab_size = 50272
elif args.model == "huggyllama/llama-7b":
    model_name = "7B"
    vocab_size = 32000
elif args.model == "princeton-nlp/Sheared-LLaMA-2.7B":
    model_name = "2p7B"
    vocab_size = 32000
else: 
    raise ValueError(f"No such a model: {args.model}.")


if args.method == "transform":
    from sampling import transform_key_func, transform_Y

    generator = torch.Generator()
    A_inverse = lambda inputs : transform_key_func(generator,inputs, vocab_size, args.seed, args.c, args.seed_way)
    from score_functions import h_opt_dif, h_id_dif, h_ind_dif

elif args.method == "gumbel":

    from sampling import gumbel_key_func, gumbel_Y
    generator = torch.Generator()
    A_gumbel = lambda inputs : gumbel_key_func(generator,inputs, vocab_size, args.seed, args.c, args.seed_way)
    from score_functions import h_ars, h_log, h_ind, h_opt_gum

else:
    raise ValueError(f"No such a watermark: {args.method}. Only from 'transform' and 'gumbel'.")

def compute_ind_q(q, mu, var, T):
    check_point = np.arange(1,1+T)
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)


exp_name = f"results_data/{model_name}-{args.method}-c{args.c}-m{args.m}-T{args.T}-{args.seed_way}-{args.seed}-temp{args.temp}"
results = pickle.load(open(exp_name + ".pkl", "rb"))
check_points = np.arange(1, 1+args.m)
prompts= results['prompts']
watermarked_text = results["watermark"]["tokens"].numpy()
highest_probs = results["watermark"]["top_probs"].numpy()

exp_null_name = f"results_data/{model_name}-raw-m{args.m}-T{args.T}.pkl"
null_results = pickle.load(open(exp_null_name, "rb"))
null_data = null_results['null']['tokens']
c = args.c

from IPython import embed

if args.method == "transform":
    ##############################################
    ##
    ## Check the Type II errors
    ##
    ###############################################

    # here Dif_Ys = - |U_t - eta_t|
    Us = results["watermark"]["Us"].numpy()
    etas = results["watermark"]["etas"].numpy()
    Dif_Ys = -np.abs(Us-etas)

    result_dict = dict()
    result_dict["top_probs"] = highest_probs.tolist()

    result_dict["dif"] = h_id_dif(Dif_Ys)[0].tolist()

    result_dict["dif-ind-01"] = h_ind_dif(Dif_Ys, -0.1, alpha=args.alpha)[0].tolist()
    result_dict["dif-ind-02"] = h_ind_dif(Dif_Ys, -0.2, alpha=args.alpha)[0].tolist()
    result_dict["dif-ind-05"] = h_ind_dif(Dif_Ys, -0.5, alpha=args.alpha)[0].tolist()

    result_dict["dif-opt-03"] = h_opt_dif(Dif_Ys, delta=0.3, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-02"] = h_opt_dif(Dif_Ys, delta=0.2, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-01"] = h_opt_dif(Dif_Ys, delta=0.1, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-005"] = h_opt_dif(Dif_Ys, delta=0.05, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-001"] = h_opt_dif(Dif_Ys, delta=0.01, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-0005"] = h_opt_dif(Dif_Ys, delta=0.005, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-0001"] = h_opt_dif(Dif_Ys, delta=0.001, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()

    json.dump(result_dict, open(exp_name+"-result.json", 'w'))

    ##############################################
    ##
    ## Check the Type I errors
    ##
    ###############################################

    computed_Ys = []
    for i in tqdm(range(args.T)):
        text = null_data[i]
        prompt = prompts[i]
        full_texts =  torch.cat([prompt[-c:],text])

        this_Ys = []
        for j in range(args.m):
            xi, pi = A_inverse(full_texts[:c+j].unsqueeze(0))
            Y = transform_Y(full_texts[c+j].unsqueeze(0).unsqueeze(0), pi, xi)[0]
            this_Ys.append(Y.unsqueeze(0))

        this_Ys = torch.vstack(this_Ys)
        computed_Ys.append(this_Ys.squeeze())
    computed_Ys = torch.vstack(computed_Ys).numpy()

    result_dict = dict()
    result_dict["dif"] = h_id_dif(computed_Ys, alpha=args.alpha)[0].tolist()

    result_dict["dif-ind-01"] = h_ind_dif(computed_Ys, -0.1, alpha=args.alpha)[0].tolist()
    result_dict["dif-ind-02"] = h_ind_dif(computed_Ys, -0.2, alpha=args.alpha)[0].tolist()
    result_dict["dif-ind-05"] = h_ind_dif(computed_Ys, -0.5, alpha=args.alpha)[0].tolist()

    result_dict["dif-opt-02"] = h_opt_dif(computed_Ys, delta=0.2, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-01"] = h_opt_dif(computed_Ys, delta=0.1, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-005"] = h_opt_dif(computed_Ys, delta=0.05, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-001"] = h_opt_dif(computed_Ys, delta=0.01, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-0005"] = h_opt_dif(computed_Ys, delta=0.005, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    result_dict["dif-opt-0001"] = h_opt_dif(computed_Ys, delta=0.001, alpha=args.alpha, vocab_size=vocab_size, model_name=model_name)[0].tolist()
    
    json.dump(result_dict, open(exp_name+"-null.json", 'w'))

elif args.method == "gumbel":
    ##############################################
    ##
    ## Check the Type II errors
    ##
    ##############################################

    # Y_t = U_{t, w_t}
    Ys = results["watermark"]["Ys"].numpy()

    result_dict = dict()
    result_dict["top_probs"] = highest_probs.tolist()

    result_dict["ars"] = h_ars(Ys, alpha=args.alpha)[0].tolist()
    result_dict["log"] = h_log(Ys, alpha=args.alpha)[0].tolist()

    result_dict["ind-08"] = h_ind(Ys, 0.8, alpha=args.alpha)[0].tolist()
    result_dict["ind-09"] = h_ind(Ys, 0.9, alpha=args.alpha)[0].tolist()
    result_dict["ind-1/e"] = h_ind(Ys, 1/np.exp(1), alpha=args.alpha)[0].tolist()


    result_dict["opt-0001"] = h_opt_gum(Ys, 0.001, alpha=args.alpha)[0].tolist()
    result_dict["opt-0005"] = h_opt_gum(Ys, 0.005, alpha=args.alpha)[0].tolist()
    result_dict["opt-005"] = h_opt_gum(Ys, 0.05, alpha=args.alpha)[0].tolist()
    result_dict["opt-001"] = h_opt_gum(Ys, 0.01, alpha=args.alpha)[0].tolist()
    result_dict["opt-02"] = h_opt_gum(Ys, 0.2, alpha=args.alpha)[0].tolist()
    result_dict["opt-015"] = h_opt_gum(Ys, 0.15, alpha=args.alpha)[0].tolist()
    result_dict["opt-01"] = h_opt_gum(Ys, 0.1, alpha=args.alpha)[0].tolist()

    json.dump(result_dict, open(exp_name+"-result"+".json", 'w'))

    ##############################################
    ##
    ## Check the Type I errors
    ##
    ##############################################

    computed_Ys = []
    for i in tqdm(range(args.T)):
        text = null_data[i]
        prompt = prompts[i]
        full_texts =  torch.cat([prompt[-c:],text])

        this_Ys = []
        for j in range(args.m):
            xi, pi = A_gumbel(full_texts[:c+j].unsqueeze(0))
            Y = gumbel_Y(full_texts[c+j].unsqueeze(0).unsqueeze(0), pi, xi)
            this_Ys.append(Y.unsqueeze(0))
        this_Ys = torch.vstack(this_Ys)
        computed_Ys.append(this_Ys.squeeze())
    computed_Ys = torch.vstack(computed_Ys).numpy()

    result_dict = dict()
    result_dict["ars"] = h_ars(computed_Ys, alpha=args.alpha)[0].tolist()
    result_dict["log"] = h_log(computed_Ys, alpha=args.alpha)[0].tolist()

    result_dict["ind-08"] = h_ind(computed_Ys, 0.8, alpha=args.alpha)[0].tolist()
    result_dict["ind-09"] = h_ind(computed_Ys, 0.9, alpha=args.alpha)[0].tolist()
    result_dict["ind-1/e"] = h_ind(computed_Ys, 1/np.exp(1), alpha=args.alpha)[0].tolist()

    result_dict["opt-02"] = h_opt_gum(computed_Ys, 0.2, alpha=args.alpha)[0].tolist()
    result_dict["opt-015"] = h_opt_gum(computed_Ys, 0.15, alpha=args.alpha)[0].tolist()
    result_dict["opt-01"] = h_opt_gum(computed_Ys, 0.1, alpha=args.alpha)[0].tolist()
    result_dict["opt-005"] = h_opt_gum(computed_Ys, 0.05, alpha=args.alpha)[0].tolist()
    result_dict["opt-001"] = h_opt_gum(computed_Ys, 0.01, alpha=args.alpha)[0].tolist()
    result_dict["opt-0005"] = h_opt_gum(computed_Ys, 0.005, alpha=args.alpha)[0].tolist()
    result_dict["opt-0001"] = h_opt_gum(computed_Ys, 0.001, alpha=args.alpha)[0].tolist()

    json.dump(result_dict, open(exp_name+"-null.json", 'w'))

else:
    raise ValueError

