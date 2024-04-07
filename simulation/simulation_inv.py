#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
import json
import random

from scipy.stats import norm, binom

K = 1000
c = 5
Delta = 0.5
Final_T = 700
key = 23333
alpha = 0.05
N_trial = 5000
generate_data = False

name = f"results_data/K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-inv"
print("Used Delta is:", Delta)


def rng(key, m=2**31, a=1103515245, c=12345):
    return ((a*key + c) % m)/(m-1)


def uniform_Ps(Delta,K=K):
    generated = np.random.uniform(size=K)
    P = generated/generated[1:].sum() * Delta
    P[0] = 1-Delta
    assert P.max() == 1-Delta and np.abs(np.sum(P)- 1)<= 1e-4
    return P
    

def generate_uniform_local(inputs, c, key):
    assert len(inputs) >= c
    random.seed(tuple(inputs[-(c-1):]+[key]))
    xi = np.random.uniform(size=1)
    pi = np.random.permutation(K)
    return xi, pi


def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def find_next_token(xi, probs, pi):
    inv_pi = inv(pi)
    inv_probs = probs[inv_pi]
    i = 0
    s = 0
    while s <= xi:
        s += inv_probs[i]
        i += 1
    return inv_pi[i-1]


def generate_watermark_text(prompt, T=60, c=5, Delta=0.5, key=1):
    inputs = prompt.copy()
    selected_Us = []
    selected_Pis = []
    selected_difs = []
    highest_probs = []

    for _ in range(T):
        Probs = uniform_Ps(Delta)

        xi, pi = generate_uniform_local(inputs, c, key)
        next_token = find_next_token(xi, Probs, pi)

        eta = (pi[next_token]-1)/(K-1)

        selected_difs.append(-np.abs(xi-eta))

        selected_Us.append(xi)
        selected_Pis.append(pi[next_token])

        inputs.append(next_token)

    selected_Us = np.array(selected_Us).reshape(-1)
    selected_Pis = np.array(selected_Pis).reshape(-1)
    selected_difs = np.array(selected_difs).reshape(-1)

    return inputs, selected_difs, selected_Us, selected_Pis, highest_probs


check_points = np.arange(1,1+Final_T)
def compute_ind_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)



import scipy.integrate as integrate


if generate_data:
    prompts = []
    watermarked_text = []
    Us = []
    Pis = []
    Ds = []

    for trial in tqdm(range(N_trial)):
        prompt = np.random.randint(K, size=c).tolist()
        prompts.append(prompt)

        if Delta is not None:
            Delta_ins = np.random.uniform(0.001, Delta)
        else:
            Delta_ins = None

        watermark, selected_difs, selected_Us, selected_Pis, highest_probs = generate_watermark_text(prompt, T=Final_T, c=c, key=key, Delta=Delta_ins)
        watermarked_text.append(watermark)

        Us.append(selected_Us)
        Pis.append(selected_Pis)
        Ds.append(selected_difs)

    save_dict = dict()
    save_dict["p"] = np.array(prompts).tolist()
    save_dict["w"] = np.array(watermarked_text).tolist()
    save_dict["d"] = np.array(Ds).tolist()
    save_dict["u"] = np.array(Us).tolist()
    save_dict["pi"] = np.array(Pis).tolist()
    save_dict["h"] = np.array(highest_probs).tolist()
    json.dump(save_dict, open(name+".json", 'w'))

else:
    save_dict = json.load(open(name+".json", "r"))

    prompts = save_dict["p"] 
    watermarked_text = save_dict["w"]

    Ys = save_dict["y"]
    Pis = np.array(save_dict["pi"])
    Us = np.array(save_dict["u"])
    Ds = np.array(save_dict["d"])
    highest_probs = save_dict["h"] 

etas = np.array(Pis)/(K-1)
Us = np.array(Us)

    
def h_opt_dif(Us, etas,delta=0.1):
    final_Y = -np.abs(Us-etas)
    final_Y = np.log(np.maximum(1+final_Y/(1-delta),1e-4)/np.maximum(1+final_Y,0))
    cumsum_Ys = np.cumsum(final_Y, axis=1)

    Null_Ys_U = np.random.uniform(size=(10000, Us.shape[1]))
    Null_Ys_pi_s = np.random.randint(low=0, high=K, size=(10000, Us.shape[1]))
    Null_etas = np.array(Null_Ys_pi_s)/(K-1)
    null_final_Y = -np.abs(Null_Ys_U-Null_etas)
    null_final_Y = np.log(np.maximum(1+null_final_Y/(1-delta),1e-4)/np.maximum(1+null_final_Y,0))
    null_cumsum_Ys = np.cumsum(null_final_Y, axis=1)
    h_ind_qs = np.quantile(null_cumsum_Ys, 1-alpha, axis=0)

    results = (cumsum_Ys >= h_ind_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_id_dif(Ds):
    mu_dif = -1/3
    var_dif = 1/6 - 1/9
    h_id_dif_qs = compute_ind_q(1-alpha, mu_dif, var_dif, check_points)

    Ds = np.array(Ds)
    cumsum_Ds = np.cumsum(Ds, axis=1)

    results = (cumsum_Ds >= h_id_dif_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_ind_dif(Ds,delta=-0.05):
    assert -1 <= delta <= 0
    h_id_dif_qs = binom.ppf(n=check_points, p = 1-(1+delta)**2, q = 1-alpha)

    Ds = np.array(Ds)
    h_ind_Ds = Ds >= delta
    cumsum_Ds = np.cumsum(h_ind_Ds, axis=1)

    results = (cumsum_Ds >= h_id_dif_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


##############################################
##
## Check the Type II errors
##
##############################################
result_dict = dict()

result_dict["dif-opt-02"] = h_opt_dif(Us, etas, delta=0.2)[0].tolist()
result_dict["dif-opt-01"] = h_opt_dif(Us, etas, delta=0.1)[0].tolist()
result_dict["dif-opt-005"] = h_opt_dif(Us, etas, delta=0.05)[0].tolist()
result_dict["dif-opt-001"] = h_opt_dif(Us, etas, delta=0.01)[0].tolist()
result_dict["dif-opt-0005"] = h_opt_dif(Us, etas, delta=0.005)[0].tolist()
result_dict["dif-opt-0001"] = h_opt_dif(Us, etas, delta=0.001)[0].tolist()

result_dict["dif"] = h_id_dif(Ds)[0].tolist()
result_dict["dif-ind-01"] = h_ind_dif(Ds, -0.1)[0].tolist()
result_dict["dif-ind-02"] = h_ind_dif(Ds, -0.2)[0].tolist()
result_dict["dif-ind-05"] = h_ind_dif(Ds, -0.5)[0].tolist()

# pprint(result_dict["dif"])
json.dump(result_dict, open(name+"-result"+".json", 'w'))


##############################################
##
## Check the Type I errors
##
###############################################
Null_Ys_U = np.random.uniform(size=(N_trial, Final_T))
Null_Ys_pi_s = np.random.randint(low=0, high=K, size=(N_trial, Final_T))
Null_etas = np.array(Null_Ys_pi_s)/(K-1)
Null_Ds = -np.abs(Null_Ys_U-Null_etas)

result_dict = dict()
result_dict["dif"] = h_id_dif(Null_Ds)[0].tolist()

result_dict["dif-ind-01"] = h_ind_dif(Null_Ds, -0.1)[0].tolist()
result_dict["dif-ind-02"] = h_ind_dif(Null_Ds, -0.2)[0].tolist()
result_dict["dif-ind-05"] = h_ind_dif(Null_Ds, -0.5)[0].tolist()

result_dict["dif-opt-02"] = h_opt_dif(Null_Ys_U, Null_etas, delta=0.2)[0].tolist()
result_dict["dif-opt-01"] = h_opt_dif(Null_Ys_U, Null_etas, delta=0.1)[0].tolist()
result_dict["dif-opt-005"] = h_opt_dif(Null_Ys_U, Null_etas, delta=0.05)[0].tolist()
result_dict["dif-opt-001"] = h_opt_dif(Null_Ys_U, Null_etas, delta=0.01)[0].tolist()
result_dict["dif-opt-0001"] = h_opt_dif(Null_Ys_U, Null_etas, delta=0.001)[0].tolist()
result_dict["dif-opt-0005"] = h_opt_dif(Null_Ys_U, Null_etas, delta=0.005)[0].tolist()

json.dump(result_dict, open(name+"-null.json", 'w'))
