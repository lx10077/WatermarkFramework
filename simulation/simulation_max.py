#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import json
import random
from scipy.stats import gamma, norm, binom
from scipy.integrate import quad

K = 1000
c = 5
Delta = 0.7
Final_T = 700
key = 23333
alpha = 0.05
N_trial = 5000
generate_data = False

name = f"results_data/K{K}N{N_trial}c{c}key{key}T{Final_T}Delta{Delta}-alpha{alpha}-max"
print("Used Delta is:", Delta)


def uniform_Ps(Delta,K=K):
    generated = np.random.uniform(size=K)
    P = generated/generated[1:].sum() * Delta
    P[0] = 1-Delta
    assert P.max() == 1-Delta and np.abs(np.sum(P)- 1)<= 1e-4
    return P
    

def generate_uniform_local(inputs, c, key):
    assert len(inputs) >= c
    random.seed(tuple(inputs[-(c-1):]+[key]))
    generated = np.random.uniform(size=K)
    return generated


def generate_watermark_text(prompt, T=60, c=5, Delta=0.5, key=1):
    inputs = prompt.copy()
    selected_xis = []
    highest_probs = []

    for _ in range(T):
        Probs = uniform_Ps(Delta)
        uniform_xi = generate_uniform_local(inputs, c, key)
        next_token = np.argmax(uniform_xi ** (1/Probs))

        inputs.append(next_token)
        selected_xis.append(uniform_xi[next_token])
        highest_probs.append(np.max(Probs))

    return inputs, selected_xis, highest_probs


## CDF and PDF
def F(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += Probs[k]*x**(1/Probs[k])
    return rho

def f(x, Probs):
    rho = np.zeros_like(x)
    for k in range(len(Probs)):
        rho += x**(1/Probs[k]-1)
    return rho


## Compute critial values
check_points = np.arange(1,1+Final_T)


def compute_gamma_q(q, check_point):
    qs = []
    for t in check_point:
        qs.append(gamma.ppf(q=q,a=t))
    return np.array(qs)

def compute_ind_q(q, mu, var, check_point):
    qs = []
    q = norm.ppf(q)
    for t in check_point:
        qs.append(t*mu+ q*np.sqrt(t*var))
    return np.array(qs)

h_ars_qs = compute_gamma_q(1-alpha, check_points)
h_log_qs = compute_gamma_q(alpha, check_points)


if generate_data:
    prompts = []
    watermarked_text = []
    highest_probs_lst = []
    Ys = []

    for trial in tqdm(range(N_trial)):
        prompt = np.random.randint(K, size=c).tolist()
        prompts.append(prompt)
        
        Delta_ins = np.random.uniform(0.001, Delta)

        watermark, generated_Ys, highest_probs = generate_watermark_text(prompt, T=Final_T, c=c, key=key, Delta=Delta_ins)
        watermarked_text.append(watermark)
        Ys.append(generated_Ys)
        highest_probs_lst.append(highest_probs)

    save_dict = dict()
    save_dict["p"] = np.array(prompts).tolist()
    save_dict["w"] = np.array(watermarked_text).tolist()
    save_dict["y"] = np.array(Ys).tolist()
    save_dict["h"] = np.array(highest_probs_lst).tolist()

    json.dump(save_dict, open(name+".json", 'w'))

else:
    save_dict = json.load(open(name+".json", "r"))

    prompts = save_dict["p"] 
    watermarked_text = save_dict["w"] 
    Ys = save_dict["y"]
    highest_probs_lst = save_dict["h"] 


def h_ars(Ys):
    Ys = np.array(Ys)
    h_ars_Ys = -np.log(1-Ys)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_ars_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_opt(Ys, delta0=0.2,theo=False):
    # This is for the optimal score function
    Ys = np.array(Ys)

    def f(r, delta):
        inte_here = np.floor(1/(1-delta))
        rest = 1-(1-delta)*inte_here
        return np.log(inte_here*r**(delta/(1-delta))+ r**(1/rest-1))
    
    h_ars_Ys = f(Ys, delta0)
    
    if theo:
        mu = quad(lambda x: f(x, delta0), 0, 1,epsabs = 1e-10,epsrel=1e-10)
        EX2 = quad(lambda x: f(x, delta0)**2, 0, 1,epsabs = 1e-10,epsrel=1e-10)

        mu, EX2 = mu[0], EX2[0]
        Var = EX2 - mu**2

        h_help_qs = compute_ind_q(1-alpha, mu, Var, check_points)
    else:
        Null_Ys = np.random.uniform(size=(N_trial*2, Final_T))
        Simu_Y = f(Null_Ys, delta0)
        Simu_Y = np.cumsum(Simu_Y, axis=1)
        h_help_qs = np.quantile(Simu_Y, 1-alpha, axis=0)

    cumsum_Ys = np.cumsum(h_ars_Ys, axis=1)
    results = (cumsum_Ys >= h_help_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_log(Ys):
    Ys = np.array(Ys)
    h_log_Ys = np.log(Ys)
    cumsum_Ys = np.cumsum(h_log_Ys, axis=1)
    
    results = (cumsum_Ys >= -h_log_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)


def h_ind(Ys, ind_delta=0.5):
    Ys = np.array(Ys)
    h_ind_Ys = (Ys >= ind_delta)
    cumsum_Ys = np.cumsum(h_ind_Ys, axis=1)
    h_ind_qs = binom.ppf(n=check_points, p = 1-ind_delta, q = 1-alpha)
    results = (cumsum_Ys >= h_ind_qs)
    return np.mean(results,axis=0), np.std(results,axis=0)

##############################################
##
## Check the Type II errors
##
##############################################
result_dict = dict()
result_dict["ars"] = h_ars(Ys)[0].tolist()
result_dict["log"] = h_log(Ys)[0].tolist()

result_dict["opt-0001"] = h_opt(Ys,0.001)[0].tolist()
result_dict["opt-0005"] = h_opt(Ys,0.005)[0].tolist()
result_dict["opt-005"] = h_opt(Ys,0.05)[0].tolist()
result_dict["opt-001"] = h_opt(Ys,0.01)[0].tolist()
result_dict["opt-02"] = h_opt(Ys,0.2)[0].tolist()
result_dict["opt-01"] = h_opt(Ys,0.1)[0].tolist()

result_dict["ind-05"] = h_ind(Ys,0.5)[0].tolist()
result_dict["ind-08"] = h_ind(Ys,0.8)[0].tolist()
result_dict["ind-02"] = h_ind(Ys,0.2)[0].tolist()
result_dict["ind-03"] = h_ind(Ys,0.3)[0].tolist()
result_dict["ind-01"] = h_ind(Ys,0.1)[0].tolist()
result_dict["ind-09"] = h_ind(Ys,0.9)[0].tolist()
result_dict["ind-1/e"] = h_ind(Ys,1/np.exp(1))[0].tolist()
json.dump(result_dict, open(name+"-result"+".json", 'w'))


##############################################
##
## Check the Type I errors
##
##############################################
result_dict = dict()
Null_Ys = np.random.uniform(size=(N_trial, Final_T))
result_dict["ars"] = h_ars(Null_Ys)[0].tolist()

Null_Ys = np.random.uniform(size=(N_trial, Final_T))
result_dict["opt-02"] = h_opt(Null_Ys, 0.2)[0].tolist()
result_dict["opt-01"] = h_opt(Null_Ys, 0.1)[0].tolist()
result_dict["opt-005"] = h_opt(Null_Ys, 0.05)[0].tolist()
result_dict["opt-001"] = h_opt(Null_Ys, 0.01)[0].tolist()
result_dict["opt-0005"] = h_opt(Null_Ys, 0.005)[0].tolist()
result_dict["opt-0001"] = h_opt(Null_Ys, 0.001)[0].tolist()

Null_Ys = np.random.uniform(size=(N_trial, Final_T))
result_dict["log"] = h_log(Null_Ys)[0].tolist()
result_dict["ind-05"] = h_ind(Null_Ys, 0.5)[0].tolist()
result_dict["ind-08"] = h_ind(Null_Ys, 0.8)[0].tolist()
result_dict["ind-02"] = h_ind(Null_Ys, 0.2)[0].tolist()
result_dict["ind-03"] = h_ind(Null_Ys, 0.3)[0].tolist()
result_dict["ind-01"] = h_ind(Null_Ys, 0.1)[0].tolist()
result_dict["ind-09"] = h_ind(Null_Ys, 0.9)[0].tolist()
result_dict["ind-1/e"] = h_ind(Null_Ys, 1/np.exp(1))[0].tolist()
json.dump(result_dict, open(name+"-null"+".json", 'w'))
