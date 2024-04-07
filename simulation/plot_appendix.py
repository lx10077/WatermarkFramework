#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import math as ma
import numpy as np
from tqdm import tqdm
from IPython import embed
from scipy.integrate import quad, dblquad
from scipy.stats import gamma, norm

import json
# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'lines.linewidth': 1,
    'font.size': 13,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} '
})


use_log = True

fig0_name = "results_data/K1000N5000c5key23333T1000Delta0.1-alpha0.05-max-result"
fig1_name = "results_data/K1000N5000c5key23333T700Delta0.3-alpha0.05-max-result"
fig3_name = "results_data/K1000N5000c5key23333T700Delta0.7-alpha0.05-max-result"

fig0_name1 = f"results_data/K1000N5000c5key23333T1000Delta0.1-alpha0.05-inv-result"
fig1_name1 = f"results_data/K1000N5000c5key23333T700Delta0.3-alpha0.05-inv-result"
fig3_name1 = f"results_data/K1000N5000c5key23333T700Delta0.7-alpha0.05-inv-result"


def labelize(name):
    if name == "ars":
        return r"$h_{\mathrm{ars}}$"
    if name == "opt-02":
        return r"$h_{\mathrm{gum},0.2}^{\star}$"
    if name == "opt-005":
        return r"$h_{\mathrm{gum},0.05}^{\star}$"
    if name == "opt-0005":
        return r"$h_{\mathrm{gum},0.005}^{\star}$"
    if name == "opt-0001":
        return r"$h_{\mathrm{gum},0.001}^{\star}$"
    if name == "opt-01":
        return r"$h_{\mathrm{gum},0.1}^{\star}$"
    if name == "opt-001":
        return r"$h_{\mathrm{gum},0.01}^{\star}$"
    elif name == "log":
        return r"$h_{\mathrm{log}}$"
    elif name == "ind-1/e":
        return r"$h_{\mathrm{ind},1/e}$"
    else:
        raise KeyError(f"{name}")


def labelize_inv(name):
    if name == "dif-opt-01":
        return r"$h_{\mathrm{dif},0.1}^{\star}$"
    if name == "dif-opt-005":
        return r"$h_{\mathrm{dif},0.005}^{\star}$"
    if name == "dif-opt-0001":
        return r"$h_{\mathrm{dif},0.001}^{\star}$"
    if name == "dif-opt-001":
        return r"$h_{\mathrm{dif},0.01}^{\star}$"
    elif name == "id":
        return r"$h_{\mathrm{id}}$"
    elif name == "dif":
        return r"$h_{\mathrm{neg}}$"
    else:
        raise KeyError(f"{name}")
    


alpha = 0.05
N_trial = 5000

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10,9))
linestyles = ["-", "-.", ":", "--", "-."]
colors = ["tab:blue", "tab:orange", "tab:gray", "black", "tab:red", "tab:brown", "tab:purple", "tab:pink",]

first = 700
name = fig0_name
save_dict = json.load(open(name+".json", "r"))
for j, algo in enumerate(["ars", "log", "ind-1/e", "opt-001", "opt-0005"]):
    mean = 1-np.array(save_dict[algo])
    ax[0][0].plot(np.arange(1,1+first), mean[:first], label=labelize(algo), linestyle=linestyles[j%len(linestyles)],color=colors[j%len(colors)])

ax[0][0].set_title(r"$H_1, \Delta \sim {U}(0.001, 0.1)$")
ax[0][0].set_ylabel(r"Type II error")
ax[0][0].set_xlabel(r"Watermarked text length")
if use_log:
    ax[0][0].set_yscale('log')


first = 700
name = fig1_name
save_dict = json.load(open(name+".json", "r"))
for j, algo in enumerate(["ars", "log", "ind-1/e", "opt-001", "opt-0005"]):
    mean = 1-np.array(save_dict[algo])
    ax[1][0].plot(np.arange(1,1+first), mean[:first], label=labelize(algo), linestyle=linestyles[j%len(linestyles)],color=colors[j%len(colors)])

ax[1][0].set_title(r"$H_1, \Delta \sim {U}(0.001, 0.3)$")
ax[1][0].set_ylabel(r"Type II error")
ax[1][0].set_xlabel(r"Watermarked text length")
if use_log:
    ax[1][0].set_yscale('log')


first = 300
name = fig3_name
save_dict = json.load(open(name+".json", "r"))
for j, algo in enumerate(["ars", "log", "ind-1/e", "opt-001", "opt-0005"]):
    mean = 1-np.array(save_dict[algo])
    ax[2][0].plot(np.arange(1,1+first), mean[:first], label=labelize(algo), linestyle=linestyles[j%len(linestyles)],color=colors[j%len(colors)])

ax[2][0].set_title(r"$H_1, \Delta \sim {U}(0.001, 0.7)$")
ax[2][0].set_ylabel(r"Type II error")
ax[2][0].set_xlabel(r"Watermarked text length")
if use_log:
    ax[2][0].set_yscale('log')
ax[2][0].legend()


colors = ["tab:blue", "tab:orange", "black", "tab:red", "tab:gray", "tab:brown", "tab:purple", "tab:pink",]
linestyles = ["-", "-.", "--", "-."]

first = 700
name = fig0_name1
save_dict = json.load(open(name+".json", "r"))
for j, algo in enumerate(["dif", "dif-opt-01", "dif-opt-001", "dif-opt-0001"]):
    mean = 1-np.array(save_dict[algo])[:first]
    ax[0][1].plot(np.arange(1,1+first), mean, label=labelize_inv(algo), linestyle=linestyles[j%(len(linestyles))],color=colors[j%len(colors)])

ax[0][1].set_title(r"$H_1, \Delta \sim {U}(0.001, 0.1)$")
ax[0][1].set_ylabel(r"Type II error")
ax[0][1].set_xlabel(r"Watermarked text length")
ax[0][1].set_yscale('log')


first = 700
name = fig1_name1
save_dict = json.load(open(name+".json", "r"))
for j, algo in enumerate(["dif", "dif-opt-01", "dif-opt-001", "dif-opt-0001"]):
    mean = 1-np.array(save_dict[algo])[:first]
    ax[1][1].plot(np.arange(1,1+first), mean, label=labelize_inv(algo), linestyle=linestyles[j%(len(linestyles))],color=colors[j%len(colors)])

ax[1][1].set_title(r"$H_1, \Delta \sim {U}(0.001, 0.3)$")
ax[1][1].set_ylabel(r"Type II error")
ax[1][1].set_xlabel(r"Watermarked text length")
ax[1][1].set_yscale('log')


first = 700
name = fig3_name1
save_dict = json.load(open(name+".json", "r"))
for j, algo in enumerate(["dif", "dif-opt-01", "dif-opt-001", "dif-opt-0001"]):
    mean = 1-np.array(save_dict[algo])[:first]
    ax[2][1].plot(np.arange(1,1+first), mean, label=labelize_inv(algo), linestyle=linestyles[j%(len(linestyles))],color=colors[j%len(colors)])

ax[2][1].set_title(r"$H_1, \Delta \sim {U}(0.001, 0.7)$")
ax[2][1].set_ylabel(r"Type II error")
ax[2][1].set_xlabel(r"Watermarked text length")
ax[2][1].set_yscale('log')
ax[2][1].legend()


plt.tight_layout()
plt.savefig(f'results_data/simu-main-appendix.pdf', dpi=300)
