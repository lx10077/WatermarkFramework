#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 6])
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({
    'font.size': 13,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsfonts} \usepackage{amsmath} '
})


size = "1p3"
temp = 0.1

print(size)
exp1_name = f"results_data/{size}B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp{temp}-result"
exp3_name = f"results_data/{size}B-gumbel-c4-m200-T500-skipgram_prf-15485863-temp{temp}-null"
exp2_name = f"results_data/{size}B-transform-c4-m200-T500-skipgram_prf-15485863-temp{temp}-result"
exp4_name = f"results_data/{size}B-transform-c4-m200-T500-skipgram_prf-15485863-temp{temp}-null"
K = 1000
alpha = 0.05


def labelize(name):
    if name == "ars":
        return r"$h_{\mathrm{ars}}$"
    if name == "opt-02":
        return r"$h_{\mathrm{gum}, 0.2}^{\star}$"
    if name == "opt-005":
        return r"$h_{\mathrm{gum}, 0.05}^{\star}$"
    if name == "opt-0005":
        return r"$h_{\mathrm{gum}, 0.005}^{\star}$"
    if name == "opt-0001":
        return r"$h_{\mathrm{gum}, 0.001}^{\star}$"
    if name == "opt-01":
        return r"$h_{\mathrm{gum}, 0.1}^{\star}$"
    if name == "opt-001":
        return r"$h_{\mathrm{gum}, 0.01}^{\star}$"
    elif name == "log":
        return r"$h_{\mathrm{log}}$"
    elif name == "ind-05":
        return r"$h_{\mathrm{ind}, 0.5}$"
    elif name == "ind-08":
        return r"$h_{\mathrm{ind}}, 0.8$"
    elif name == "ind-02":
        return r"$h_{\mathrm{ind}}, 0.2$"
    elif name == "ind-03":
        return r"$h_{\mathrm{ind}}, 0.3$"
    elif name == "ind-01":
        return r"$h_{\mathrm{ind}}, 0.1$"
    elif name == "ind-09":
        return r"$h_{\mathrm{ind}}, 0.9$"
    elif name == "ind-1/e":
        return r"$h_{\mathrm{ind},\mathrm{e}^{-1}}$"
    else:
        raise KeyError(f"{name}")


def labelize_inv(name):
    if name == "dif":
        return r"$h_{\mathrm{neg}}$"
    if name == "dif-opt-01":
        return r"$h_{\mathrm{dif}, 0.1}^{\star}$"
    if name == "dif-opt-02":
        return r"$h_{\mathrm{dif}, 0.2}^{\star}$"
    if name == "dif-opt-005":
        return r"$h_{\mathrm{dif}, 0.05}^{\star}$"
    if name == "dif-opt-001":
        return r"$h_{\mathrm{dif}, 0.01}^{\star}$"
    if name == "dif-opt-0005":
        return r"$h_{\mathrm{dif}, 0.005}^{ \star}$"
    if name == "dif-opt-00005":
        return r"$h_{\mathrm{dif}, 0.0005}^{\star}$"
    if name == "dif-opt-00001":
        return r"$h_{\mathrm{dif}, 0.0001}^{\star}$"
    if name == "dif-opt-0001":
        return r"$h_{\mathrm{dif}, 0.001}^{\star}$"
    if name == "dif-opt-01":
        return r"$h_{\mathrm{dif}, 0.1}^{ \star}$"
    if name == "dif-opt-02":
        return r"$h_{\mathrm{dif}, 0.2}^{\star}$"
    if name == "dif-opt-03":
        return r"$h_{\mathrm{dif}, 0.3}^{\star}$"
    if name == "dif-opt-005":
        return r"$h_{\mathrm{dif}, 0.05}^{\star}$"
    if name == "dif-opt-001":
        return r"$h_{\mathrm{dif}, 0.01}^{\star}$"
    if name == "dif-opt-0005":
        return r"$h_{\mathrm{dif}, 0.005}^{\star}$"
    else:
        raise KeyError(f"No {name}" )


linestyles = ["-", ":","--", "-."]
colors = [  "tab:blue", "tab:orange", "tab:gray","tab:red","black", "tab:brown", "tab:purple",   "tab:pink",]


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,6))
x_l = np.arange(1,201)

save_dict = json.load(open(exp3_name+".json", "r"))
first = 200
for j, algo in enumerate(["ars", "log", "ind-1/e", "opt-01"]):
    mean = np.array(save_dict[algo])[:first]
    ax[0][0].plot(x_l[3:first], mean[3:first], label=labelize(algo), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])

ax[0][0].axhline(y=0.05, color="black", linestyle="dotted")
ax[0][0].set_ylabel(r"Type I error")
ax[0][0].set_xlabel(r"Unwatermarked text length")
ax[0][0].set_yscale('log')


save_dict = json.load(open(exp1_name+".json", "r"))
first = 200
for j, algo in enumerate(["ars", "log", "ind-1/e", "opt-01"]):
    mean = 1-np.array(save_dict[algo])[:first]
    ax[0][1].plot(x_l[3:first], mean[3:first], label=labelize(algo), linestyle=linestyles[j%len(linestyles)], color=colors[j%len(colors)])

ax[0][1].legend()
ax[0][1].set_ylabel(r"Type II error")
ax[0][1].set_xlabel(r"Watermarked text length")
ax[0][1].set_yscale('log')


linestyles = ["-", ":","--", "-."]
colors = [  "tab:blue", "tab:orange", "tab:gray","tab:red","black", "tab:brown", "tab:purple",   "tab:pink",]


x_ll = np.arange(1, 201)
save_dict = json.load(open(exp4_name+".json", "r"))
for j, algo in enumerate([ "dif","dif-opt-01", "dif-opt-001","dif-opt-0001"]):
    mean = np.array(save_dict[algo])
    ax[1][0].plot(x_ll[3:200], mean[3:200], label=labelize_inv(algo), linestyle=linestyles[j%5],color=colors[j%8])

ax[1][0].axhline(y=0.05, color="black", linestyle="dotted")
ax[1][0].set_ylabel(r"Type I error")
ax[1][0].set_xlabel(r"Unwatermarked text length")
ax[1][0].set_yscale('log')


save_dict = json.load(open(exp2_name+".json", "r"))
for j, algo in enumerate([ "dif","dif-opt-01", "dif-opt-001","dif-opt-0001"]):
    mean = 1-np.array(save_dict[algo])
    ax[1][1].plot(x_ll[:200], mean[:200], label=labelize_inv(algo), linestyle=linestyles[j%5],color=colors[j%8])

ax[1][1].legend()
ax[1][1].set_ylabel(r"Type II error")
ax[1][1].set_xlabel(r"Watermarked text length")
ax[1][1].set_yscale('log')


plt.tight_layout()
plt.savefig(f'real-data-{size}-{temp}.pdf', dpi=300)
