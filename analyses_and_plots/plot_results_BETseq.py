import os
import re
import json
import gzip
import codecs
import math
from math import log, ceil
import numpy as np
import modisco
import modisco.tfmodisco_workflow.workflow
from modisco.tfmodisco_workflow import workflow
import h5py
import pandas as pd
import modisco.util
from collections import Counter
from modisco.visualization import viz_sequence
import modisco.affinitymat.core
import modisco.cluster.phenograph.core
import modisco.cluster.phenograph.cluster
import modisco.cluster.core
import modisco.aggregator
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr, gaussian_kde
font = {'weight' : 'bold', 'size'   : 14}
import optparse

parser = optparse.OptionParser()
parser.add_option('--target',
    action="store", dest="target",
    help="target", default=None)
options, args = parser.parse_args()
target = options.target

if not os.path.exists("comparison_figs/"+target+"_betseq"):
    os.makedirs("comparison_figs/"+target+"_betseq")
if not os.path.exists("comparison_figs/"+target+"_betseq/de_novo"):
    os.makedirs("comparison_figs/"+target+"_betseq/de_novo")
if not os.path.exists("comparison_figs/"+target+"_betseq/calibrated"):
    os.makedirs("comparison_figs/"+target+"_betseq/calibrated")

class CalibratorFactory(object):
    def __call__(self, valid_preacts, valid_labels):
        raise NotImplementedError()

class LinearRegression(CalibratorFactory):
    def __init__(self, verbose=True):
        self.verbose = verbose 

    def __call__(self, valid_preacts, valid_labels):
        lr = LR().fit(valid_preacts.reshape(-1, 1), valid_labels)
    
        def calibration_func(preact):
            return lr.predict(preact.reshape(-1, 1))

        return calibration_func

def plot_and_save(xvals, yvals, key, xlabel, ylabel, path, same_scale=False):
    xy = np.vstack([xvals,yvals])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    matplotlib.rc('font', **font)
    if same_scale:
        min_lim = min(np.min(xvals), np.min(yvals))
        max_lim = max(np.max(xvals), np.max(yvals))
        plt.xlim(min_lim-0.5, max_lim+0.5)
        plt.ylim(min_lim-0.5, max_lim+0.5)
        plt.gca().set_aspect('equal', adjustable='box')
    metadata = {}
    metadata["key"] = key
    metadata["x-axis"] = xlabel
    metadata["y-axis"] = ylabel
    metadata["Number of points"] = len(xvals)
    metadata["spearman"] = spearmanr(xvals, yvals)[0]
    metadata["pearson"] = pearsonr(xvals, yvals)[0]
    metadata["rmse"] = math.sqrt(mean_squared_error(xvals, yvals))
    with open(path+key+'_metadata.json', 'w') as fp: json.dump(metadata, fp)
    plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.2)
    if same_scale:
        plt.plot([min_lim-0.5, max_lim+0.5], [min_lim-0.5, max_lim+0.5], color="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(path+key+'.png', dpi=300, format='png')
    plt.clf()

seqs = []
all_xvals = []
seqToLabel = {}
firstLine = True
flanks_path = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/gcPBM/all_scaled_nn_preds.txt"
with open(flanks_path) as inp:
    for line in inp:
        if firstLine:
            firstLine = False
            continue
        flank,protein,_,_,_,ddG,_ = line.strip().split('\t')
        insert = flank[:5] + "CACGTG" + flank[5:]
        if protein.lower() in target:
            seqs.append(insert)
            all_xvals.append(float(ddG))
            seqToLabel[insert] = float(ddG)

num_samples = min(1000, ceil(0.1*len(seqs)))
print(num_samples, len(seqs))
calibration_samples = np.random.choice(seqs, num_samples, replace=False)
sample_labels = []
for seq in calibration_samples:
    sample_labels.append(seqToLabel[seq])
sample_labels = np.array(sample_labels)

# MoDISco Lite
if not os.path.exists("comparison_figs/"+target+"_betseq/de_novo/modisco_lite"):
    os.makedirs("comparison_figs/"+target+"_betseq/de_novo/modisco_lite")
if not os.path.exists("comparison_figs/"+target+"_betseq/calibrated/modisco_lite"):
    os.makedirs("comparison_figs/"+target+"_betseq/calibrated/modisco_lite")
    
filename = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/modisco-lite/"+target+"/modisco_counts_results.h5"
f = h5py.File(filename, 'r')
pattern_list = len(f['pos_patterns'])

def trim_motif(cwm_fwd, max_length=None):
    trim_threshold=0.3
    score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
    trim_thresh_fwd = np.max(score_fwd) * trim_threshold
    pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
    start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1)
    # max length restricted to seq length which is 16bp
    if max_length != None and (end_fwd - start_fwd) > max_length:
        center = int((start_fwd+end_fwd)/2)
        start_fwd = center - int(max_length/2)
        end_fwd = center + int(max_length/2)
    trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
    return trimmed_cwm_fwd

CWMs = []
for idx in range(min(10, pattern_list)):
    # max length restricted to seq length which is 16bp
    cwm = trim_motif(f['pos_patterns']['pattern_'+str(idx)]['contrib_scores'], 16)
    CWMs.append(cwm)
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(8,5), dpi=300)
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(ax, cwm,
                                       height_padding_factor=0.2,
                                       length_padding=1.0,
                                       subticks_frequency=1.0,
                                       highlight={})
    fig.savefig('comparison_figs/'+target+'_betseq/de_novo/modisco_lite/cwm_'+str(idx)+'.png', dpi=300)

complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'} 
def getRevComp(seq):  # reverse complement function
    ret = ""
    for bp in seq.upper(): ret += complement[bp]
    return ret[::-1]

def generate_matrix(seq):
    seq_matrix = np.zeros((len(seq), 4))
    for j in range(len(seq)):
        if seq[j] == 'A':
            seq_matrix[j,0] = 1
        elif seq[j] == 'C':
            seq_matrix[j,1] = 1
        elif seq[j] == 'G':
            seq_matrix[j,2] = 1
        elif seq[j] == 'T':
            seq_matrix[j,3] = 1
    return seq_matrix

def get_PWM_sum_score(sequence, score_matrix):
    score_len = score_matrix.shape[0]
    score = 0
    for j in range(len(sequence) - score_len + 1):
        seq_matrix = generate_matrix(sequence[j:j+score_len])
        score += np.sum(score_matrix * seq_matrix)
    rc_sequence = getRevComp(sequence)
    rc_score = 0
    for j in range(len(rc_sequence) - score_len + 1):
        seq_matrix = generate_matrix(rc_sequence[j:j+score_len])
        rc_score += np.sum(score_matrix * seq_matrix)
    return max(score, rc_score)

def get_PWM_max_score(sequence, score_matrix):
    score_len = score_matrix.shape[0]
    scores = []
    for j in range(len(sequence) - score_len + 1):
        seq_matrix = generate_matrix(sequence[j:j+score_len])
        scores.append(np.sum(score_matrix * seq_matrix))
    rc_sequence = getRevComp(sequence)
    for j in range(len(rc_sequence) - score_len + 1):
        seq_matrix = generate_matrix(rc_sequence[j:j+score_len])
        scores.append(np.sum(score_matrix * seq_matrix))
    return max(scores)

for idx, cwm in enumerate(CWMs):
#     if os.path.exists('comparison_figs/'+target+'_betseq/calibrated/modisco/max_residuals_'+str(idx)+'.png'): continue
#     print("Modisco pattern "+str(idx))
    yvals_sum = []
    yvals_max = []
    for seq in seqs:
        yvals_sum.append(get_PWM_sum_score(seq, cwm))
        yvals_max.append(get_PWM_max_score(seq, cwm))
    yvals_sum = np.array(yvals_sum)
    yvals_max = np.array(yvals_max)
    plot_and_save(all_xvals, yvals_sum, "modisco_sum_"+str(idx),
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Score",
              path='comparison_figs/'+target+'_betseq/de_novo/modisco_lite/')
    plot_and_save(all_xvals, yvals_max, "modisco_max_"+str(idx),
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Score",
              path='comparison_figs/'+target+'_betseq/de_novo/modisco_lite/')

    sample_sums = []
    sample_maxs = []
    for seq in calibration_samples:
        sample_sums.append(get_PWM_sum_score(seq, cwm))
        sample_maxs.append(get_PWM_max_score(seq, cwm))
    sample_sums = np.array(sample_sums)
    sample_maxs = np.array(sample_maxs)
    lr1 = LinearRegression()
    calibration_func1 = lr1(sample_sums, sample_labels)
    yvals_sum = calibration_func1(yvals_sum)
    lr2 = LinearRegression()
    calibration_func2 = lr2(sample_maxs, sample_labels)
    yvals_max = calibration_func2(yvals_max)
    plot_and_save(all_xvals, yvals_sum, "modisco_sum_"+str(idx),
                  xlabel="True Log Signal Intensity",
                  ylabel="Pred. Log Signal Intensity",
                  path='comparison_figs/'+target+'_betseq/calibrated/modisco_lite/',
                  same_scale=True)
    residuals = all_xvals - yvals_sum  # residuals are y - yhat
    plot_and_save(yvals_sum, residuals, "sum_residuals_"+str(idx),
                  xlabel="Predictions",
                  ylabel="Residuals",
                  path='comparison_figs/'+target+'_betseq/calibrated/modisco_lite/')
    plot_and_save(all_xvals, yvals_max, "modisco_max_"+str(idx),
                  xlabel="True Log Signal Intensity",
                  ylabel="Pred. Log Signal Intensity",
                  path='comparison_figs/'+target+'_betseq/calibrated/modisco_lite/',
                  same_scale=True)
    residuals = all_xvals - yvals_max  # residuals are y - yhat
    plot_and_save(yvals_max, residuals, "max_residuals_"+str(idx),
                  xlabel="Predictions",
                  ylabel="Residuals",
                  path='comparison_figs/'+target+'_betseq/calibrated/modisco_lite/')
    
# Affinity Distill
if not os.path.exists("comparison_figs/"+target+"_betseq/de_novo/affinity_distill"):
    os.makedirs("comparison_figs/"+target+"_betseq/de_novo/affinity_distill")
if not os.path.exists("comparison_figs/"+target+"_betseq/calibrated/affinity_distill"):
    os.makedirs("comparison_figs/"+target+"_betseq/calibrated/affinity_distill")

pred_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/preds/"
obj_text = codecs.open(pred_dir+target+"_betseq.json", 'r', encoding='utf-8').read()
seqToDeltaLogCounts = json.loads(obj_text)

yvals = []
for seq in seqs:
    yvals.append(float(seqToDeltaLogCounts[seq][-1]))
yvals = np.array(yvals)
plot_and_save(all_xvals, yvals, "affinity_distill",
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Score",
              path='comparison_figs/'+target+'_betseq/de_novo/affinity_distill/')

sample_preds = []
for seq in calibration_samples:
    sample_preds.append(float(seqToDeltaLogCounts[seq][-1]))
sample_preds = np.array(sample_preds)
pre_yvals = []
for seq in seqs:
    pre_yvals.append(float(seqToDeltaLogCounts[seq][-1]))
pre_yvals = np.array(pre_yvals)
lr1 = LinearRegression()
calibration_func1 = lr1(sample_preds, sample_labels)
yvals = calibration_func1(pre_yvals)
plot_and_save(all_xvals, yvals, "affinity_distill",
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Log Signal Intensity",
              path='comparison_figs/'+target+'_betseq/calibrated/affinity_distill/',
              same_scale=True)
residuals = all_xvals - yvals  # residuals are y - yhat
plot_and_save(yvals, residuals, "residuals",
              xlabel="Predictions",
              ylabel="Residuals",
              path='comparison_figs/'+target+'_betseq/calibrated/affinity_distill/')

# Weeder
if not os.path.exists("comparison_figs/"+target+"_betseq/de_novo/weeder"):
    os.makedirs("comparison_figs/"+target+"_betseq/de_novo/weeder")
if not os.path.exists("comparison_figs/"+target+"_betseq/calibrated/weeder"):
    os.makedirs("comparison_figs/"+target+"_betseq/calibrated/weeder")

lite_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"
filepath = lite_dir+target+"/100_around_summits.fa.matrix.w2"
PWMs = []
lines = []
for idx, line in enumerate(open(filepath)):
    if line[0] == ">":
        if len(lines) == 4:
            PWMs.append(np.array(lines)[:,1:].astype('float').T)
            lines = []
            if len(PWMs) == 10: break
    else: lines.append(line.rstrip().split('\t'))

for idx, pwm in enumerate(PWMs):
    centered_pwm = np.array([i-np.mean(i) for i in pwm])
    matplotlib.rc('font', **font)
    fig = plt.figure(figsize=(8,5), dpi=300)
    ax = fig.add_subplot(111)
    viz_sequence.plot_weights_given_ax(ax, centered_pwm,
                                        height_padding_factor=0.2,
                                        length_padding=1.0,
                                        subticks_frequency=1.0,
                                        highlight={})
    fig.savefig('comparison_figs/'+target+'_betseq/de_novo/weeder/pwm_'+str(idx)+'.png', dpi=300)

for idx, pwm in enumerate(PWMs):
    yvals_sum = []
    yvals_max = []
    for seq in seqs:
        yvals_sum.append(get_PWM_sum_score(seq, pwm))
        yvals_max.append(get_PWM_max_score(seq, pwm))
    yvals_sum = np.array(yvals_sum)
    yvals_max = np.array(yvals_max)
    plot_and_save(all_xvals, yvals_sum, "weeder_sum_"+str(idx),
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Score",
              path='comparison_figs/'+target+'_betseq/de_novo/weeder/')
    plot_and_save(all_xvals, yvals_max, "weeder_max_"+str(idx),
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Score",
              path='comparison_figs/'+target+'_betseq/de_novo/weeder/')

    sample_sums = []
    sample_maxs = []
    for seq in calibration_samples:
        sample_sums.append(get_PWM_sum_score(seq, pwm))
        sample_maxs.append(get_PWM_max_score(seq, pwm))
    sample_sums = np.array(sample_sums)
    sample_maxs = np.array(sample_maxs)
    lr1 = LinearRegression()
    calibration_func1 = lr1(sample_sums, sample_labels)
    yvals_sum = calibration_func1(yvals_sum)
    lr2 = LinearRegression()
    calibration_func2 = lr2(sample_maxs, sample_labels)
    yvals_max = calibration_func2(yvals_max)
    plot_and_save(all_xvals, yvals_sum, "weeder_sum_"+str(idx),
                  xlabel="True Log Signal Intensity",
                  ylabel="Pred. Log Signal Intensity",
                  path='comparison_figs/'+target+'_betseq/calibrated/weeder/',
                  same_scale=True)
    residuals = all_xvals - yvals_sum  # residuals are y - yhat
    plot_and_save(yvals_sum, residuals, "sum_residuals_"+str(idx),
                  xlabel="Predictions",
                  ylabel="Residuals",
                  path='comparison_figs/'+target+'_betseq/calibrated/weeder/')
    plot_and_save(all_xvals, yvals_max, "weeder_max_"+str(idx),
                  xlabel="True Log Signal Intensity",
                  ylabel="Pred. Log Signal Intensity",
                  path='comparison_figs/'+target+'_betseq/calibrated/weeder/',
                  same_scale=True)
    residuals = all_xvals - yvals_max  # residuals are y - yhat
    plot_and_save(yvals_max, residuals, "max_residuals_"+str(idx),
                  xlabel="Predictions",
                  ylabel="Residuals",
                  path='comparison_figs/'+target+'_betseq/calibrated/weeder/')

# STREME
if not os.path.exists("comparison_figs/"+target+"_betseq/de_novo/streme"):
    os.makedirs("comparison_figs/"+target+"_betseq/de_novo/streme")
if not os.path.exists("comparison_figs/"+target+"_betseq/calibrated/streme"):
    os.makedirs("comparison_figs/"+target+"_betseq/calibrated/streme")

lite_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"
filepath = lite_dir+target+"/streme_out/streme.txt"
reading = False
PWMs = []
for idx, line in enumerate(open(filepath)):
    if "letter-probability" in line:
        reading = True
        width = int(line.rstrip().split(' ')[5])
        lines = []
        continue
    if reading:
        lines.append(line.rstrip().split(' '))
        width -= 1
        if width == 0:
            reading = False
            prob_mat = np.array(lines)[:,1:].astype('float')
            pwm = np.log2((prob_mat/0.25)+1e-4)
            centered_pwm = np.array([i-np.mean(i) for i in pwm])
            matplotlib.rc('font', **font)
            fig = plt.figure(figsize=(8,5), dpi=300)
            ax = fig.add_subplot(111)
            viz_sequence.plot_weights_given_ax(ax, centered_pwm,
                                                height_padding_factor=0.2,
                                                length_padding=1.0,
                                                subticks_frequency=1.0,
                                                highlight={})
            fig.savefig('comparison_figs/'+target+'_betseq/de_novo/streme/pwm_'+str(len(PWMs))+'.png', dpi=300)  
            PWMs.append(pwm)
            if len(PWMs) == 10: break

for idx, pwm in enumerate(PWMs):
    yvals_sum = []
    yvals_max = []
    for seq in seqs:
        yvals_sum.append(get_PWM_sum_score(seq, pwm))
        yvals_max.append(get_PWM_max_score(seq, pwm))
    yvals_sum = np.array(yvals_sum)
    yvals_max = np.array(yvals_max)
    plot_and_save(all_xvals, yvals_sum, "streme_sum_"+str(idx),
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Score",
              path='comparison_figs/'+target+'_betseq/de_novo/streme/')
    plot_and_save(all_xvals, yvals_max, "streme_max_"+str(idx),
              xlabel="True Log Signal Intensity",
              ylabel="Pred. Score",
              path='comparison_figs/'+target+'_betseq/de_novo/streme/')

    sample_sums = []
    sample_maxs = []
    for seq in calibration_samples:
        sample_sums.append(get_PWM_sum_score(seq, pwm))
        sample_maxs.append(get_PWM_max_score(seq, pwm))
    sample_sums = np.array(sample_sums)
    sample_maxs = np.array(sample_maxs)
    lr1 = LinearRegression()
    calibration_func1 = lr1(sample_sums, sample_labels)
    yvals_sum = calibration_func1(yvals_sum)
    lr2 = LinearRegression()
    calibration_func2 = lr2(sample_maxs, sample_labels)
    yvals_max = calibration_func2(yvals_max)
    plot_and_save(all_xvals, yvals_sum, "streme_sum_"+str(idx),
                  xlabel="True Log Signal Intensity",
                  ylabel="Pred. Log Signal Intensity",
                  path='comparison_figs/'+target+'_betseq/calibrated/streme/',
                  same_scale=True)
    residuals = all_xvals - yvals_sum  # residuals are y - yhat
    plot_and_save(yvals_sum, residuals, "sum_residuals_"+str(idx),
                  xlabel="Predictions",
                  ylabel="Residuals",
                  path='comparison_figs/'+target+'_betseq/calibrated/streme/')
    plot_and_save(all_xvals, yvals_max, "streme_max_"+str(idx),
                  xlabel="True Log Signal Intensity",
                  ylabel="Pred. Log Signal Intensity",
                  path='comparison_figs/'+target+'_betseq/calibrated/streme/',
                  same_scale=True)
    residuals = all_xvals - yvals_max  # residuals are y - yhat
    plot_and_save(yvals_max, residuals, "max_residuals_"+str(idx),
                  xlabel="Predictions",
                  ylabel="Residuals",
                  path='comparison_figs/'+target+'_betseq/calibrated/streme/')

# Overall
results_dir = "comparison_figs/"+target+"_betseq/calibrated/"
w = 0.8    # bar width
x = [1, 2, 3, 4]     # x-coordinates of bars

criteria = "rmse"
meta = {}
weeder = []
weeder_x = []
meta["best_weeder"] = 10^5
for filename in os.listdir(results_dir+"weeder/"):
    if "weeder_max_" in filename and filename.endswith("metadata.json"):
        data = json.load(open(results_dir+"weeder/"+filename))
        weeder.append(float(data[criteria]))
        weeder_x.append(1 + np.random.random() * (w/2) - (w/4))   # distribute coords randomly across half width of bar
        if weeder[-1] <= meta["best_weeder"]:
            meta["best_weeder"] = weeder[-1]
            meta["best_weeder_name"] = filename
        if filename == "weeder_max_0_metadata.json":
            top_weeder_x = weeder_x[-1]
            top_weeder_y = weeder[-1]
streme = []
streme_x = []
meta["best_streme"] = 10^5
for filename in os.listdir(results_dir+"streme/"):
    if "streme_max_" in filename and filename.endswith("metadata.json"):
        data = json.load(open(results_dir+"streme/"+filename))
        streme.append(float(data[criteria]))
        streme_x.append(2 + np.random.random() * (w/2) - (w/4))
        if streme[-1] <= meta["best_streme"]:
            meta["best_streme"] = streme[-1]
            meta["best_streme_name"] = filename
        if filename == "streme_max_0_metadata.json":
            top_streme_x = streme_x[-1]
            top_streme_y = streme[-1]
modisco = []
modisco_x = []
meta["best_modisco"] = 10^5
for filename in os.listdir(results_dir+"modisco_lite/"):
    if "modisco_max_" in filename and filename.endswith("metadata.json"):
        data = json.load(open(results_dir+"modisco_lite/"+filename))
        modisco.append(float(data[criteria]))
        modisco_x.append(3 + np.random.random() * (w/2) - (w/4))
        if modisco[-1] <= meta["best_modisco"]:
            meta["best_modisco"] = modisco[-1]
            meta["best_modisco_name"] = filename
        if filename == "modisco_max_0_metadata.json":
            top_modisco_x = modisco_x[-1]
            top_modisco_y = modisco[-1]
affinity_distill = []
data = json.load(open(results_dir+"affinity_distill/affinity_distill_metadata.json"))
affinity_distill.append(float(data[criteria]))
meta["affinity_distill"] = affinity_distill[0]

y = [weeder, streme, modisco, affinity_distill]
coords_x = [weeder_x, streme_x, modisco_x]
top_x = [top_weeder_x, top_streme_x, top_modisco_x]
top_y = [top_weeder_y, top_streme_y, top_modisco_y]
fig, ax = plt.subplots()
ax.bar(x, height=[np.mean(yi) for yi in y],
       yerr=[np.std(yi) for yi in y],    # error bars
       capsize=12, # error bar cap width in points
       width=w,    # bar width
       tick_label=["Weeder2", "STREME", "MoDISco", "Distill"],
       color=(0,0,0,0),  # face color transparent
       edgecolor='#1f77b4')
for i in range(len(coords_x)):
    ax.scatter(coords_x[i], y[i], color='dimgrey')
    ax.annotate("top", (top_x[i], top_y[i]))
ax.scatter(top_x, top_y, color='red')
with open(results_dir+criteria+'.json', 'w') as fp: json.dump(meta, fp)
fig.savefig(results_dir+criteria+'.png', dpi=300, format='png')

for criteria in ["spearman", "pearson"]:
    meta = {}
    weeder = []
    weeder_x = []
    meta["best_weeder"] = -100
    for filename in os.listdir(results_dir+"weeder/"):
        if "weeder_max_" in filename and filename.endswith("metadata.json"):
            data = json.load(open(results_dir+"weeder/"+filename))
            weeder.append(float(data[criteria])*100)
            weeder_x.append(1 + np.random.random() * (w/2) - (w/4))   # distribute coords randomly across half width of bar
            if weeder[-1] >= meta["best_weeder"]:
                meta["best_weeder"] = weeder[-1]
                meta["best_weeder_name"] = filename
            if filename == "weeder_max_0_metadata.json":
                top_weeder_x = weeder_x[-1]
                top_weeder_y = weeder[-1]
    streme = []
    streme_x = []
    meta["best_streme"] = -100
    for filename in os.listdir(results_dir+"streme/"):
        if "streme_max_" in filename and filename.endswith("metadata.json"):
            data = json.load(open(results_dir+"streme/"+filename))
            streme.append(float(data[criteria])*100)
            streme_x.append(2 + np.random.random() * (w/2) - (w/4))
            if streme[-1] >= meta["best_streme"]:
                meta["best_streme"] = streme[-1]
                meta["best_streme_name"] = filename
            if filename == "streme_max_0_metadata.json":
                top_streme_x = streme_x[-1]
                top_streme_y = streme[-1]
    modisco = []
    modisco_x = []
    meta["best_modisco"] = -100
    for filename in os.listdir(results_dir+"modisco_lite/"):
        if "modisco_max_" in filename and filename.endswith("metadata.json"):
            data = json.load(open(results_dir+"modisco_lite/"+filename))
            modisco.append(float(data[criteria])*100)
            modisco_x.append(3 + np.random.random() * (w/2) - (w/4))
            if modisco[-1] >= meta["best_modisco"]:
                meta["best_modisco"] = modisco[-1]
                meta["best_modisco_name"] = filename
            if filename == "modisco_max_0_metadata.json":
                top_modisco_x = modisco_x[-1]
                top_modisco_y = modisco[-1]
    affinity_distill = []
    data = json.load(open(results_dir+"affinity_distill/affinity_distill_metadata.json"))
    affinity_distill.append(float(data[criteria])*100)
    meta["affinity_distill"] = affinity_distill[0]

    y = [weeder, streme, modisco, affinity_distill]
    coords_x = [weeder_x, streme_x, modisco_x]
    top_x = [top_weeder_x, top_streme_x, top_modisco_x]
    top_y = [top_weeder_y, top_streme_y, top_modisco_y]
    fig, ax = plt.subplots()
    ax.bar(x,
           height=[np.mean(yi) for yi in y],
           yerr=[np.std(yi) for yi in y],    # error bars
           capsize=12, # error bar cap width in points
           width=w,    # bar width
           tick_label=["Weeder2", "STREME", "MoDISco", "Distill"],
           color=(0,0,0,0),  # face color transparent
           edgecolor='#1f77b4')
    for i in range(len(coords_x)):
        ax.scatter(coords_x[i], y[i], color='dimgrey')
        ax.annotate("top", (top_x[i], top_y[i]))
    ax.set_ylim([0, 100])
    ax.scatter(top_x, top_y, color='red')
    with open(results_dir+criteria+'.json', 'w') as fp: json.dump(meta, fp)
    fig.savefig(results_dir+criteria+'.png', dpi=300, format='png')