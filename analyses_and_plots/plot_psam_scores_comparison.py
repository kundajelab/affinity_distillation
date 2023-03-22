import os
import csv
import json
import gzip
import math
import numpy as np
import pyBigWig
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr, gaussian_kde
from modisco.visualization import viz_sequence
import optparse

parser = optparse.OptionParser()
parser.add_option('--target',
    action="store", dest="target",
    help="target", default=None)
options, args = parser.parse_args()
target = options.target
tf = target.split('_')[0].lower()
font = {'weight' : 'bold', 'size'   : 14}

if not os.path.exists("comparison_figs/"+target+"/PSAM"):
    os.makedirs("comparison_figs/"+target+"/PSAM")

tfToPSAM = {
    'pho4': '/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/PSAMs/Pho4_PSAM_extended.csv',
    'cbf1': '/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/PSAMs/Cbf1_PSAM_extended.csv',
    'gr': '/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/PSAMs/GR_PSAM_Pollymeasurements.csv',
    'max': '/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/PSAMs/MAX_PSAM_MaerklQuake2007.csv',
    'myc': '/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/PSAMs/MYC_PSAM_SolomonAmatiLand1993.csv'
}

lines = []
with open(tfToPSAM[tf]) as handle:
    reader = csv.reader(handle)
    for row in reader:
        lines.append(row)
PSAM = np.array(lines)[1:,1:].T.astype('float')
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(8,5), dpi=300)
ax = fig.add_subplot(111)
viz_sequence.plot_weights_given_ax(ax, PSAM,
                                    height_padding_factor=0.2,
                                    length_padding=1.0,
                                    subticks_frequency=1.0,
                                    highlight={})
fig.savefig('comparison_figs/'+target+'/PSAM/PSAM_matrix.png', dpi=300)

def ddG(Kd1, Kd2, R=1.9872036e-3, T=295):
    return R*T*np.log(Kd2/Kd1)  # if numerically unstable, add 1e-5 to both denominator and overall fraction
ddG_mat = [[-ddG(x,1) for x in j] for j in PSAM]
centered_ddG_mat = np.array([i-np.mean(i) for i in ddG_mat])
matplotlib.rc('font', **font)
fig = plt.figure(figsize=(8,5), dpi=300)
ax = fig.add_subplot(111)
viz_sequence.plot_weights_given_ax(ax, centered_ddG_mat,
                                    height_padding_factor=0.2,
                                    length_padding=1.0,
                                    subticks_frequency=1.0,
                                    highlight={})
fig.savefig('comparison_figs/'+target+'/PSAM/ddG_matrix.png', dpi=300)

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

def get_PSAM_score(sequence, score_matrix):
    score_len = score_matrix.shape[0]
    score = 0
    for j in range(len(sequence) - score_len + 1):
        seq_matrix = generate_matrix(sequence[j:j+score_len])
        prod_matrix = score_matrix * seq_matrix
        score += np.prod(prod_matrix[np.nonzero(prod_matrix)])
    rc_sequence = getRevComp(sequence)
    rc_score = 0
    for j in range(len(rc_sequence) - score_len + 1):
        seq_matrix = generate_matrix(rc_sequence[j:j+score_len])
        prod_matrix = score_matrix * seq_matrix
        rc_score += np.prod(prod_matrix[np.nonzero(prod_matrix)])
    return max(score, rc_score)

bigWigs = (pyBigWig.open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/basename_prefix.pooled.positive.bigwig"),
           pyBigWig.open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/basename_prefix.pooled.negative.bigwig"))
bedFile = open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/100_around_summits.bed")
peaksFile = open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/100_around_summits.fa")

R=1.9872036e-3
T=295
peak_seqs = []
log_psams = []
ddg_mats_max = []
ddg_mats_sum = []
for idx,line in enumerate(peaksFile):
    if line[0] == '>': continue
    curr_seq = line.strip().replace("N", "")
    peak_seqs.append(curr_seq)
    log_psams.append(R*T*np.log(get_PSAM_score(curr_seq, PSAM)))
    ddg_mats_max.append(get_PWM_max_score(curr_seq, np.array(ddG_mat)))
    ddg_mats_sum.append(get_PWM_sum_score(curr_seq, np.array(ddG_mat)))
        
peak_coords = []
pos_counts = []
neg_counts = []
log_counts = []
for line in bedFile:
    chrm,s,e,_ = line.strip().split('\t')
    if "_" in chrm: continue
    start = int(s)
    end = int(e)
    peak_coords.append((chrm, start, end))
    posvals = np.array(bigWigs[0].values(chrm, start, end))
    where_are_NaNs = np.isnan(posvals)
    posvals[where_are_NaNs] = 0.0
    pos_counts.append(posvals)
    negvals = np.array(bigWigs[1].values(chrm, start, end))
    where_are_NaNs = np.isnan(negvals)
    negvals[where_are_NaNs] = 0.0
    neg_counts.append(negvals)
    log_counts.append(np.log(1+np.sum(posvals)+np.sum(negvals)))

xvals = log_psams
yvals = log_counts
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.figure()
matplotlib.rc('font', **font)
metadata = {}
metadata["key"] = "R T Log (PSAM_score)"
metadata["x-axis"] = "affinity score"
metadata["y-axis"] = "log(raw counts+1)"
metadata["Number of points"] = len(xvals)
metadata["spearman"] = spearmanr(xvals, yvals)[0]
metadata["pearson"] = pearsonr(xvals, yvals)[0]
metadata["rmse"] = math.sqrt(mean_squared_error(xvals, yvals))
with open("comparison_figs/"+target+"/PSAM/RTLogPsam_metadata.json", 'w') as fp: json.dump(metadata, fp)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.5)
plt.savefig("comparison_figs/"+target+"/PSAM/RTLogPsam.png", dpi=300, format='png')
plt.clf()

xvals = ddg_mats_max
yvals = log_counts
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.figure()
matplotlib.rc('font', **font)
metadata = {}
metadata["key"] = "ddG mat PWM max score"
metadata["x-axis"] = "affinity score"
metadata["y-axis"] = "log(raw counts+1)"
metadata["Number of points"] = len(xvals)
metadata["spearman"] = spearmanr(xvals, yvals)[0]
metadata["pearson"] = pearsonr(xvals, yvals)[0]
metadata["rmse"] = math.sqrt(mean_squared_error(xvals, yvals))
with open("comparison_figs/"+target+"/PSAM/ddGMax_metadata.json", 'w') as fp: json.dump(metadata, fp)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.5)
plt.savefig("comparison_figs/"+target+"/PSAM/ddGMax.png", dpi=300, format='png')
plt.clf()

xvals = ddg_mats_sum
yvals = log_counts
font = {'weight' : 'bold', 'size'   : 14}
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.figure()
matplotlib.rc('font', **font)
metadata = {}
metadata["key"] = "ddG mat PWM sum score"
metadata["x-axis"] = "affinity score"
metadata["y-axis"] = "log(raw counts+1)"
metadata["Number of points"] = len(xvals)
metadata["spearman"] = spearmanr(xvals, yvals)[0]
metadata["pearson"] = pearsonr(xvals, yvals)[0]
metadata["rmse"] = math.sqrt(mean_squared_error(xvals, yvals))
with open("comparison_figs/"+target+"/PSAM/ddGSum_metadata.json", 'w') as fp: json.dump(metadata, fp)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.5)
plt.savefig("comparison_figs/"+target+"/PSAM/ddGSum.png", dpi=300, format='png')
plt.clf()