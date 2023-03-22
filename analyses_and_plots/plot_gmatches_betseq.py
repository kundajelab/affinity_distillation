import os
import json
import gzip
import math
import codecs
import pyBigWig
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import optparse

parser = optparse.OptionParser()
parser.add_option('--target',
    action="store", dest="target",
    help="target", default=None)
options, args = parser.parse_args()
target = options.target

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

bedFile = open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/100_around_summits.bed")
peaksFile = open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/100_around_summits.fa")
peak_seqs = []
for line in peaksFile:
    if line[0] == '>': continue
    else: peak_seqs.append(line.strip())
peak_coords = []
for line in bedFile:
    chrm,s,e,_ = line.strip().split('\t')
    peak_coords.append((chrm, int(s), int(e)))

seqToCoord = {}
for probe in seqs:
    for idx, peak_seq in enumerate(peak_seqs):
        chrm, start, end = peak_coords[idx]
        loc = peak_seq.find(probe)
        if loc != -1:
            if probe not in seqToCoord:
                seqToCoord[probe] = []
            seqToCoord[probe].append((chrm, start+loc))

if len(seqToCoord.keys()) == 0:
    print("No Matches")
    
pred_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/preds/"
obj_text = codecs.open(pred_dir+target+"_betseq.json", 'r', encoding='utf-8').read()
seqToDeltaLogCounts = json.loads(obj_text)

xvals = []
yvals = []
for seq in seqToCoord:
    xvals.append(seqToLabel[seq])
    yvals.append(float(seqToDeltaLogCounts[seq][-1]))

path = 'comparison_figs/'+target+'_betseq/'
font = {'weight' : 'bold', 'size'   : 14}
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.figure()
matplotlib.rc('font', **font)
metadata = {}
metadata["x-axis"] = "affinity"
metadata["y-axis"] = "delta log counts"
metadata["Number of points"] = len(xvals)
metadata["spearman"] = spearmanr(xvals, yvals)[0]
metadata["pearson"] = pearsonr(xvals, yvals)[0]
metadata["rmse"] = math.sqrt(mean_squared_error(xvals, yvals))
with open(path+'gmatches_metadata.json', 'w') as fp: json.dump(metadata, fp)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.5)
plt.savefig(path+'gmatches.png', dpi=300, format='png')
plt.clf()