import os
import json
import gzip
import math
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
parser.add_option('--library',
    action="store", dest="library",
    help="library", default=None)
options, args = parser.parse_args()
target = options.target
library = options.library

seqs = []
all_xvals = []
seqToLabel = {}
if library == "betseq":
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
else:
    if "gr" in target:
        seqToDdg = {}
        firstLine = True
        with open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/GR/GR_bindingcurves_WT_1_out.csv") as inp:
            for line in inp:
                if firstLine:
                    firstLine = False
                    continue
                Oligo,Kd_estimate,ddG,Motif,Sequence = line.strip().split(',')
                seq = Sequence.upper()[11:33]
                pre = Sequence.upper()[:11]
                post = Sequence.upper()[33:]
                if pre != "CGCAATTGCGA":
                    print(pre)
                    print("CGCAATTGCGA")
                if post != "ACCTTCCTCTCCGGCGGTATGAC":
                    print(post)
                    print("ACCTTCCTCTCCGGCGGTATGAC")
                if seq not in seqToDdg:
                    seqToDdg[seq] = []
                seqToDdg[seq].append(float(ddG))
        firstLine = True
        with open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/GR/GR_bindingcurves_WT_2_out.csv") as inp:
            for line in inp:
                if firstLine:
                    firstLine = False
                    continue
                Oligo,Kd_estimate,ddG,Motif,Sequence = line.strip().split(',')
                seq = Sequence.upper()[11:33]
                pre = Sequence.upper()[:11]
                post = Sequence.upper()[33:]
                if pre != "CGCAATTGCGA":
                    print(pre)
                    print("CGCAATTGCGA")
                if post != "ACCTTCCTCTCCGGCGGTATGAC":
                    print(post)
                    print("ACCTTCCTCTCCGGCGGTATGAC")
                seqToDdg[seq].append(float(ddG))
        for seq in seqToDdg:
            seqs.append(seq)
            all_xvals.append(np.mean(seqToDdg[seq]))
            seqToLabel[seq] = np.mean(seqToDdg[seq])
    elif "pho4" in target or "cbf1" in target:
        data_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/yeast/"
        if "cbf1" in target: filename = "GSM4980359_1uMCbf1_His_alldata.txt.gz"
        else: filename = "GSM4980362_400nMPho4_GST_alldata.txt.gz"
        firstLine = True
        with gzip.open(data_dir+filename, 'rt') as inp:
            for line in inp:
                if firstLine:
                    firstLine = False
                    continue
                curr = line.strip().split('\t')[4]
                if curr == "_": continue
                if curr[36:] != "GTCTTGATTCGCTTGACGCTGCTG": print("Exception: ", curr)
                seqs.append(curr[:36])
                all_xvals.append(math.log(float(line.strip().split('\t')[-1])))
                seqToLabel[seqs[-1]] = all_xvals[-1]
    else:
        all_libraries = {
            "ets1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
            "elk1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
            "gabpa": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log.xlsx",
            "e2f1": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
            "e2f3": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
            "e2f4": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log.xlsx",
            "max": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
            "mxi": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
            "myc": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log.xlsx",
            "runx1": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log.xlsx",
            "runx2": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log.xlsx"
        }
        column = {
            "ets1": "Ets1_100nM",
            "elk1": "Elk1_50nM",
            "gabpa": "Gabpa_100nM",
            "e2f1": "E2f1_250nM",
            "e2f3": "E2f3_250nM",
            "e2f4": "E2f4_500nM",
            "max": "Max",
            "mxi": "Mad_r",
            "myc": "Myc",
            "runx1": "Runx1_50nM",
            "runx2": "Runx2_50nM"
        }
        if target[0].isdigit():
            key = target.split('_')[1].lower()
            if key[-1] == '-': key = key[:-1]
        else: key = target.split('_')[0]
        data_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/gcPBM/"
        dfs = pd.read_excel(data_dir+all_libraries[key])
        all_xvals = dfs[column[key]]
        seqs = dfs['Sequence']
        seqToLabel = {}
        for idx in range(len(seqs)):
            seqToLabel[seqs[idx]] = float(all_xvals[idx])

bigWigs = (pyBigWig.open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/basename_prefix.pooled.positive.bigwig"),
           pyBigWig.open("/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"+target+"/basename_prefix.pooled.negative.bigwig"))
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

else:
    window_len = 100
    seqToLogCounts = {}
    for seq in seqToCoord:
        currentPosCounts= []
        currentNegCounts = []
        for chrm, match_start in seqToCoord[seq]: 
            if "_" in chrm: continue
            center = match_start+int(len(seq)/2)
            start = int(center-(window_len/2))
            end = int(center+(window_len/2))
            posvals = np.array(bigWigs[0].values(chrm, start, end))
            where_are_NaNs = np.isnan(posvals)
            posvals[where_are_NaNs] = 0.0
            currentPosCounts.append(posvals)
            negvals = np.array(bigWigs[1].values(chrm, start, end))
            where_are_NaNs = np.isnan(negvals)
            negvals[where_are_NaNs] = 0.0
            currentNegCounts.append(negvals)
        seqToLogCounts[seq] = np.log(1+np.sum(np.mean(np.array(currentPosCounts), axis = 0)+ \
                                              np.mean(np.array(currentNegCounts), axis = 0)))

    xvals = []
    yvals = []
    for seq in seqToLogCounts:
        xvals.append(seqToLabel[seq])
        yvals.append(seqToLogCounts[seq])
    key = target+'_'+library
    if library == 'betseq': path = 'comparison_figs/'+key+'/'
    else: path = 'comparison_figs/'+target+'/'

    font = {'weight' : 'bold', 'size'   : 14}
    xy = np.vstack([xvals,yvals])
    z = gaussian_kde(xy)(xy)
    plt.figure()
    matplotlib.rc('font', **font)
    metadata = {}
    metadata["key"] = key
    metadata["x-axis"] = "affinity"
    metadata["y-axis"] = "log(raw counts+1)"
    metadata["Number of points"] = len(xvals)
    metadata["spearman"] = spearmanr(xvals, yvals)[0]
    metadata["pearson"] = pearsonr(xvals, yvals)[0]
    metadata["rmse"] = math.sqrt(mean_squared_error(xvals, yvals))
    with open(path+key+'_raw_metadata.json', 'w') as fp: json.dump(metadata, fp)
    plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.5)
    plt.savefig(path+key+'_raw.png', dpi=300, format='png')
    plt.clf()