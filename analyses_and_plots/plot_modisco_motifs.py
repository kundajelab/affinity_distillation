import os
import sys
import pickle
import numpy as np
from math import exp
from scipy import stats
import matplotlib as mpl
mpl.use('Agg')
import matplotlib
import matplotlib.pyplot as plt
from vizsequence.viz_sequence import plot_weights_given_ax
from scipy.special import softmax
import keras
import keras.losses
from keras.models import Model, Sequential, load_model
from keras import backend as K
import numpy.random as rng
import seaborn as sns
from collections import OrderedDict
from basepair.losses import twochannel_multinomial_nll
import modisco
import modisco.tfmodisco_workflow.workflow
from modisco.tfmodisco_workflow import workflow
import h5py
import modisco.util
from collections import Counter
from modisco.visualization import viz_sequence
import modisco.affinitymat.core
import modisco.cluster.phenograph.core
import modisco.cluster.phenograph.cluster
import modisco.cluster.core
import modisco.aggregator
import optparse

parser = optparse.OptionParser()
parser.add_option('--target',
    action="store", dest="target",
    help="target", default=None)
options, args = parser.parse_args()
target = options.target

if not os.path.exists("comparison_figs/modisco_reports/"+target):
    os.makedirs("comparison_figs/modisco_reports/"+target)

CWMs = []
meta_data = []
filename = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/modisco-lite/"+target+"/modisco_counts_results.h5"
f = h5py.File(filename, 'r')
pattern_list = len(f['pos_patterns'])

def trim_motif(cwm_fwd):
    trim_threshold=0.3
    score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
    trim_thresh_fwd = np.max(score_fwd) * trim_threshold
    pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
    start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1)
    trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
    return trimmed_cwm_fwd

for idx in range(pattern_list):
    cwm = trim_motif(f['pos_patterns']['pattern_'+str(idx)]['contrib_scores'])
    meta_data.append("counts task,\n"+"pattern_"+str(idx)+", "+\
                     str(list(f['pos_patterns']['pattern_'+str(idx)]['seqlets']['n_seqlets'])[0])+" seqlets")
    CWMs.append(cwm)

filename = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/modisco-lite/"+target+"/modisco_profile_results.h5"
f = h5py.File(filename, 'r')
pattern_list = len(f['pos_patterns'])

for idx in range(pattern_list):
    cwm = trim_motif(f['pos_patterns']['pattern_'+str(idx)]['contrib_scores'])
    meta_data.append("profile task,\n"+"pattern_"+str(idx)+\
                     ", "+str(list(f['pos_patterns']['pattern_'+str(idx)]['seqlets']['n_seqlets'])[0])+" seqlets")
    CWMs.append(cwm)

font = {'weight' : 'bold', 'size' : 6}
matplotlib.rc('font', **font)
num_rows = 6
num_cols = 3
for i in range(0,len(CWMs),(num_rows*num_cols)):
    fig = plt.figure(figsize=(6,10), dpi=300)
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    for idx in range(i,i+(num_rows*num_cols)):
        if idx >= len(CWMs): break
        ax = fig.add_subplot(num_rows,num_cols,(idx%(num_rows*num_cols))+1)
        ax.set_title(meta_data[idx])
        viz_sequence.plot_weights_given_ax(ax, CWMs[idx],
                                           height_padding_factor=0.2,
                                           length_padding=1.0,
                                           subticks_frequency=1.0,
                                           highlight={})
        ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("comparison_figs/modisco_reports/"+target+"/fig_"+str(int(idx/(num_rows*num_cols)))+'.png', dpi=300)
    fig.clf()
print("DONE")