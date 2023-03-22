import keras_genomics
from keras_genomics.layers.convolutional import RevCompConv1D
import keras
import keras.layers as kl
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from keras.models import load_model
from keras.utils import CustomObjectScope
from deeplift.dinuc_shuffle import dinuc_shuffle
import pandas as pd
import h5py
import json
import codecs
import os
import gzip
from math import log
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import optparse

parser = optparse.OptionParser()

parser.add_option('--target',
    action="store", dest="target",
    help="target", default=None)
parser.add_option('--gpus',
    action="store", dest="gpus",
    help="gpus", default=None)
parser.add_option('--params',
    action="store", dest="params",
    help="which NN params", default=None)
options, args = parser.parse_args()

library = {
    "ets1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log_normalized.xlsx",
    "elk1": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log_normalized.xlsx",
    "gabpa": "GSE97793_Combined_ets1_100nM_elk1_100nM_50nM_gabpa_100nM_log_normalized.xlsx",
    "e2f1": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log_normalized.xlsx",
    "e2f3": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log_normalized.xlsx",
    "e2f4": "GSE97886_Combined_E2f1_200nM_250nM_E2f3_250nM_E2f4_500nM_800nM_log_normalized.xlsx",
    "max": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log_normalized.xlsx",
    "mxi": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log_normalized.xlsx",
    "myc": "GSE97885_Combined_Max_Myc_Mad_Mad_r_log_normalized.xlsx",
    "runx1": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log_normalized.xlsx",
    "runx2": "GSE97691_Combined_Runx1_10nM_50nM_Runx2_10nM_50nM_log_normalized.xlsx"
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

target = options.target
model_name = target
if "noCtl" in target:
    target = target.replace('_noCtl', '')
if target[0].isdigit():
    key = target.split('_')[1].lower()
    if key[-1] == '-': key = key[:-1]
else: key = target.split('_')[0]
data_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vitro/gcPBM/"
dfs = pd.read_excel(data_dir+library[key])
all_xvals = dfs[column[key]]
seqs = dfs['Sequence']

os.environ["CUDA_VISIBLE_DEVICES"]=options.gpus
def multinomial_nll(true_counts, logits):
    """Compute the multinomial negative log-likelihood
    Args:
      true_counts: observed count values
      logits: predicted logit values
    """
    counts_per_example = tf.reduce_sum(true_counts, axis=-1)
    dist = tfp.distributions.Multinomial(total_count=counts_per_example,
                                         logits=logits)
    return (-tf.reduce_sum(dist.log_prob(true_counts)) / 
            tf.to_float(tf.shape(true_counts)[0]))

#from https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/losses.py#L87
class MultichannelMultinomialNLL(object):
    def __init__(self, n):
        self.__name__ = "MultichannelMultinomialNLL"
        self.n = n

    def __call__(self, true_counts, logits):
        for i in range(self.n):
            loss = multinomial_nll(true_counts[..., i], logits[..., i])
            if i == 0:
                total = loss
            else:
                total += loss
        return total

    def get_config(self):
        return {"n": self.n}

models_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/example_models/"
with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
    model = load_model(models_dir+model_name+'.h5')

params_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/params/"
params = json.load(open(params_dir+options.params))
fastapath = params['genome_fasta']
GenomeDict={}
sequence=''
inputdatafile = open(fastapath)
for line in inputdatafile:
    if line[0]=='>':
        if sequence != '':
            GenomeDict[chrm] = ''.join(sequence)
        chrm = line.strip().split('>')[1]
        sequence=[]
        Keep=False
        continue
    else:
        sequence.append(line.strip())
GenomeDict[chrm] = ''.join(sequence)

peaks_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"
seq_len = int(params['seq_len'])
out_pred_len = int(params['out_pred_len'])
seq_peaks = []
with gzip.open(peaks_dir+target+'/test_1k_around_summits.bed.gz', 'rt') as inp:
    for line in inp:
        chrm = line.strip().split('\t')[0]
        center = int((int(line.strip().split('\t')[1])+int(line.strip().split('\t')[2]))/2)
        start = center - int(seq_len/2)
        end = center + int(seq_len/2)
        candidate_seq = GenomeDict[chrm][start:end].upper()
        if len(candidate_seq) == seq_len: seq_peaks.append(candidate_seq)
    
ltrdict = {
           'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1],
           'n':[0,0,0,0],'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],
           'T':[0,0,0,1],'N':[0,0,0,0]}
def getOneHot(ISM_sequences):
  # takes in list of sequences
    one_hot_seqs = []
    for seq in ISM_sequences:
        one_hot = []
        for i in range(len(seq)):
            one_hot.append(ltrdict[seq[i:i+1]])
        one_hot_seqs.append(one_hot)
    return np.array(one_hot_seqs)

def fill_into_center(seq, insert):
    pos = int((len(seq)-len(insert))/2)
    new_seq = seq[:pos] + insert + seq[pos+len(insert):]
    return new_seq

from deeplift.dinuc_shuffle import dinuc_shuffle

num_samples = min(100, len(seq_peaks))
pred_dict = {}  # key is oligo and val is 100 preds structured ([before], [after], final)
for idx, insert in enumerate(seqs):
    if idx % 1000 == 0:
        print("Done with ", idx)
    pre_seqs = []
    post_seqs = []
    indices = np.random.choice(len(seq_peaks), num_samples, replace=False)
    for idx in indices:
        pre_seq = dinuc_shuffle(seq_peaks[idx])
        post_seq = fill_into_center(pre_seq, insert)
        pre_seqs.append(pre_seq)
        post_seqs.append(post_seq)
    if "noCtl" in model_name:
        pre = model.predict(getOneHot(pre_seqs))
        post = model.predict(getOneHot(post_seqs))
    else:
        pre = model.predict([getOneHot(pre_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
        post = model.predict([getOneHot(post_seqs), np.zeros((num_samples,)), np.zeros((num_samples,out_pred_len,2))])
    pred_dict[insert] = (pre[0].tolist(), post[0].tolist(), str(np.mean(post[0]-pre[0])))

pred_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/preds/"
json.dump(pred_dict,
          codecs.open(pred_dir+model_name+'.json', 'w', encoding='utf-8'),
          separators=(',', ':'), sort_keys=True, indent=4)

# In order to "unjsonify" the array use:
# obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
# b_new = json.loads(obj_text)
# a_new = np.array(b_new)

# xy = np.vstack([all_xvals,yvals])
# z = gaussian_kde(xy)(xy)
# smallFont = {'size' : 10}
# plt.rc('font', **smallFont)
# fig, ax = plt.subplots()
# ax.scatter(all_xvals, yvals, c=z, edgecolor='', alpha=0.5)
# plt.xlabel("Log gcPBM Signal")
# plt.ylabel("Delta Log Counts")
# plt.title("spearman: "+str(spearmanr(all_xvals, yvals)[0])+
#           ", pearson: "+ str(pearsonr(all_xvals, yvals)[0]))
# fig.savefig('preds/'+target+'_eval.png', dpi=fig.dpi)
