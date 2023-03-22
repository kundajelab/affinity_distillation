from __future__ import division, print_function, absolute_import
from collections import namedtuple
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
import shap
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

target = options.target
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
    model = load_model(models_dir+target+'.h5')

params_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/params/"
lite_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"
params = json.load(open(params_dir+options.params))
seq_len = int(params['seq_len'])
out_pred_len = int(params['out_pred_len'])
inputs_coordstovals = coordstovals.core.CoordsToValsJoiner(
    coordstovals_list=[
      coordbased.coordstovals.fasta.PyfaidxCoordsToVals(
        genome_fasta_path=params['genome_fasta'],
        mode_name="sequence",
        center_size_to_use=seq_len),
      coordstovals.bigwig.PosAndNegSmoothWindowCollapsedLogCounts(
        pos_strand_bigwig_path=lite_dir+target+"/ctl/control_pos_strand.bw",
        neg_strand_bigwig_path=lite_dir+target+"/ctl/control_neg_strand.bw",
        counts_mode_name="control_logcount",
        profile_mode_name="control_profile",
        center_size_to_use=out_pred_len,
        smoothing_windows=[1,50])])

targets_coordstovals = coordstovals.bigwig.PosAndNegSeparateLogCounts(
    pos_strand_bigwig_path=lite_dir+target+"/basename_prefix.pooled.positive.bigwig",
    neg_strand_bigwig_path=lite_dir+target+"/basename_prefix.pooled.negative.bigwig",
    counts_mode_name="task0_logcount",
    profile_mode_name="task0_profile",
    center_size_to_use=out_pred_len)

Coordinates = namedtuple("Coordinates",
                         ["chrom", "start", "end", "isplusstrand"])
Coordinates.__new__.__defaults__ = (True,)


def apply_mask(tomask, mask):
    if isinstance(tomask, dict):
        return dict([(key, val[mask]) for key,val in tomask.items()])
    elif isinstance(tomask, list):
        return [x[mask] for x in mask]
    else:
        return x[mask]


class KerasBatchGenerator(keras.utils.Sequence):
  
    """
    Args:
        coordsbatch_producer (KerasSequenceApiCoordsBatchProducer)
        inputs_coordstovals (CoordsToVals)
        targets_coordstovals (CoordsToVals)
        sampleweights_coordstovals (CoordsToVals)
        coordsbatch_transformer (AbstracCoordBatchTransformer)
        qc_func (callable): function that can be used to filter
            out poor-quality sequences.
        sampleweights_coordstoval: either this argument or
            sampleweights_from_inputstargets could be used to
            specify sample weights. sampleweights_coordstoval
            takes a batch of coords as inputs.
        sampleweights_from_inputstargets: either this argument or
            sampleweights_coordstoval could be used to
            specify sample weights. sampleweights_from_inputstargets
            takes the inputs and targets values to generate the weights.
    """
    def __init__(self, coordsbatch_producer,
                       inputs_coordstovals,
                       targets_coordstovals,
                       coordsbatch_transformer=None,
                       qc_func=None,
                       sampleweights_coordstovals=None,
                       sampleweights_from_inputstargets=None):
        self.coordsbatch_producer = coordsbatch_producer
        self.inputs_coordstovals = inputs_coordstovals
        self.targets_coordstovals = targets_coordstovals
        self.coordsbatch_transformer = coordsbatch_transformer
        self.sampleweights_coordstovals = sampleweights_coordstovals
        self.sampleweights_from_inputstargets =\
            sampleweights_from_inputstargets
        if sampleweights_coordstovals is not None:
            assert sampleweights_from_inputstargets is None
        if sampleweights_from_inputstargets is not None:
            assert sampleweights_coordstovals is None
        self.qc_func = qc_func
 
    def __getitem__(self, index):
        coords_batch = self.coordsbatch_producer[index]
        if (self.coordsbatch_transformer is not None):
            coords_batch = self.coordsbatch_transformer(coords_batch)
        inputs = self.inputs_coordstovals(coords_batch)
        if (self.targets_coordstovals is not None):
            targets = self.targets_coordstovals(coords_batch)
        else:
            targets=None
        if (self.qc_func is not None):
            qc_mask = self.qc_func(inputs=inputs, targets=targets)
            inputs = apply_mask(tomask=inputs, mask=qc_mask)
            if (targets is not None):
                targets = apply_mask(tomask=targets, mask=qc_mask)
        else:
            qc_mask = None
        if (self.sampleweights_coordstovals is not None):
            sample_weights = self.sampleweights_coordstovals(coords_batch)
            return (coords_batch, inputs, targets, sample_weights)
        elif (self.sampleweights_from_inputstargets is not None):
            sample_weights = self.sampleweights_from_inputstargets(
                                inputs=inputs, targets=targets)
            return (coords_batch, inputs, targets, sample_weights)
        else:
            if (self.targets_coordstovals is not None):
                return (coords_batch, inputs, targets)
            else:
                return coords_batch, inputs
   
    def __len__(self):
        return len(self.coordsbatch_producer)
    
    def on_epoch_end(self):
        self.coordsbatch_producer.on_epoch_end()

keras_data_batch_generator = KerasBatchGenerator(
  coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=lite_dir+target+"/1k_around_summits.bed.gz",
            batch_size=128,
            shuffle_before_epoch=False, 
            seed=1234),
  inputs_coordstovals=inputs_coordstovals,
  targets_coordstovals=targets_coordstovals)

def combine_mult_and_diffref(mult, orig_inp, bg_data):
    to_return = []
    for l in [0]:
        projected_hypothetical_contribs = np.zeros_like(bg_data[l]).astype("float")
        assert len(orig_inp[l].shape)==2
        #At each position in the input sequence, we iterate over the one-hot encoding
        # possibilities (eg: for genomic sequence, this is ACGT i.e.
        # 1000, 0100, 0010 and 0001) and compute the hypothetical 
        # difference-from-reference in each case. We then multiply the hypothetical
        # differences-from-reference with the multipliers to get the hypothetical contributions.
        #For each of the one-hot encoding possibilities,
        # the hypothetical contributions are then summed across the ACGT axis to estimate
        # the total hypothetical contribution of each position. This per-position hypothetical
        # contribution is then assigned ("projected") onto whichever base was present in the
        # hypothetical sequence.
        #The reason this is a fast estimate of what the importance scores *would* look
        # like if different bases were present in the underlying sequence is that
        # the multipliers are computed once using the original sequence, and are not
        # computed again for each hypothetical sequence.
        for i in range(orig_inp[l].shape[-1]):
            hypothetical_input = np.zeros_like(orig_inp[l]).astype("float")
            hypothetical_input[:,i] = 1.0
            hypothetical_difference_from_reference = (hypothetical_input[None,:,:]-bg_data[l])
            hypothetical_contribs = hypothetical_difference_from_reference*mult[l]
            projected_hypothetical_contribs[:,:,i] = np.sum(hypothetical_contribs,axis=-1) 
        to_return.append(np.mean(projected_hypothetical_contribs,axis=0))
    to_return.append(np.zeros_like(orig_inp[1]))
    return to_return

def shuffle_several_times(s):
    numshuffles=20
    return [np.array([dinuc_shuffle(s[0]) for i in range(numshuffles)]),
            np.array([s[1] for i in range(numshuffles)])]

profile_model_counts_explainer = shap.explainers.deep.TFDeepExplainer(
    ([model.input[0], model.input[1]],
     tf.reduce_sum(model.outputs[0],axis=-1)),
    shuffle_several_times,
    combine_mult_and_diffref=combine_mult_and_diffref)

#See Google slide deck for explanations
#We meannorm as per section titled "Adjustments for Softmax Layers"
# in the DeepLIFT paper
meannormed_logits = (
    model.outputs[1]-
    tf.reduce_mean(model.outputs[1],axis=1)[:,None,:])
#'stop_gradient' will prevent importance from being propagated through
# this operation; we do this because we just want to treat the post-softmax
# probabilities as 'weights' on the different logits, without having the
# network explain how the probabilities themselves were derived
#Could be worth contrasting explanations derived with and without stop_gradient
# enabled...
stopgrad_meannormed_logits = tf.stop_gradient(meannormed_logits)
softmax_out = tf.nn.softmax(stopgrad_meannormed_logits,axis=1)
#Weight the logits according to the softmax probabilities, take the sum for each
# example. This mirrors what was done for the bpnet paper.
weightedsum_meannormed_logits = tf.reduce_sum(softmax_out*meannormed_logits,
                                              axis=(1,2))
profile_model_profile_explainer = shap.explainers.deep.TFDeepExplainer(
    ([model.input[0], model.input[2]],
     weightedsum_meannormed_logits),
    shuffle_several_times,
    combine_mult_and_diffref=combine_mult_and_diffref)

test_preds_logcount = []
test_biastrack_logcount = []
test_biastrack_profile = []
test_coords = []
test_seqs = []
test_preds_profile = []
test_labels_logcount = []
test_labels_profile = []
for batch_idx in range(len(keras_data_batch_generator)):
    batches, batch_inputs, batch_labels = keras_data_batch_generator[batch_idx]
    batch_coords = [str(batch) for batch in batches]
    test_coords.append(batch_coords)
    test_seqs.append(batch_inputs['sequence'])
    test_biastrack_logcount.append(batch_inputs['control_logcount'])
    test_biastrack_profile.append(batch_inputs['control_profile'])
    test_preds = model.predict(batch_inputs)
    test_preds_logcount.append(test_preds[0])
    test_preds_profile.append(test_preds[1])
    test_labels_logcount.append(batch_labels['task0_logcount'])
    test_labels_profile.append(batch_labels['task0_profile'])
test_biastrack_logcount = np.concatenate(test_biastrack_logcount, axis=0)
test_biastrack_profile = np.concatenate(test_biastrack_profile,axis=0)
test_seqs = np.concatenate(test_seqs,axis=0)
test_coords = np.concatenate(test_coords,axis=0)
test_preds_logcount = np.concatenate(test_preds_logcount, axis=0)
test_preds_profile = np.concatenate(test_preds_profile, axis=0)
test_labels_logcount = np.concatenate(test_labels_logcount, axis=0)
test_labels_profile = np.concatenate(test_labels_profile, axis=0)

#The shap scores
test_post_counts_hypimps,_ = profile_model_counts_explainer.shap_values(
    [test_seqs, np.zeros((len(test_seqs), 1))],
    progress_message=10)
test_post_profile_hypimps,_ = profile_model_profile_explainer.shap_values(
    [test_seqs, np.zeros((len(test_seqs), out_pred_len, 2))],
    progress_message=10)
test_post_counts_hypimps = np.array(test_post_counts_hypimps)
test_post_profile_hypimps = np.array(test_post_profile_hypimps)
test_post_counts_actualimps = test_post_counts_hypimps*test_seqs
test_post_profile_actualimps = test_post_profile_hypimps*test_seqs

models_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/"
if not os.path.exists(models_dir+'imp-scores/'+target):
    os.makedirs(models_dir+'imp-scores/'+target)
np.save(models_dir+'imp-scores/'+target+'/post_counts_hypimps.npy', test_post_counts_hypimps)
np.save(models_dir+'imp-scores/'+target+'/post_profile_hypimps.npy', test_post_profile_hypimps) 
np.save(models_dir+'imp-scores/'+target+'/post_counts_actualimps.npy', test_post_counts_actualimps) 
np.save(models_dir+'imp-scores/'+target+'/post_profile_actualimps.npy', test_post_profile_actualimps) 
np.save(models_dir+'imp-scores/'+target+'/labels_profile.npy', test_labels_profile) 
np.save(models_dir+'imp-scores/'+target+'/labels_logcount.npy', test_labels_logcount) 
np.save(models_dir+'imp-scores/'+target+'/preds_profile.npy', test_preds_profile) 
np.save(models_dir+'imp-scores/'+target+'/biastrack_profile.npy', test_biastrack_profile) 
np.save(models_dir+'imp-scores/'+target+'/biastrack_logcount.npy', test_biastrack_logcount) 
np.save(models_dir+'imp-scores/'+target+'/preds_logcount.npy', test_preds_logcount) 
np.save(models_dir+'imp-scores/'+target+'/seqs.npy', test_seqs) 
np.save(models_dir+'imp-scores/'+target+'/coords.npy', test_coords) 