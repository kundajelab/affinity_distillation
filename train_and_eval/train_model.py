import keras_genomics
from keras_genomics.layers.convolutional import RevCompConv1D
import keras
import keras.layers as kl
from keras.utils import CustomObjectScope
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from seqdataloader.batchproducers import coordbased
from seqdataloader.batchproducers.coordbased import coordstovals
from seqdataloader.batchproducers.coordbased import coordbatchproducers
from seqdataloader.batchproducers.coordbased import coordbatchtransformers
from scipy.stats import spearmanr, pearsonr, gaussian_kde
import os
import json
import optparse

font = {'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)

parser = optparse.OptionParser()
parser.add_option('--gpus',
    action="store", dest="gpus",
    help="which gpus", default=None)
parser.add_option('--target',
    action="store", dest="target",
    help="what is the target", default=None)
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

#If we want to avoid zero-padding, then the input seq len will be determined
# by parameters of the convolutions
class AbstractProfileModel(object):
    
    def get_output_profile_len(self):
        raise NotImplementedError()
  
    def get_model(self):
        raise NotImplementedError()

def trim_flanks_of_conv_layer(conv_layer, output_len, width_to_trim, filters):
    layer = keras.layers.Lambda(
        lambda x: x[:,
          int(0.5*(width_to_trim)):-(width_to_trim-int(0.5*(width_to_trim)))],
        output_shape=(output_len, filters))(conv_layer)
    return layer
        
#model architecture is based on 
#https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/basepair/models.py#L534
#The non-cli parameters are specified in:
# https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/src/chipnexus/train/seqmodel/joint-model-valid.gin
#The cli parameters are in line 165 of:
# https://docs.google.com/spreadsheets/d/1n3l2HXKSNpmNUOifD41uRzDEAgmOqXMQDxquRaz6WLg/edit#gid=0
# which seems to match https://github.com/kundajelab/basepair/blob/cda0875571066343cdf90aed031f7c51714d991a/src/chipnexus/train/seqmodel/ChIP-seq-default.gin
class RcBPnetArch(AbstractProfileModel):   

    def __init__(self, input_seq_len, c_task_weight, filters,
                       n_dil_layers, conv1_kernel_size,
                       dil_kernel_size,
                       outconv_kernel_size, lr):
        self.input_seq_len = input_seq_len
        self.c_task_weight = c_task_weight
        self.filters = filters
        self.n_dil_layers = n_dil_layers
        self.conv1_kernel_size = conv1_kernel_size
        self.dil_kernel_size = dil_kernel_size
        self.outconv_kernel_size = outconv_kernel_size
        self.lr = lr
    
    def get_embedding_len(self):
        embedding_len = self.input_seq_len
        embedding_len -= (self.conv1_kernel_size-1)     
        for i in range(1, self.n_dil_layers+1):
            dilation_rate = (2**i)
            embedding_len -= dilation_rate*(self.dil_kernel_size-1)
        return embedding_len
    
    def get_output_profile_len(self):
        embedding_len = self.get_embedding_len()
        out_profile_len = embedding_len - (self.outconv_kernel_size - 1)
        return out_profile_len
    
    def get_keras_model(self):
      
        out_pred_len = self.get_output_profile_len()
        
        inp = kl.Input(shape=(self.input_seq_len, 4), name='sequence')
        first_conv = RevCompConv1D(filters=self.filters,
                               kernel_size=self.conv1_kernel_size,
                               padding='valid',
                               activation='relu')(inp)
        curr_layer_size = self.input_seq_len - (self.conv1_kernel_size-1)
        bias_counts_input = kl.Input(shape=(1,), name="control_logcount")
        bias_profile_input = kl.Input(shape=(out_pred_len, 2),
                                      name="control_profile")
        prev_layers = [first_conv]
        for i in range(1, self.n_dil_layers + 1):
            dilation_rate = 2**i
            if i == 1:
                prev_sum = first_conv
            else:
                print(prev_layers)
                prev_sum = kl.merge.Average()(prev_layers)
            conv_output = RevCompConv1D(filters=self.filters,
                                  kernel_size=self.dil_kernel_size,
                                  padding='valid',
                                  activation='relu',
                                  dilation_rate=dilation_rate)(prev_sum)          
            width_to_trim = dilation_rate*(self.dil_kernel_size-1)
            curr_layer_size = (curr_layer_size - width_to_trim)
            prev_layers = [trim_flanks_of_conv_layer(
              conv_layer=x, output_len=curr_layer_size,
              width_to_trim=width_to_trim, filters=2*self.filters)
              for x in prev_layers]
            prev_layers.append(conv_output)

        combined_conv = kl.merge.Average()(prev_layers)

        #Counts prediction
        gap_combined_conv = kl.GlobalAvgPool1D()(combined_conv)
        count_out = kl.Reshape((-1,), name="task0_logcount")(
            RevCompConv1D(filters=1, kernel_size=1)(
              kl.Reshape((1,-1))(kl.concatenate([
                  #concatenation of the bias layer both before and after
                  # is needed for rc symmetry
                  kl.Lambda(lambda x: x[:, ::-1])(bias_counts_input),
                  gap_combined_conv,
                  bias_counts_input], axis=-1))))

        profile_out_prebias = RevCompConv1D(
                               filters=1,
                               kernel_size=self.outconv_kernel_size,
                               padding='valid')(combined_conv)
        profile_out = RevCompConv1D(
            filters=1, kernel_size=1, name="task0_profile")(
                    kl.concatenate([
                        #concatenation of the bias layer both before and after
                        # is needed for rc symmetry
                        kl.Lambda(lambda x: x[:, :, ::-1])(bias_profile_input),
                        profile_out_prebias,
                        bias_profile_input], axis=-1))

        model = keras.models.Model(
          inputs=[inp, bias_counts_input, bias_profile_input],
          outputs=[count_out, profile_out])
        model.compile(keras.optimizers.Adam(lr=self.lr),
                      loss=['mse', MultichannelMultinomialNLL(2)],
                      loss_weights=[self.c_task_weight, 1])
        return model


params_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/params/"
params = json.load(open(params_dir+options.params))
seq_len = params['seq_len']
modelwrapper = RcBPnetArch(
    input_seq_len=seq_len, c_task_weight=params['c_task_weight'],
    filters=params['filters'], n_dil_layers=params['n_dil_layers'],
    conv1_kernel_size=params['conv1_kernel_size'],
    dil_kernel_size=params['dil_kernel_size'],
    outconv_kernel_size=params['outconv_kernel_size'],
    lr=params['lr'])
out_pred_len = modelwrapper.get_output_profile_len()
print(out_pred_len, seq_len-out_pred_len)

lite_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"
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

keras_train_batch_generator = coordbased.core.KerasBatchGenerator(
  coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
      bed_file=lite_dir+target+"/train_1k_around_summits.bed.gz",
      coord_batch_transformer=
          coordbatchtransformers.UniformJitter(
              maxshift=200, chromsizes_file=params['genome_sizes']),
      batch_size=128,
      shuffle_before_epoch=True, 
      seed=1234),
  inputs_coordstovals=inputs_coordstovals,
  targets_coordstovals=targets_coordstovals)

keras_valid_batch_generator = coordbased.core.KerasBatchGenerator(
  coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=lite_dir+target+"/valid_1k_around_summits.bed.gz",
            batch_size=128,
            shuffle_before_epoch=False, 
            seed=1234),
  inputs_coordstovals=inputs_coordstovals,
  targets_coordstovals=targets_coordstovals)

thebatch = keras_train_batch_generator[0]
for tupleidx,tupleentry in enumerate(thebatch):
    print("Tuple entry",tupleidx)
    for key in tupleentry:
        print(key, tupleentry[key].shape)

models_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/example_models/"

model = modelwrapper.get_keras_model()
print(model.summary())
early_stopping_callback = keras.callbacks.EarlyStopping(
                            patience=10, restore_best_weights=True)
loss_history = model.fit_generator(keras_train_batch_generator,
                    epochs=500,
                    validation_data=keras_valid_batch_generator,
                    callbacks=[early_stopping_callback])
model.set_weights(early_stopping_callback.best_weights)
model.save(models_dir+target+'.h5')

# with CustomObjectScope({'MultichannelMultinomialNLL': MultichannelMultinomialNLL,'RevCompConv1D': RevCompConv1D}):
#     model = keras.models.load_model(models_dir+target+'.h5')

keras_test_batch_generator = coordbased.core.KerasBatchGenerator(
  coordsbatch_producer=coordbatchproducers.SimpleCoordsBatchProducer(
            bed_file=lite_dir+target+"/test_1k_around_summits.bed.gz",
            batch_size=128,
            shuffle_before_epoch=False, 
            seed=1234),
  inputs_coordstovals=inputs_coordstovals,
  targets_coordstovals=targets_coordstovals)

test_preds_logcount = []
test_biastrack_logcount = []
test_biastrack_profile = []
test_seqs = []
test_preds_profile = []
test_labels_logcount = []
test_labels_profile = []
for batch_idx in range(len(keras_test_batch_generator)):
    batch_inputs, batch_labels = keras_test_batch_generator[batch_idx]
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
test_preds_logcount = np.concatenate(test_preds_logcount, axis=0)
test_preds_profile = np.concatenate(test_preds_profile, axis=0)
test_labels_logcount = np.concatenate(test_labels_logcount, axis=0)
test_labels_profile = np.concatenate(test_labels_profile, axis=0)
test_labels_logtotalcount = np.log(np.sum(np.exp(test_labels_logcount) - 1,axis=-1) + 1)

plt.figure()
xvals = test_biastrack_logcount
yvals = test_labels_logtotalcount
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.2)
min_lim = min(np.min(xvals), np.min(yvals))
max_lim = max(np.max(xvals), np.max(yvals))
plt.xlim(min_lim-0.5, max_lim+0.5)
plt.ylim(min_lim-0.5, max_lim+0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Bias track log counts")
plt.ylabel("True log total counts")
plt.title(str(spearmanr(xvals, yvals)[0])+", "+
          str(pearsonr(xvals, yvals)[0]))
plt.plot([min_lim-0.5, max_lim+0.5], [min_lim-0.5, max_lim+0.5], color="black")
plt.savefig('figs/'+target+'_test_bias.png', dpi=300, format='png')
plt.clf()

#do a scatterplot of total count predictions
plt.figure()
xvals = test_preds_logcount[:,0]
yvals = test_labels_logcount[:,0]
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.2)
min_lim = min(np.min(xvals), np.min(yvals))
max_lim = max(np.max(xvals), np.max(yvals))
plt.xlim(min_lim-0.5, max_lim+0.5)
plt.ylim(min_lim-0.5, max_lim+0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted log counts - Forward Strand")
plt.ylabel("True log counts - Forward Strand")
plt.title(str(spearmanr(xvals, yvals)[0])+", "+
          str(pearsonr(xvals, yvals)[0]))
plt.plot([min_lim-0.5, max_lim+0.5], [min_lim-0.5, max_lim+0.5], color="black")
plt.savefig('figs/'+target+'_test_fwd.png', dpi=300, format='png')
plt.clf()

plt.figure()
xvals = test_preds_logcount[:,1]
yvals = test_labels_logcount[:,1]
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.2)
min_lim = min(np.min(xvals), np.min(yvals))
max_lim = max(np.max(xvals), np.max(yvals))
plt.xlim(min_lim-0.5, max_lim+0.5)
plt.ylim(min_lim-0.5, max_lim+0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted log counts - Reverse Strand")
plt.ylabel("True log counts - Reverse Strand")
plt.title(str(spearmanr(xvals, yvals)[0])+", "+
          str(pearsonr(xvals, yvals)[0]))
plt.plot([min_lim-0.5, max_lim+0.5], [min_lim-0.5, max_lim+0.5], color="black")
plt.savefig('figs/'+target+'_test_rev.png', dpi=300, format='png')
plt.clf()

o_train_preds_logcount = []
o_train_biastrack_logcount = []
o_train_biastrack_profile = []
o_train_seqs = []
o_train_preds_profile = []
o_train_labels_logcount = []
o_train_labels_profile = []

orig_seqs = []

for batch_idx in range(len(keras_train_batch_generator)):
    batch_inputs, batch_labels = keras_train_batch_generator[batch_idx]
    o_train_seqs.append(batch_inputs['sequence']) 
    o_train_biastrack_logcount.append(batch_inputs['control_logcount'])
    o_train_biastrack_profile.append(batch_inputs['control_profile'])    
    train_preds = model.predict(batch_inputs)
    o_train_preds_logcount.append(train_preds[0])
    o_train_preds_profile.append(train_preds[1])
    o_train_labels_logcount.append(batch_labels['task0_logcount'])
    o_train_labels_profile.append(batch_labels['task0_profile'])
o_train_biastrack_logcount = np.concatenate(o_train_biastrack_logcount, axis=0)
o_train_biastrack_profile = np.concatenate(o_train_biastrack_profile,axis=0)
o_train_seqs = np.concatenate(o_train_seqs,axis=0)
o_train_preds_logcount = np.concatenate(o_train_preds_logcount, axis=0)
o_train_preds_profile = np.concatenate(o_train_preds_profile, axis=0)
o_train_labels_logcount = np.concatenate(o_train_labels_logcount, axis=0)
o_train_labels_profile = np.concatenate(o_train_labels_profile, axis=0)

o_train_labels_logtotalcount = np.log(np.sum(np.exp(o_train_labels_logcount) - 1,axis=-1) + 1)

plt.figure()
xvals = o_train_biastrack_logcount
yvals = o_train_labels_logtotalcount
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.2)
min_lim = min(np.min(xvals), np.min(yvals))
max_lim = max(np.max(xvals), np.max(yvals))
plt.xlim(min_lim-0.5, max_lim+0.5)
plt.ylim(min_lim-0.5, max_lim+0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Bias track log counts")
plt.ylabel("True log total counts")
plt.title(str(spearmanr(xvals, yvals)[0])+", "+
          str(pearsonr(xvals, yvals)[0]))
plt.plot([min_lim-0.5, max_lim+0.5], [min_lim-0.5, max_lim+0.5], color="black")
plt.savefig('figs/'+target+'_train_bias.png', dpi=300, format='png')
plt.clf()

#do a scatterplot of total count predictions
plt.figure()
xvals = o_train_preds_logcount[:,0]
yvals = o_train_labels_logcount[:,0]
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.2)
min_lim = min(np.min(xvals), np.min(yvals))
max_lim = max(np.max(xvals), np.max(yvals))
plt.xlim(min_lim-0.5, max_lim+0.5)
plt.ylim(min_lim-0.5, max_lim+0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted log counts - Forward Strand")
plt.ylabel("True log counts - Forward Strand")
plt.title(str(spearmanr(xvals, yvals)[0])+", "+
          str(pearsonr(xvals, yvals)[0]))
plt.plot([min_lim-0.5, max_lim+0.5], [min_lim-0.5, max_lim+0.5], color="black")
plt.savefig('figs/'+target+'_train_fwd.png', dpi=300, format='png')
plt.clf()

plt.figure()
xvals = o_train_preds_logcount[:,1]
yvals = o_train_labels_logcount[:,1]
xy = np.vstack([xvals,yvals])
z = gaussian_kde(xy)(xy)
plt.scatter(xvals, yvals,  c=z, edgecolor='', alpha=0.2)
min_lim = min(np.min(xvals), np.min(yvals))
max_lim = max(np.max(xvals), np.max(yvals))
plt.xlim(min_lim-0.5, max_lim+0.5)
plt.ylim(min_lim-0.5, max_lim+0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel("Predicted log counts - Reverse Strand")
plt.ylabel("True log counts - Reverse Strand")
plt.title(str(spearmanr(xvals, yvals)[0])+", "+
          str(pearsonr(xvals, yvals)[0]))
plt.plot([min_lim-0.5, max_lim+0.5], [min_lim-0.5, max_lim+0.5], color="black")
plt.savefig('figs/'+target+'_train_rev.png', dpi=300, format='png')
plt.clf()

sorted_test_indices = [x[0] for x in 
                       sorted(enumerate(test_labels_logtotalcount[:500]),
                              key=lambda x: -x[1])]

def smooth(vals):
    return np.convolve(vals, np.ones(1,)/1, mode='same')

for idx in sorted_test_indices[:5]:
    true_profile = test_labels_profile[idx] 
    print("idx",idx)
    print("Counts",np.sum(true_profile,axis=0))
    print("Predcounts",np.exp(test_preds_logcount[idx])-1)
    for oneovertemp in [1.0]:
        print("oneovertemp",oneovertemp)
        print(test_labels_profile[idx].shape)
        print("Pred profile shape", test_preds_profile[idx].shape)
        pred_profile = (np.sum(test_labels_profile[idx], axis=0)[None,:] #total counts
                      *(np.exp(test_preds_profile[idx]*oneovertemp)/
                        np.sum(np.exp(test_preds_profile[idx]*oneovertemp),axis=0)[None,:]) )   
        plt.figure(figsize=(20,3))
        start_view = 0
        end_view = seq_len
        total_flanking = seq_len - out_pred_len
        left_flank = int(0.5*total_flanking)
        right_flank = total_flanking - left_flank
        plt.plot(np.arange(out_pred_len)+left_flank, smooth(true_profile[:,0]), alpha=0.3, label="Obs. pos.")
        plt.plot(np.arange(out_pred_len)+left_flank, -smooth(true_profile[:,1]), alpha=0.3, label="Obs. neg.")
        plt.plot(np.arange(out_pred_len)+left_flank, pred_profile[:,0], label="Pred. pos.")
        plt.plot(np.arange(out_pred_len)+left_flank, -pred_profile[:,1], label="Pred. neg.")
        plt.xlim(start_view,end_view)
        plt.legend(loc="upper right")
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.savefig('figs/'+target+str(idx)+'.png', dpi=300, format='png')
        plt.clf()