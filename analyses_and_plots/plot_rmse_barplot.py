import os
import json
import optparse
import numpy as np
import matplotlib.pyplot as plt

parser = optparse.OptionParser()
parser.add_option('--target',
    action="store", dest="target",
    help="target", default=None)
options, args = parser.parse_args()

target = options.target

results_dir = "comparison_figs/"+target+"/calibrated/"
w = 0.4    # bar width
step = 0.7
x = [step, 2*step, 3*step, 4*step]     # x-coordinates of bars

criteria = "rmse"
meta = {}
weeder = []
weeder_x = []
meta["best_weeder"] = 10^5
for filename in os.listdir(results_dir+"weeder/"):
    if "weeder_max_" in filename and filename.endswith("metadata.json"):
        data = json.load(open(results_dir+"weeder/"+filename))
        weeder.append(float(data[criteria]))
        weeder_x.append(step + np.random.random() * (w/2) - (w/4))   # distribute coords randomly across half width of bar
        if weeder[-1] <= meta["best_weeder"]:
            meta["best_weeder"] = weeder[-1]
            meta["best_weeder_name"] = filename
        if filename == "weeder_max_0_metadata.json":
            top_weeder_x = weeder_x[-1]
            top_weeder_y = weeder[-1]
meta["mean_weeder"] = np.mean(weeder)
meta["num_weeder_motifs"] = len(weeder)

streme = []
streme_x = []
meta["best_streme"] = 10^5
for filename in os.listdir(results_dir+"streme/"):
    if "streme_max_" in filename and filename.endswith("metadata.json"):
        data = json.load(open(results_dir+"streme/"+filename))
        streme.append(float(data[criteria]))
        streme_x.append((2*step) + np.random.random() * (w/2) - (w/4))
        if streme[-1] <= meta["best_streme"]:
            meta["best_streme"] = streme[-1]
            meta["best_streme_name"] = filename
        if filename == "streme_max_0_metadata.json":
            top_streme_x = streme_x[-1]
            top_streme_y = streme[-1]
meta["mean_streme"] = np.mean(streme)
meta["num_streme_motifs"] = len(streme)

modisco = []
modisco_x = []
meta["best_modisco"] = 10^5
for filename in os.listdir(results_dir+"modisco_lite/"):
    if "modisco_max_" in filename and filename.endswith("metadata.json"):
        data = json.load(open(results_dir+"modisco_lite/"+filename))
        modisco.append(float(data[criteria]))
        modisco_x.append((3*step) + np.random.random() * (w/2) - (w/4))
        if modisco[-1] <= meta["best_modisco"]:
            meta["best_modisco"] = modisco[-1]
            meta["best_modisco_name"] = filename
        if filename == "modisco_max_0_metadata.json":
            top_modisco_x = modisco_x[-1]
            top_modisco_y = modisco[-1]
meta["mean_modisco"] = np.mean(modisco)
meta["num_modisco_motifs"] = len(modisco)

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
    ax.scatter(coords_x[i], y[i], color='dimgrey', alpha=0.8)
ax.scatter(top_x, top_y, color='red', label="top")
with open(results_dir+criteria+'.json', 'w') as fp: json.dump(meta, fp)
fig.savefig(results_dir+criteria+'.png', dpi=300, format='png')