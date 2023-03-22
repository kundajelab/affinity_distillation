import os
import subprocess
import shutil

scores_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/imp-scores-lite/"
output_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/models/modisco-lite/"
meme_file_dir = "/oak/stanford/groups/akundaje/amr1/pho4_final/lite_data/in-vivo/"
shap_counts_file = "/post_counts_actualimps.npy"
shap_profile_file = "/post_profile_actualimps.npy"
seqs_file = "/seqs.npy"
meme_file = "/streme_out/streme.txt"

for subdir, dirs, files in os.walk(scores_dir):
    for target in dirs:
        print("working on ", target)
        if not os.path.exists(output_dir+target):
            os.makedirs(output_dir+target)
        subprocess.run(["modisco", "motifs", "-s", scores_dir+target+seqs_file,
                        "-a", scores_dir+target+shap_counts_file, "-n", "20000",
                        "-o", output_dir+target+"/modisco_counts_results.h5"])
        subprocess.run(["modisco", "motifs", "-s", scores_dir+target+seqs_file,
                        "-a", scores_dir+target+shap_profile_file, "-n", "20000",
                        "-o", output_dir+target+"/modisco_profile_results.h5"])
        print("generating reports...")
        subprocess.run(["modisco", "report", "-i", output_dir+target+"/modisco_counts_results.h5",
                        "-o", output_dir+target+"/counts_report/", "-s", output_dir+target+"/counts_report/",
                        "-m", meme_file_dir+target+meme_file])
        subprocess.run(["modisco", "report", "-i", output_dir+target+"/modisco_profile_results.h5",
                        "-o", output_dir+target+"/profile_report/", "-s", output_dir+target+"/profile_report/",
                        "-m", meme_file_dir+target+meme_file])
        print("done with ", target)