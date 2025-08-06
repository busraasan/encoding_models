import os
import numpy as np
import matplotlib.pyplot as plt

# 1) Define what folds and sessions you have
folds    = [5]
sessions = [1, 3, 5]
subject = 2
model_name = "GPT-2"
layer = 8
seq_len = 20
task_name = "LucyStory"

# 2) Prepare an array to hold the mean correlation for each combination
heatmap_data = np.zeros((len(folds), len(sessions)))

# 3) Loop through and load each file
data_dir = f"../../training_saves/{task_name}/{model_name}/training_results/"

for i, f in enumerate(folds):
    for j, sess in enumerate(sessions):
        fname = f"predict_sub-0{subject}_ses-{sess:03d}_with_{model_name}_layer_{layer}_len_{seq_len}_folds_{f}.npy"
        fullpath = os.path.join(data_dir, fname)
        try:
            dd = np.load(fullpath, allow_pickle=True).item()
        except FileNotFoundError:
            print(f"Missing file: {fullpath}")
            heatmap_data[i, j] = np.nan
            continue

        heatmap_data[i, j] = np.nanmean(dd['corrs_t'])

# 4) Compute row-means (avg per fold) and col-means (avg per session), if you like
row_means = np.nanmean(heatmap_data, axis=1)
col_means = np.nanmean(heatmap_data, axis=0)

# 5) Plot the heatmap
fig, ax = plt.subplots(figsize=(7, 4))  # a bit wider to fit the extra info
im = ax.imshow(heatmap_data, 
               aspect='auto', 
               cmap='inferno', 
               vmin=np.nanmin(heatmap_data), 
               vmax=np.nanmax(heatmap_data))

# set tick labels *including* the per-fold average
ax.set_xticks(np.arange(len(sessions)))
ax.set_yticks(np.arange(len(folds)))
ax.set_xticklabels([f"ses-{s:03d}" for s in sessions])
ax.set_yticklabels([
    f"n_fold={f}\n(avg={row_means[i]:.2f})"
    for i, f in enumerate(folds)
])

plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# annotate each cell
for i in range(len(folds)):
    for j in range(len(sessions)):
        ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                ha="center", va="center", color="w")

# # optionally, draw the overall session means *above* the heatmap
# for j, cm in enumerate(col_means):
#     ax.text(j, -0.3, f"{cm:.2f}", ha="center", va="bottom", color="k")

# adjust & save
ax.set_title("Mean corr ← folds × sessions for BERT layer=10, seq_len=20")
cbar = fig.colorbar(im, ax=ax, label="Mean corr")
plt.tight_layout()

# save
outdir = os.path.join(data_dir, "plots")
os.makedirs(outdir, exist_ok=True)
png_name = f"heatmap_{model_name}_{task_name}_sub-{subject:02d}.png"
plt.savefig(os.path.join(outdir, png_name))
plt.show()
