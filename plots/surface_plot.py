import matplotlib.pyplot as plt
from nilearn import datasets, plotting
import numpy as np
from nibabel.freesurfer.io import read_geometry
# Your 2×20484 array:
#   dd["corrs_t"][0] → first map
#   dd["corrs_t"][1] → second map
fold = 2
session = 1
subject = 2
model_name = "GPT-2"
layer = 10
seq_len = 20
task_name = "LucyStory"

ses_name = "ses-" + str(session).zfill(3)
data_dir = f"../../training_saves/{task_name}/{model_name}/training_results/"
fname = f"predict_sub-0{subject}_ses-{session:03d}_with_{model_name}_layer_{layer}_len_{seq_len}_folds_{fold}.npy"

dd = np.load(data_dir+fname, allow_pickle=True).item()
print(dd["corrs_t"].shape)
surf_data = dd["corrs_t"]

# 1. Fetch the fsaverage5 meshes
fs = datasets.fetch_surf_fsaverage('fsaverage5')

# 2. Decide which of the two maps to plot
#    Here we plot the first one; change the index to 1 for the second.

# 3. Split into left & right hemisphere arrays
texture_left  = surf_data[0, :]
texture_right = surf_data[1, :]


# 4. Plot static surface maps

# 4a. Left hemisphere
fig = plotting.plot_surf_stat_map(
    surf_mesh=fs.infl_left,      # left inflated mesh
    stat_map=texture_left,       # values on left hemi
    hemi='left',
    title=f'{task_name} {ses_name} — Left Hemi',
    colorbar=True,
    bg_map=fs.sulc_left,         # sulcal depth background
    vmax=0.55,
    cmap='cold_hot'
)

# 4b. Right hemisphere
fig = plotting.plot_surf_stat_map(
    surf_mesh=fs.infl_right,     # right inflated mesh
    stat_map=texture_right,      # values on right hemi
    hemi='right',
    title=f'{task_name} {ses_name} — Right Hemi',
    colorbar=True,
    bg_map=fs.sulc_right,
    vmax=0.55,
    cmap='cold_hot'
)

# 5. Tweak tick label sizes if you like
for ax in fig.axes:
    ax.tick_params(labelsize=20)

plt.show()
fig.savefig(data_dir+"/plots"+f"/volumemap_{model_name}_{task_name}_sub-{subject:02d}_fold_{fold}.png")

