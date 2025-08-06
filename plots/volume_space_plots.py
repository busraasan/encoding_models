from pathlib import Path
from nilearn import plotting, datasets
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import nibabel as nib
import numpy as np

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

mni_template_2mm = datasets.load_mni152_template(resolution=2)

fig = plt.figure(figsize=(14, 6))          # width × height in inches

fs = datasets.fetch_surf_fsaverage('fsaverage5')

# 3. Plot static surface maps
#    Left hemisphere
fig = plotting.plot_surf_stat_map(
    fs.infl_left,               # inflated mesh for left hemi
    texture_left,
    hemi='left',
    title=f'{task_name} {ses_name} — Left Hemisphere',
    colorbar=True,
    bg_map=fs.sulc_left,        # sulcal depth background
    vmax=0.55,
    cmap='cold_hot'
)

#    Right hemisphere
fig = plotting.plot_surf_stat_map(
    fs.infl_right,              # inflated mesh for right hemi
    texture_right,
    hemi='right',
    title=f'{task_name} {ses_name} — Right Hemisphere',
    colorbar=True,
    bg_map=fs.sulc_right,
    vmax=0.55,
    cmap='cold_hot'
)

# 4. (Optional) Adjust tick labels size
for ax in fig.axes:
    ax.tick_params(labelsize=20)

plt.show()

# view_brain = plotting.plot_stat_map(
#     data_dir+fname,
#     bg_img=mni_template_2mm,
#     display_mode='z',
#     cut_coords=[-5, 10, 23, 35],  
#     title=f'Between Subjects / Corr Map for {task_name} {ses_name} space MNI',
#     figure=fig,
#     colorbar=True,  # Add colorbar to the plot
#     vmax=0.55,  # Set maximum value for color scaling

# )

# for ax in fig.axes:
#     ax.tick_params(labelsize=20)


# view_html = plotting.view_img(
#     data_dir+fname,
#     bg_img=mni_template_2mm,
#     title=f'Between Subjects / Corr Map for {task_name} {ses_name} space MNI',
#     vmax=0.55,  # Set maximum value for color scaling
# )

fig.savefig(data_dir+"/plots"+f"/volumemap_{model_name}_{task_name}_sub-{subject:02d}_fold_{fold}.png")
# view_brain.close()
# view_html.save_as_html(data_dir+"/plots"+f"/volumemap_{model_name}_{task_name}_sub-{subject:02d}_fold_{fold}.html")
