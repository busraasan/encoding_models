import cortex
import nibabel as nib
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import re
from pathlib import Path

s = "6"
model = "GPT-2"
layer = "8"
corr_data_dir = "/BRAIN/neuromod-data/static00/training_saves/all_stories/GPT-2/training_results"
data_dir = '/BRAIN/neuromod-data/static00/try/'
method = "encoding_models_corr_map"
plotting_topic = "corr"
context_len = 20
region = "LANG_REGION"
folds = "9"
session="-1"

corr_file = f"{corr_data_dir}/predict_sub-{s.zfill(2)}_with_{model}_layer_{layer}_len_{str(context_len)}_folds_{folds}_{region}_session_{session}.npy"

corr_data = np.load(corr_file, allow_pickle=True).item()["corrs_t"]

story_dict = {
    "Tunnelpart2Story": [8,10,12],
    "Tunnelpart1Story": [8,10,12],
    "PiemanStory": [7,9,11],
    "LucyStory": [1,3,5],
    "PrettymouthStory": [8,10,12],
    "NotthefallintactStory": [2,4,6],
    "SlumlordStory": [7,9,11],
    "BlackStory": [2,4,6],
    "ForgotStory": [1,3,5]
}

story_list = list(story_dict.keys())
total_mean = []

for fold in np.arange(corr_data.shape[0]):
    total_mean.append(np.nanmean(corr_data[fold, :]))
    print(story_list[fold]+" Corr: ", np.nanmean(corr_data[fold, :]))

fold_means = np.array([np.nanmean(corr_data[f]) for f in range(corr_data.shape[0])])

# Sanity check: one label per fold
assert len(story_list) == corr_data.shape[0], "story_list length must match number of folds"

# Line plot with square markers for each story
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
x = np.arange(len(fold_means))
ax.plot(x, fold_means, marker='s', linestyle='-')  # squares mark each story
ax.set_xticks(x)
ax.set_xticklabels(story_list, rotation=45, ha='right')
ax.set_xlabel('Story (Fold)')
ax.set_ylabel('Mean correlation (NaNs ignored)')
ax.set_title('Per-story (fold) mean correlation for Subject 0'+s)
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

ymin = 0.01
ymax = 0.15
yr = ymax - ymin if ymax > ymin else max(1e-6, abs(ymax))  # avoid 0 range
ax.set_ylim(ymin - 0.05*yr, ymax + 0.18*yr)  # add headroom

# (Optional) label each point with its value
for i, y in enumerate(fold_means):
    ax.annotate(f'{y:.3f}', (x[i], y), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=10)

out_dir = Path("corr_figures")
out_dir.mkdir(parents=True, exist_ok=True)
base_name = f"per_story_mean_corr_sub-{str(s).zfill(2)}_{model}_layer-{layer}_len-{context_len}_folds-{folds}"
out_base = out_dir / base_name

ext="png"
fig.savefig(out_base.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")

print("Mean correlation: ", np.mean(total_mean))

mean_corr = np.nanmean(corr_data, axis=0)

# List of mask file paths (replace with your actual paths)
source_folder = "/BRAIN/neuromod-data/static00/anat.atlases/tpl-sub"+s.zfill(2)+"T1w/"
mask_paths = [
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-AngularG_mask.nii.gz",
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-AntTemp_mask.nii.gz",
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-dmpfc_mask.nii.gz",
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-IFG_mask.nii.gz",
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-IFGorb_mask.nii.gz",
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-MFG_mask.nii.gz",
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-pCingulate_mask.nii.gz",
    source_folder+"tpl-sub0"+s+"T1w_res-func_atlas-langToneva_label-PostTemp_mask.nii.gz",
]

ref_img = nib.load(f'{data_dir}/sub-{s.zfill(2)}/transforms/align_auto/reference.nii.gz')
ref_shape = ref_img.shape
ref_affine = ref_img.affine

if mean_corr.T.shape != ref_shape:
    print(mean_corr.shape, ref_shape)
    print("Shape mismatch between your data and the reference shape!")
    exit()
else:
    print("Shape match between your data and the reference shape! Continue...")



aligned_mean = mean_corr.T  # shape should match ref_img (and masks)

region_means = {}
region_counts = {}

# For an overall mean across the *union* of all masks
union_mask = np.zeros(ref_shape, dtype=bool)

for mpath in mask_paths:
    # Derive a readable region name from the filename
    mbase = os.path.basename(mpath)
    m = re.search(r'label-([^_]+)_mask\.nii\.gz$', mbase)
    region_name = m.group(1) if m else mbase

    mask_img = nib.load(mpath)
    mask_data = mask_img.get_fdata()

    if mask_data.shape != ref_shape:
        raise ValueError(
            f"Mask shape {mask_data.shape} != reference shape {ref_shape} for {mbase}"
        )

    # Binarize the mask (works for 0/1 or probabilistic masks)
    mask_bool = mask_data > 0

    # Update union for the overall masked mean
    union_mask |= mask_bool

    # Pull values from aligned_mean inside this mask, ignore NaNs
    vals = aligned_mean[mask_bool]
    valid = vals[~np.isnan(vals)]

    region_means[region_name] = float(valid.mean()) if valid.size else np.nan
    region_counts[region_name] = int(valid.size)

# Overall mean across the union of all masks (ignoring NaNs)
union_vals = aligned_mean[union_mask]
union_valid = union_vals[~np.isnan(union_vals)]
overall_masked_mean = float(union_valid.mean()) if union_valid.size else np.nan

# --- print results ---
print("\nPer-region mean correlations (NaNs ignored):")
for name in sorted(region_means):
    print(f"{name:15s}  mean ={region_means[name]: .6f}  n_vox={region_counts[name]}")


# Create cortex volume
surf_vol = cortex.Volume(
    mean_corr,
    f'sub-{s.zfill(2)}',
    'align_auto',
    vmin=np.nanmin(mean_corr),
    vmax=np.nanmax(mean_corr),
    cmap='magma',
    recache=True,
    surface='inflated',
)


# Start Pycortex web viewer
fig = cortex.webshow(surf_vol, port=37270, open_browser=False)
print("Pycortex server is running... Press Ctrl+C to exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down.")




