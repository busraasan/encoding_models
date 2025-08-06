from process_tr import *
from utils.utils import *
from extract_lm_features import *
import argparse
import os
from utils.ridge_tools import cross_val_ridge
import warnings
from sklearn.model_selection import KFold

story_dict = {
    "PiemanStory": [7,9,11],
    "LucyStory": [1,3,5],
    # "PrettymouthStory": [8,10,12],
    "NotthefallintactStory": [2,4,6],
    # "Tunnelpart2Story": [8,10,12],
    # "Tunnelpart1Story": [8,10,12],
    "SlumlordStory": [7,9,11],
    "BlackStory": [2,4,6],
    "ForgotStory": [1,3,5]
}

TRIM_START = 0
TRIM_END = 1

experiment_TR_offset_start = 3
experiment_TR_offset_end = 6

recording_path = "/BRAIN/neuromod-data/static00/narratives.stimuli/audio_files_filtered/"

for task_name, sessions in story_dict.items():
    session = sessions[-1]  # Use the first session for now

    fmri_data = load_fmri_data_lang_reg(datadir="/BRAIN/neuromod-data/static00/narratives.fmriprep",
                                        subject=2,
                                        session='ses-' + str(session).zfill(3),
                                        task=task_name,
                                        resample_masks=False,
                                        start_trim=TRIM_START,
                                        end_trim=TRIM_END)
    
    if task_name == "NotthefallintactStory":
        wav_file = "notthefall_audio.wav"
    else:
        wav_file = task_name.lower().split("story")[0] + "_audio.wav"
    
    tr_times = read_tr_times(os.path.join(recording_path, wav_file))
    fmri_data = fmri_data[experiment_TR_offset_start:-experiment_TR_offset_end]
    
    print(fmri_data.shape)
    print(tr_times.shape)