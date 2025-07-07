import time as tm
import numpy as np
import torch
import os
import argparse

model_options = ['bert','transformer_xl','elmo','meta-llama/Llama-2-7b']

parser = argparse.ArgumentParser()
parser.add_argument("--nlp_model", default='bert', choices=model_options)                
parser.add_argument("--sequence_length", type=int, default=20, help='length of context to provide to NLP model (default: 1)')
parser.add_argument("--output_dir", required=True, help='directory to save extracted representations to')

args = parser.parse_args()

def save_layer_representations(model_layer_dict, model_name, seq_len, save_dir):             
    for layer in model_layer_dict.keys():
        np.save('{}/{}_length_{}_layer_{}.npy'.format(save_dir,model_name,seq_len,layer+1),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1

