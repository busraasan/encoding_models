import time as tm
import numpy as np
import torch
import os
import argparse
from models.encoding_model import EncodingModel

model_options = ['google-bert/bert-base-uncased', 'meta-llama/Llama-2-7b']

parser = argparse.ArgumentParser()
parser.add_argument("--nlp_model", default='google-bert/bert-base-uncased', choices=model_options)                
parser.add_argument("--sequence_length", type=int, default=20, help='length of context to provide to NLP model (default: 1)')
parser.add_argument("--output_dir", required=True, help='directory to save extracted representations to')

args = parser.parse_args()

def save_layer_representations(model_layer_dict, model_name, seq_len, save_dir):             
    for layer in model_layer_dict.keys():
        np.save('{}/{}_length_{}_layer_{}.npy'.format(save_dir,model_name,seq_len,layer+1),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Create the text array to process chunk by chunk
with open("../stimuli_transcriptions/lucy_audio.txt") as f:
    text = f.read()[:1000]
text_array = np.array(text.split())

# Initialize the encoding model
encoding_model = EncodingModel(
            model='google-bert/bert-base-uncased',
            tokenizer='google-bert/bert-base-uncased',
            seq_len=20,
            text_array=text_array,
            remove_chars=[",","\"","@"],
        )

# Run the chunks through the model and get sequence representation per sequence in the text_array
nlp_features = encoding_model.get_layer_representations()

# Save representations
save_layer_representations(nlp_features, args.nlp_model, args.sequence_length, args.output_dir)
