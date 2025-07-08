import time as tm
import numpy as np
import torch
import os
import argparse
from models.encoding_model import EncodingModel
from sklearn.decomposition import PCA

model_options = ['BERT', 'Llama-7B']

model_list = {
    "BERT":"google-bert/bert-base-uncased",
    "Llama-7B":"meta-llama/Llama-2-7b"
}

def save_layer_representations(model_layer_dict: dict, model_name: str, seq_len: int, save_dir: str) -> int:             
    for layer in model_layer_dict.keys():
        np.save('{}/{}/{}_length_{}_layer_{}.npy'.format(save_dir, model_name, model_name, seq_len, layer+1),np.vstack(model_layer_dict[layer]))  
    print('Saved extracted features to {}'.format(save_dir))
    return 1

def save_layer_representations_dict(model_layer_dict: dict, model_name: str, seq_len: int, save_dir: str, task_name: str) -> None:
    """
    Stacks each layer’s outputs, packs them into a dict, and saves as a single .npz file.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Build a plain dict: keys are layer indices, values are (N_time × D) arrays
    out_dict = {
        f"layer_{layer+1}": np.vstack(model_layer_dict[layer])
        for layer in model_layer_dict
    }
    fname = os.path.join(
        save_dir,
        f"{task_name}/{model_name}/{model_name}_length_{seq_len}_all_layers.npz"
    )
    # Save compressed
    np.savez_compressed(fname, **out_dict)
    print(f"Saved all layers → {fname}")

def load_layer_representations_dict(path: str) -> dict:
    """
    Loads the .npz back into a regular dict of arrays.
    """
    npzfile = np.load(path)
    # Convert the NpzFile to a true dict
    return { key: npzfile[key] for key in npzfile.files }

def get_nlp_features_fixed_length(seq_len: int, feat_dir: str, model_type: str , train_indicator, representations):
    
    '''
    Take representations of a specific layer and convert them into test and train sets.
    Return both PCA and dense versions of the datasets.
    '''
    
    if model_type == 'BERT' or model_type == 'Llama-7B':
        train = representations[train_indicator]
        test = representations[~train_indicator]
    else:
        print('Unrecognized NLP feature type {}. Available options BERT and Llama-7B use'.format(args.nlp_model))
    
    pca = PCA(n_components=10, svd_solver='full')
    pca.fit(train)
    train_pca = pca.transform(train)
    test_pca = pca.transform(test)

    return train, test, train_pca, test_pca 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--nlp_model", default='BERT', choices=model_options)                
    parser.add_argument("--sequence_length", type=int, default=20, help='length of context to provide to NLP model (default: 1)')
    parser.add_argument("--output_dir", help='directory to save extracted representations to')
    parser.add_argument("--stimuli_file", default="../stimuli_transcriptions/black_audio.txt")
    parser.add_argument("--task_name", default="BlackStory")


    args = parser.parse_args()

    if not os.path.exists(args.output_dir+"/"+args.task_name+"/"+args.nlp_model):
        os.makedirs(args.output_dir+"/"+args.task_name+"/"+args.nlp_model)

    # Create the text array to process chunk by chunk
    with open(args.stimuli_file) as f:
        text = f.read()
    text_array = np.array(text.split())

    # Initialize the encoding model
    encoding_model = EncodingModel(
                model=model_list[args.nlp_model],
                tokenizer=model_list[args.nlp_model],
                seq_len=20,
                text_array=text_array,
                remove_chars=[",","\"","@"],
            )

    # Run the chunks through the model and get sequence representation per sequence in the text_array
    nlp_features = encoding_model.get_layer_representations() # Shape: (num_seqs, num_hidden)
    # Save representations
    save_layer_representations_dict(nlp_features, args.nlp_model, args.sequence_length, args.output_dir, args.task_name)