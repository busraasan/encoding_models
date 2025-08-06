import transformers
from transformers import AutoModel, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from typing import List
import torch
import numpy as np
import os
import time as tm

'''
    Use the embedding of the last word before [CLS] or end_sentence token as the embedding for the sequence.
    Each word's embedding consists of the average token embedding constructing that word.
'''

class EncodingModel():
    # the index of the word for which to extract the representations (in the input "[CLS] word_1 ... word_n [SEP]")
    # for CLS, set to 0; for SEP set to -1; for last word set to -2
    
    def __init__(self, model_name:str, model:str, tokenizer:str, seq_len: int, text_array: List[str], remove_chars: List[str]):
        self.seq_len = seq_len
        self.text_array = text_array # whole text as an array, elements are words and punctuations
        self.remove_chars = remove_chars
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def get_layer_representations(self, word_ind_to_extract=-2) -> dict:
        '''
        Get the position of the word to get the represenations for.
        Return the representations from each layer as a dictionary.
        '''
        
        # get the token embeddings per word
        print('Getting token embeddings...')
        token_embeddings = []
        for word in self.text_array:
            current_token_embedding = self.get_token_embeddings([word]) # one word consists of multiple tokens
            token_embeddings.append(np.mean(current_token_embedding.detach().numpy(), 1)) # average the representations of the tokens to get an embedding per word
        
        # store layer-wise embeddings of particular length
        layerwise_embeddings = {}
        for layer in range(self.model.config.num_hidden_layers):
            layerwise_embeddings[layer] = []
        layerwise_embeddings[-1] = token_embeddings

        if word_ind_to_extract < 0: # the index is specified from the end of the array, so invert the index
            from_start_word_ind_to_extract = self.seq_len + 2 + word_ind_to_extract  # add 2 for CLS + SEP tokens
        else:
            from_start_word_ind_to_extract = word_ind_to_extract

        start_time = tm.time()

        # For the very first few windows, you haven’t yet “filled” a full sliding window of length seq_len, 
        # so you just repeatedly process the first seq_len words exactly seq_len times.
        word_seq = self.text_array[:self.seq_len]
        for _ in range(self.seq_len):
            layerwise_embeddings = self.add_avrg_token_embedding_for_specific_word(word_seq,
                                            from_start_word_ind_to_extract,
                                            layerwise_embeddings)

        # Add the embedding of the last word in a sequence as the embedding for the sequence
        for end_curr_seq in range(self.seq_len, len(self.text_array)):
            word_seq = self.text_array[end_curr_seq-self.seq_len+1:end_curr_seq+1]
            layerwise_embeddings = self.add_avrg_token_embedding_for_specific_word(word_seq,
                                            from_start_word_ind_to_extract,
                                            layerwise_embeddings)

            if end_curr_seq % 100 == 0:
                print('Completed {} out of {}: {}'.format(end_curr_seq, len(self.text_array), tm.time()-start_time))
                start_time = tm.time()
        
        print('Done extracting sequences of length {}'.format(self.seq_len))
        return layerwise_embeddings

    def _prepare_tokens(self, words_in_array: List[str]):
        '''
        Goal -- Transformation: word sequence -> token sequence, keep corresponding indices in each array
        Take the word and chunk it into tokens.
        Get the token indexes for the sequence and count which indices of the tokens in a token sequence that word corresponds to.
        Return the word's token indexes and the corresponding places in the token seq dictionary word_ind_to_token_ind.
        '''

        for word in words_in_array:
            if word in self.remove_chars:
                print('An input word is also in remove_chars. This word will be removed and may lead to misalignment. Proceed with caution.')

        seq_tokens = []
        n_seq_tokens = 0

        word_ind_to_token_ind = {}            # dict that maps index of word in words_in_array to index of tokens in seq_tokens
    
        for i, word in enumerate(words_in_array):
            word_ind_to_token_ind[i] = []      # initialize token indices array for current word

            if word in ['[CLS]', '[SEP]']:     # [CLS] and [SEP] are already tokenized
                word_tokens = [word]
            else:    
                word_tokens = self.tokenizer.tokenize(word)

            for token in word_tokens:
                if token not in self.remove_chars:  # don't add any tokens that are in remove_chars
                    seq_tokens.append(token)
                    word_ind_to_token_ind[i].append(n_seq_tokens)
                    n_seq_tokens = n_seq_tokens + 1

        # convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(seq_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])

        return tokens_tensor, word_ind_to_token_ind


    def get_token_embeddings(self, words_in_array: List[str]):
        '''
        Get lastlayer embeddings for a sequence of tokens
        '''

        tokens_tensor, _ = self._prepare_tokens(words_in_array)
        if "gpt" in self.model_name.lower():
            
            embed_layer = self.model.get_input_embeddings()
            token_embeddings = embed_layer(tokens_tensor)
        else:
            token_embeddings = self.model.embeddings.forward(tokens_tensor)
        
        return token_embeddings

    def predict_embeddings(self, words_in_array: List[str]):
        '''
        Get all hidden layer embeddings for a sequence of tokens
        '''
        
        tokens_tensor, word_ind_to_token_ind = self._prepare_tokens(words_in_array)
        outputs = self.model(tokens_tensor, output_hidden_states=True, return_dict=True) # full forward pass, get representations per token
        encoded_layers = torch.stack(outputs.hidden_states, dim=0) # Shape = [num_layers, batch_size, seq_token_len, hidden_dim]
        if "BERT" in self.model_name:
            pooled_output = outputs.pooler_output.detach().cpu().numpy().squeeze() # pool to last CLS token (might need to check this)
        else:
            pooled_output = None

        return encoded_layers, word_ind_to_token_ind, pooled_output


    def add_avrg_token_embedding_for_specific_word(self, word_seq, word_idx, layer_rep_dict):
        '''
        Using the predict_embeddings method, compute all the hidden layer representations for a word.
        Fill the layer representations dictionary, averaging all tokens belonging to the single specific word given by word_idx.
        This will be used as the main embedding for the sequence.
        '''

        # pick start and end tokens per model
        start_tok = (
            self.tokenizer.cls_token
            or self.tokenizer.bos_token
            or self.tokenizer.pad_token
            or ""
        )

        end_tok = (
            self.tokenizer.sep_token
            or self.tokenizer.eos_token
            or self.tokenizer.pad_token
            or ""
        )

        word_seq = [start_tok] + list(word_seq) + [end_tok] # add start and end tokens model specific
        all_sequence_embeddings, word_ind_to_token_ind, _ = self.predict_embeddings(word_seq)
        token_inds_to_avg = word_ind_to_token_ind[word_idx] # array of which tokens are part of that word
        layer_rep_dict = self.add_word_embedding(layer_rep_dict, all_sequence_embeddings, token_inds_to_avg)
        
        return layer_rep_dict

    # add all of the token embeddings for a specific word in the sequence together
    def add_word_embedding(self, layer_rep_dict, embeddings_to_add, token_inds_to_avg, specific_layer=-1):
        '''
        Given list of tokens belonging to a word, compute the average embedding for that word over the token embeddings
        '''
        if specific_layer >= 0:  # only add embeddings for one specified layer
            layer_embedding = embeddings_to_add[specific_layer]
            full_sequence_embedding = layer_embedding.detach().numpy()
            # take the average over that word's tokens as a representation
            layer_rep_dict[specific_layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avg, :], 0))
        else:
            # do not iterate over the last element as it is the word embeddings
            for layer, layer_embedding in enumerate(embeddings_to_add[:-1]):
                full_sequence_embedding = layer_embedding.detach().numpy()
                # take the average over that word's tokens as a representation
                layer_rep_dict[layer].append(np.mean(full_sequence_embedding[0, token_inds_to_avg, :], 0)) # avg over all tokens for specified word
        return layer_rep_dict

if __name__ == '__main__':

    text_array = np.load('../stimuli_words.npy') # 5176 words

    encoding_model = EncodingModel(
            model='google-bert/bert-base-uncased',
            tokenizer='google-bert/bert-base-uncased',
            seq_len=20,
            text_array=text_array,
            remove_chars=[",","\"","@"],
        )
    
    layerwise_reps = encoding_model.get_layer_representations()
    for layer in layerwise_reps:
        print(layer, layerwise_reps[layer].shape)
