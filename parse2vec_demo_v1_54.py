#ex. time python3 parse2vec.py text_vectors_test1.json string_word_transformer_ppnnone_single_context_loadmodel parse_resources/text_vectors_dictionary_chambers.json parse_resources/text_vectors_dictionary_cide.json parse_resources/text_vectors_dictionary_oxford.json parse_resources/text_vectors_dictionary_websters.json parse_resources/text_vectors_dictionary_wordnet.json parse_resources/text_vectors_thesaurus_brit_thesaurus.json parse_resources/text_vectors_thesaurus_fallows.json parse_resources/text_vectors_thesaurus_mythes_wordnet.json parse_resources/text_vectors_thesaurus_rogets.json parse_resources/text_vectors_idioms_american_heritage.json parse_resources/text_vectors_idioms_great_book_american_idioms.json parse_resources/text_vectors_idioms_idioms_amp_phrases.json transformer_model.pickle transformer_model.pickle prop_noun_set1.txt 0 0 4

import warnings
import numpy as np
#import copy as cp
import sys
import json
import copy
#from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss
#import pandas as pd
#import re
import os
import pyphalanx
import random
import pickle
from sklearn.neural_network import MLPClassifier
from scipy.stats import spearmanr
import itertools

#from torchmetrics.text import Perplexity
import time
import json_stream
import gc


AVAILABLE=97
SELECTED=98
DROPOFF=99

FEATURES_C=0
VECTORS_C=1
IDIOMS_C=2

from nltk.corpus import stopwords
import torch
import subprocess

seed = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed) # For multi-GPU setups
random.seed(seed)
enable_pickle = True
enable_pickle_defs = False
enable_measurements = False
enable_dataloader1 = False
enable_memmap = False
disable_propn = False
dep_active = False
depd_active = False
deph_active = False
suf_active = False
morph_active = False
pos_active = False
pos_active2 = False
gender_active = False
gender_prop_active = False
genderPRN_active = False
genderPPN_active = False
genderN_active = False
genderNPPN_active = False
genderNPRN_active = False
freq_active = False
freq2_active = False
sent_active = False
#global sent2_active
sent2_active = False
sent3_active = False
clause_active = False
clause2_active = False
dep2_active = False
dep3_active = False
dep2a_active = False
dep3a_active = False
cluster_active = False
masked_init_active = False
masked_init2_active = False
dep4_active = False
dep5_active = False
#global punct_active
punct_active = False
hypoth_active = False
hypoth1_active = False
coref_active = False
coref2_active = False
early_dropout_active = False
early_fusionC_active = False
early_fusionC2_active = False
early_fusionG_active = False
early_fusionG2_active = False
early_fusionGA_active = False
early_fusionG2A_active = False
early_fusionG2B_active = False
early_fusionH_active = False   # a 2 x 2 block
early_fusionH2_active = False   # a 4 X 2 block
early_fusionH3_active = False   # an L of 1, 2, 3, 4 with 0's in the 3,2,1,0
early_fusionH4_active = False # a 1 X 4 block projected back into embed_dim
early_fusionH5_active = False # a 4 X 1 block plus an 8 X 1 block of half dim and with 4 zero blocks
early_fusionH6_active = False # like H5 using the same heads and reorganized so that the overlaps are roughly doubled
early_fusionH7_active = False # a 4 X 1 block plus an 8 X 1 block of half dim and with 2x1(half), 3x1 and 6x1 (half), 1x1
early_fusionH8_active = False # a 4 X 1 block plus an 8 X 1 block of half dim and with 8 blocks, no zeros
early_fusionH9_active = False # a 1 X 8 block
early_fusionA_active = False
early_fusionW_active = False
early_fusionZ0_active = False
early_fusionZ1_active = False
early_fusion_shuffle_active = False
missing_active = False
segment_lemma_active = False
segment2_lemma_active = False
segment_dep_active = False
rag_active = False
ragU_active = False
ragV_active = False
ragW_active = False
ragR_active = False
ragF_active = False
ragG_active = False
ragK_active = False
rag_blanket_active = False
rag_blanket2_active = False
rag_blanket0_active = False
rag_filter_active = False
hard_rag_blanket = False
#dep_positions_active = False
dep_rel_positions_active = False
dep_pointers_active = False
pos_pointers_active = False
rep_pointers_active = False
rep2_pointers_active = False
folding_active = False
folding_echo_active = False
folding_bidirectional_active = False
folding_once_active = False
folding_rag_active = False
folding_rag_echo_active = False
folding_rag_bidirectional_active = False
folding_rag_once_active = False
folding_vocab_active = False
artificial_dep_active = False
artificial_dep1_active = False
artificial_dep2_active = False
artificial_dep3_active = False
artificial_dep4_active = False
artificial_dep5_active = False
artificial_dep6_active = False
artificial_dep7_active = False
artificial_dep8_active = False
artificial_dep9_active = False
nulls1_active = True
nulls2_active = False
compound_active = False
compound2_active = False
dep_missing_active = False
attention_shift_active = False
renorm_active = False
renorm2_active = False
renorm3_active = False
lemma_unk_active = True
head_weights_active = False
radial_basis_all = False
radial_basis_active = False
radial_basis2_active = False
radial_basis3_active = False
radial_basis4_active = False
radial_basis_key = 0
inattention_active = False
inattention_all_active = False
inattention_cls_active = False
inattention_cls1_active = False
inattention_cls2_active = False
inattention_cls3_active = False
inattention_ffn_active = False
inattention_sum_active = False
clause_flags_active = False
clause_flags2_active = False
clause_flags3_active = False
clause_flags4_active = False
echo_morph_active = False
echo_ent_active = False
echo_compound_active = False
echo_flag_active = False
cosine_similarity_active = False
cosine1_similarity_active = False
tanh_sigmoid_active = False
tanh_sigmoid1_active = False
tanh_sigmoid2_active = False
tanh_active = False
sigmoid_active = False

mlp_layers = 0
prune_size = 0
window_size = 2
rag_window_size = 0
prune_sizeG = 0
window_sizeG = 2
rag_window_sizeG = 0
morph_i_G = 0
morph_i_max_G = 0
ent_type_i_G = 0

folding = "_w_"
folding_full = "_w_"
rag_folding_full = "_w_"
all_folding_full = ".fkjdtenxvproiusb!yzc"
folding_options = {".":"punct", "f":"NUM", "k":"UNK", "j":"ADJ", "d":"ADV", "t":"DET", "e":"PROPN", "n":"NOUN", "x":"AUX", "v":"VERB", "p":"PREP:ADP", "r":"pobj", "o":"dobj", "i":"iobj", "u":"PRON", "s":"csubj:csubjpass:nsubj:nsubjpass", "b":"CCONJ:SCONJ","!":"INTJ","y":"X:SYM","z":"PART","c":"compound"}
folding_options1 = {".":"punct", "f":"NUM", "k":"UNK","!":"INTJ","y":"SYM:X"}
folding_options2 = {"e":"PROPN", "s":"csubj:csubjpass:nsubj:nsubjpass"}
folding_next = ["last_punct","next_punct","last_NUM", "next_NUM","last_UNK", "next_UNK","last_INTJ", "next_INTJ","last_X","next_X", "last_SYM", "next_SYM", "last_PART", "next_PART"]

clause_phrase_options = ["noun_phrase","prep_phrase","main_clause","sub_clause","clause","phrase"]
morph_options = ["morph_Tense=Past","morph_Tense=Pres","morph_Aspect=Perf","morph_Aspect=Prog","morph_Mood=Ind"]
aux_options = ["can_STR","could_STR","may_STR","might_STR","must_STR","shall_STR","should_STR","will_STR","would_STR"]
rel_options = ["acl","hacl","aclhacl","advcl","hadvcl","advclhadvcl","appos","happos", "apposhappos","ccomp", "hccomp", "ccomphccomp", "pcomp", "hpcomp", "pcomphpcomp", "relcl", "hrelcl", "relclhrelcl","xcomp","hxcomp","xcomphxcomp"]
rel_options2 = ["acl","advcl","ccomp","pcomp","relcl","xcomp"]

folding_markers = {"folding_-1":"folding_-0","folding_-0":"folding_-0","folding_0":"folding_1","folding_1":"folding_2","folding_2":"folding_2"}

missing_options = {"ADJ":1, "DET":1, "PROPN":1, "NOUN":1, "VERB":1, "ADP":1}

artificial_deps2 = {}
artificial_preps = {}
artificial_hpreps = {}
artificial_preps2 = {}
artificial_hpreps2 = {}


attn1_active = False
headX_active = False

pred_input_active = False
pred_output_active = False


y_test_len = 20
zero_shot_forced = False
input_mode = "parser"
output_mode = "surface"

model_ckpt = "bert-base-uncased"

from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(model_ckpt)

from math import sqrt

start_time = time.time()
early_fusion = False
x_dimension_1 = False
xo_active = False
xp_active = False
xq_active = False
y_dimension_1 = False
sent_max = 2
phrase_max = 16
word_max = 32

dep_g = {"acl":19,"acomp":5,"amod":5,"advcl":19,"advmod":13,"agent":6,"attr":5,"aux":14,"auxpass":14,"appos":19,"cc":4,"ccomp":4,"conj":19,"cop":9,"csubj":2,"csubjpass":2,"dative":10,"dep":13,"det":18,"dobj":17,"expl":1,"intj":16,"iobj":10,"mark":11,"meta":11,"neg":8,"nmod":6,"nn":19,"npadvmod":13,"nsubj":16,"nsubjpass":1,"num":15,"nummod":15,"oprd":6,"parataxis":8,"pcomp":6,"pobj":1,"poss":12,"possessive":7,"preconj":19,"predet":1,"prep":11,"prt":13,"punct":20,"quantmod":1,"relcl":19,"ROOT":1,"xcomp":6,"nsubjN":25,"dobjN":24,"iobjN":23,"amodN":22,"modN":21}

dep_h = {"hacomp":6,"hacl":19,"hadvcl":19,"hadvmod":14,"hagent":1,"hamod":5,"happos":19,"hattr":5,"haux":9,"hauxpass":9,"hcc":4,"hccomp":4,"hconj":19,"hcop":7,"hcsubj":2,"hcsubjpass":2,"hdative":10,"hdep":13,"hdet":12,"hdobj":17,"hexpl":1,"hintj":16,"hiobj":13,"hmark":11,"hmeta":11,"hneg":5,"hnmod":6,"hnn":19,"hnpadvmod":13,"hnsubj":15,"hnsubjpass":11,"hnum":16,"hnummod":15,"hoprd":6,"hparataxis":8,"hpcomp":6,"hpobj":18,"hposs":10,"hpossessive":5,"hpreconj":19,"hpredet":1,"hprep":8,"hprt":13,"hpunct":20,"hquantmod":1,"hrelcl":19,"hxcomp":19,"hnsubjN":25,"hdobjN":24,"hiobjN":23,"hamodN":22,"hmodN":21}

dep_segment_g = {}
dep_segment_h = {}

btags = ["ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP$", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH", "VBD", "VBG", "VBN", "VB", "VBP", "VBZ", "WDT", "WP$", "WP", "WRB", "XX"]

postags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "VERB"]

sub_deps = {}

freq_2s = []
for i in range(3,21):
    freq_2s.append(pow(2,i))
freq_2s_short = []
for i in range(1,8):
    freq_2s_short.append(pow(2,i))

labels_test = []
for i in range(21):
    labels_test.append(str(i))
    labels_test.append(str(i)+"_STR")

def get_linked_index(corpus,i,w,ids):
    word = corpus[i]
    id1 = word["ID"]
    dep_w = word[w]
    if "-" in dep_w:
        id2 = dep_w.split("-")[1]
        if id2 != "-1" and id2 in ids:
            for i2 in ids[id2]:
                if abs(i2-i) < 30 and i2 != i:
                    return [i2,id2]
    return [-1,"-1"]

def add_dict_if_dict(w,dict1,dict2):
    if w in dict1:
        dict2[dict1[w]] = w
    return dict2

def is_target_feature(w,output_strings={}):
    if output_mode == "surface":
        #if "you" in w:
        #    print(w,len(output_strings),"_STR" not in w,w[:-4],w[:-4] not in output_strings)
        if "W" in folding and len(output_strings) > 0 and ("_STR" not in w or w[:-4] not in output_strings):
            #print(w,"rej")
            return False
        return "_STR" in w and "UNK_" not in w and "PPN" not in w and "*" not in w and "NUM_" not in w and "DEF_" not in w and "LIT_" not in w and "MASK" not in w and "POSITION" not in w and "last_punct" not in w
    elif output_mode == "pos":
        return w in postags or w == "UNK"
    elif output_mode == "btag":
        return w in btags or w == "UNK"
    elif output_mode == "dep":
        return w in dep_g or w == "UNK"

def is_fold_feature(w):
    if w in ["fold_STR","next_STR","last_STR"]:
        return False
    if w in folding_markers or w in folding_next or (len(w)>8 and w[:5]) in ["fold_","next_","last_"]:
        return True
    return False

def log_sigmoid(x):
    return torch.log(1 / (1 + torch.exp(-x*10)))

def scaled_dot_product_attention(query, key, value, mode, relu=None, layer_norm_1=None):
    dim_k = sqrt(query.size(-1))
    scores = torch.bmm(query, key.transpose(1, 2)) / dim_k
    weights = torch.nn.functional.softmax(scores, dim=-1)
    return torch.bmm(weights, value)

def scaled_dot_product_inattention(query, key, value, mode, relu=None, layer_norm_1=None):
    dim_k = sqrt(query.size(-1))
    scores = torch.bmm(query, key.transpose(1, 2)) / dim_k
    weights = torch.nn.functional.sigmoid(scores)
    attention = torch.bmm(weights, value)
    return attention

def scaled_dot_product_inattention_tanh(query, key, value, mode, relu=None, layer_norm_1=None):
    dim_k = sqrt(query.size(-1))
    scores = torch.bmm(query, key.transpose(1, 2)) / dim_k
    weights = torch.nn.functional.tanh(scores)
    attention = torch.bmm(weights, value)
    return attention


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, config, options, mode, inattention, in_rag, use_inner_len, inner_len):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim).to(device)
        self.k = nn.Linear(embed_dim, head_dim).to(device)
        self.v = nn.Linear(embed_dim, head_dim).to(device)
        self.inattention = inattention
        self.in_rag = in_rag
        self.use_inner_len = use_inner_len
        self.inner_len = inner_len
        self.inattention_all_active = "inattention_all_active" in options
        self.inattention_cls3_active = "inattention_cls3_active" in options
        self.window_size = options["window_size"]
        self.prune_size = options["prune_size"]
        self.rag_window_size = options["rag_window_size"]
        self.enc_size = self.prune_size // 4
        if self.rag_window_size < self.enc_size:
            self.enc_size = self.rag_window_size
        self.enc_window = self.prune_size
        if self.rag_window_size < self.enc_window:
            self.enc_window = self.rag_window_size
        if self.in_rag:
            self.window_pre = int(self.enc_window / 2) - int(self.enc_size / 2)
            self.window_post = self.window_pre + self.enc_size
        else:
            self.window_pre = int(self.window_size / 2) - int(self.prune_size / 2) + 1 + self.rag_window_size
            self.window_post = self.window_pre + self.prune_size
        self.relu = nn.LeakyReLU().to(device)                         # Activation function
        self.layer_norm_1 = nn.LayerNorm(config.max_position_embeddings).to(device)

    def forward(self, hidden_state, mode,in_ragG=False):
        if self.inattention:
            if self.use_inner_len:
                hidden_state2 = hidden_state[:,self.window_pre:self.window_post,:]
                attn_outputs = scaled_dot_product_inattention(self.q(hidden_state), self.k(hidden_state2), self.v(hidden_state2), mode, self.relu, self.layer_norm_1)
            elif self.inattention_all_active or self.in_rag:
                if self.inattention_cls3_active:
                    attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mode, self.relu, self.layer_norm_1)
                else:
                    attn_outputs = scaled_dot_product_inattention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mode, self.relu, self.layer_norm_1)
            else:
                hidden_state2 = hidden_state[:,self.window_pre:self.window_post,:]
                if self.inattention_cls3_active:
                    attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state2), self.v(hidden_state2), mode, self.relu, self.layer_norm_1)
                else:
                    attn_outputs = scaled_dot_product_inattention(self.q(hidden_state), self.k(hidden_state2), self.v(hidden_state2), mode, self.relu, self.layer_norm_1)
        else:
            attn_outputs = scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state), mode, self.relu, self.layer_norm_1)
        return attn_outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, config, options, mode, inattention, in_rag, use_inner_len, inner_len):
        super().__init__()
        embed_dim = config.hidden_size
        self.inattention = inattention
        self.in_rag = in_rag
        self.use_inner_len = use_inner_len
        self.inner_len = inner_len
        self.num_heads = config.num_attention_heads
        self.num_heads2 = self.num_heads * 2
        self.proj = None
        self.proj2a = None
        self.proj2b = None
        self.projX = []
        self.headX_active = "headX_active" in options
        self.early_fusion = "early_fusion" in mode

        head_dim = embed_dim // self.num_heads
        head_dim2 = head_dim // 2
        head_dim3 = embed_dim // 2
        self.head_w = nn.Linear(1, self.num_heads).to(device)
        self.head2_w = nn.Linear(1, self.num_heads2).to(device)
        self.layer_norm_1 = nn.LayerNorm(self.num_heads).to(device)
        self.layer_norm_2 = nn.LayerNorm(self.num_heads2).to(device)
        self.zero_tensor = torch.tensor([0], dtype=torch.float, device=device)
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim, config, options, mode, self.inattention, self.in_rag, self.use_inner_len, self.inner_len) for _ in range(self.num_heads)]
        )
        self.heads2 = None
        self.heads3a = None
        self.heads3b = None
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state, mode,in_rag=False):
        hX = [h(hidden_state, mode,in_rag) for h in self.heads]
        x = torch.cat(hX, dim=-1)
        x = self.output_linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size).to(device)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size).to(device)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob).to(device)

    def forward(self, x,mode):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        if "testset" not in mode:
            x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config, options, mode):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size).to(device)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size).to(device)
        self.attention = MultiHeadAttention(config, options, mode, False, False, False, 0)
        self.feed_forward = FeedForward(config)

    def forward(self, x, mode):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state, mode)
        x = x + self.feed_forward(self.layer_norm_2(x),mode)
        return x


class Embeddings(nn.Module):
    def __init__(self, config, options, mode):
        super().__init__()
        print("vocab_size",config.vocab_size,"hidden_size",config.hidden_size,"max_position_embeddings",config.max_position_embeddings)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,padding_idx=0).to(device)
        self.radial_embeddings = None
        self.radial_weights = None
        self.radial_proj = None
        self.radial_proj2 = None
        self.radial_proj3 = None
        self.radial_basis_active = "radial_basis_active" in options
        self.radial_basis2_active = "radial_basis2_active" in options
        self.radial_basis3_active = "radial_basis3_active" in options
        self.radial_basis4_active = "radial_basis4_active" in options
        self.radial_basis_all = "radial_basis_all" in options
        self.radial_basis_key = options["radial_basis_key"]
        self.headX = "headX_active" in mode
        self.early_fusion = "early_fusion" in mode
        self.early_fusionC = "early_fusionC" in mode
        self.early_fusionG = "early_fusionG" in mode
        self.early_fusionG2 = "early_fusionG2" in mode
        self.early_fusionG2B = "early_fusionG2B" in mode
        self.early_fusionW = "early_fusionW" in mode
        self.transformer0 = "transformer0" in mode
        self.testset = "testset" in mode
        self.cosine_similarity_active = "cosine_similarity_active" in options
        self.cosine1_similarity_active = "cosine1_similarity_active" in options
        self.tanh_active = "tanh_active" in options
        self.sigmoid_active = "sigmoid_active" in options

        if self.radial_basis_active:
            self.radial_weights = nn.Embedding(config.vocab_size, 1,padding_idx=0).to(device)
        if self.radial_basis2_active:
            self.radial_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,padding_idx=0).to(device)

        self.modal_weights = nn.Embedding(config.vocab_size, config.hidden_act,padding_idx=0).to(device)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size).to(device)
        num_rows = int(config.max_position_embeddings / config.hidden_act)
        #print(num_rows, config.max_position_embeddings, config.hidden_act,"init------------------")
        pos_ids = [[]] * (num_rows)
        k = 0
        for i in range(num_rows):
            pos_ids[i] = [0] * config.hidden_act
            for j in range(config.hidden_act):
                pos_ids[i][j] = k
                k+=1

        self.position_ids = torch.LongTensor(pos_ids).to(device)
        if "xav" in config.model_type:
            torch.nn.init.xavier_uniform_(self.token_embeddings.weight)
            torch.nn.init.xavier_uniform_(self.position_embeddings.weight)
        self.layer_norm = nn.LayerNorm(config.hidden_size,eps=1e-12).to(device)
        self.layer_norm2 = nn.LayerNorm(config.hidden_act,eps=1e-12).to(device)
        self.dropout = nn.Dropout().to(device)
        self.dropout2d = nn.Dropout2d().to(device)

        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.q = nn.Linear(embed_dim, head_dim).to(device)
        self.k = nn.Linear(embed_dim, head_dim).to(device)
        self.v = nn.Linear(embed_dim, head_dim).to(device)
        if self.radial_basis3_active:
            self.radial_proj3 = nn.Linear(embed_dim, head_dim).to(device)
        self.h_layer = nn.Linear(config.hidden_size, config.hidden_act)
        self.h2_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.c_layer = nn.Linear(config.hidden_size, head_dim)
        self.embedding_proj3 = nn.Linear(config.hidden_act * head_dim, config.hidden_size).to(device)
        self.embedding_proj3a = nn.Linear(config.hidden_act * head_dim, config.hidden_act * head_dim).to(device)
        self.embedding_proj3b = nn.Linear(config.hidden_act * head_dim, config.hidden_act * head_dim).to(device)
        self.embedding_proj4 = nn.Linear(config.hidden_act * config.hidden_act, config.hidden_size).to(device)
        self.embedding_proj42 = nn.Linear(config.hidden_size, config.hidden_size).to(device)
        self.relu = nn.LeakyReLU().to(device)                         # Activation function
        self.layer_norm_1 = nn.LayerNorm(config.hidden_act).to(device)

        self.zero_tensor = torch.tensor([0], dtype=torch.long, device=device)

    def forward(self, input_ids, mode):
        self.token_embeddings(self.zero_tensor).zero_()
        if self.transformer0:
            with torch.no_grad():
                token_embeddings = self.token_embeddings(input_ids)
        else:
            token_embeddings = self.token_embeddings(input_ids)
        if self.cosine_similarity_active or self.cosine1_similarity_active or self.tanh_active or self.sigmoid_active:
            key_token = radial_basis_key
            size2 = token_embeddings.size(-2)
            size1 = token_embeddings.size(-1)
            token_embeddings0N = token_embeddings[:, :, key_token, :].unsqueeze(-2)

            sim_em = (torch.matmul(token_embeddings0N, token_embeddings.transpose(-1, -2)))

            if self.radial_proj is None:
                self.radial_proj = nn.Linear(size2, size1).to(device)

            if self.cosine_similarity_active:
                norms = torch.linalg.norm(token_embeddings, dim=-1).unsqueeze(-2)
                sim_em = sim_em / norms
            elif self.cosine1_similarity_active:
                norms = torch.linalg.norm(token_embeddings, dim=-1).unsqueeze(-2)
                norms1 = torch.linalg.norm(token_embeddings0N, dim=-1).unsqueeze(-2)
                sim_em = sim_em / norms
                sim_em = sim_em / norms1
            elif self.sigmoid_active:
                sim_em = torch.sigmoid(sim_em)
            elif self.tanh_active:
                sim_em = torch.tanh(sim_em)
            radial_embeddings2 = self.radial_proj(sim_em)
            token_embeddings = torch.cat([token_embeddings, radial_embeddings2], dim=-2)

        elif self.radial_basis_active or self.radial_basis2_active or self.radial_basis3_active:
            gamma = 1
            key_token = radial_basis_key
            size2 = token_embeddings.size(-2)
            size1 = token_embeddings.size(-1)
            if self.radial_basis_all:
                token_embeddings0N = token_embeddings
                square0N = torch.sum(token_embeddings ** 2, dim=-1, keepdim=True)
                square = square0N.transpose(-1,-2)
            else:
                token_embeddings0N = token_embeddings[:, :, key_token, :].unsqueeze(-2)
                if radial_basis4_active: #project the feature embedding into an activation embedding
                    token_embeddings0N = self.embedding_proj42(token_embeddings0N)
                square0N = torch.sum(token_embeddings0N ** 2, dim=-1, keepdim=True)
                square = torch.sum(token_embeddings ** 2, dim=-1, keepdim=True).transpose(-1, -2)
            euclideandistance0 = (2 * torch.matmul(token_embeddings0N, token_embeddings.transpose(-1, -2)))
            euclideandistance1 = square0N + square - euclideandistance0
            radial_embeddings1 = torch.exp(-gamma * euclideandistance1)
            if self.radial_basis_active:
                if self.radial_proj is None:
                    self.radial_proj = nn.Linear(size2, size1).to(device)
                radial_weights = self.radial_weights(input_ids)
                radial_weightsT = radial_weights.transpose(-1,-2)
                radial_embeddingsW = radial_weightsT * radial_embeddings1
                radial_embW = self.radial_proj(radial_embeddingsW)
                token_embeddings = torch.cat([token_embeddings, radial_embW], dim=-2)
            elif self.radial_basis3_active:
                if self.radial_proj is None:
                    self.radial_proj = nn.Linear(size2, size1).to(device)
                radial_embeddings2 = self.radial_proj(radial_embeddings1)
                token_embeddings = torch.cat([token_embeddings, radial_embeddings2], dim=-2)
            elif self.radial_basis2_active:
                self.radial_embeddings(self.zero_tensor).zero_()
                radial_embeddings = self.radial_embeddings(input_ids)
                radial_embeddings2 = torch.matmul(radial_embeddings1,radial_embeddings)
                token_embeddings = torch.cat([token_embeddings, radial_embeddings2], dim=-2)
        if self.early_fusion:
            if self.early_fusionC:
                dim_k = self.config.hidden_act
                token_embeddingsc = self.c_layer(token_embeddings)
                token_embeddings2 = torch.cat([token_embeddingsc[:,:,i,:] for i in range(dim_k)], dim=-1)
                token_embeddingsF = self.embedding_proj3(token_embeddings2)
            elif self.early_fusionG:
                dim_k = self.config.hidden_act
                if self.early_fusionG2:
                    if self.early_fusionG2B:
                        h = torch.tanh(token_embeddings)
                    else:
                        h = torch.tanh(self.h2_layer(token_embeddings))
                    token_embeddings2 = h.sum(-2).unsqueeze(-2)
                    token_embeddings2r = token_embeddings2.repeat(1, 1, dim_k, 1)
                    z = self.layer_norm(torch.sigmoid(self.relu(self.embedding_proj42(token_embeddings2r))))
                    token_embeddingsZ = torch.matmul(h.transpose(2,3),z)
                    token_embeddingsF = token_embeddingsZ.sum(-2).squeeze(-2)
                    if self.early_fusionG2A:
                        token_embeddingsF += token_embeddings.sum(2).squeeze(2)
                else:
                    h = torch.tanh(self.h_layer(token_embeddings))
                    token_embeddings2 = torch.cat([h for i in range(dim_k)], dim=-1)
                    z = torch.sigmoid(self.relu(self.embedding_proj4(token_embeddings2)))
                    token_embeddingsZ = torch.matmul(h,z)
                    token_embeddingsF = token_embeddingsZ.sum(-2).squeeze(-2)
                    if self.early_fusionGA:
                        token_embeddingsF += token_embeddings.sum(2).squeeze(2)

            elif self.early_fusionW:
                modal_weights = self.modal_weights(input_ids[:,:,0]).unsqueeze(-2)
                modal_norm = self.layer_norm2(modal_weights)
                dim_k = self.config.hidden_act
                token_embeddingsw = torch.matmul(modal_norm,token_embeddings) / dim_k
                token_embeddingsF = token_embeddingsw.squeeze(-2)
            else:
                token_embeddingsF = token_embeddings.sum(2).squeeze(2)
        embeddings = self.layer_norm(token_embeddingsF)

        if "testset" not in mode:
            embeddings = self.dropout(embeddings)
        return embeddings

    def set_values(self,new_weights):
        self.token_embeddings.weight = nn.Parameter(new_weights)

    def set_embeddings(self,embeddings):
        print("set_embeddings")
        self.token_embeddings = embeddings

    def get_embeddings(self):
        self.token_embeddings(self.zero_tensor).zero_()
        return self.token_embeddings, self.position_embeddings

    def embeddings_only(self,input_ids,mode):
        return self.token_embeddings(input_ids)

    def print_embeddings(self):
        print(self.token_embeddings)
        print("-----------------\n\n-------------\n\n--------------")

class TransformerEncoder(nn.Module):
    def __init__(self, config, options, mode):
        super().__init__()
        self.inattention_active = "inattention_active" in options
        self.inattention_cls_active = "inattention_cls_active" in options
        self.inattention_cls1_active = "inattention_cls1_active" in options
        self.inattention_cls2_active = "inattention_cls2_active" in options
        self.inattention_cls3_active = "inattention_cls3_active" in options
        self.inattention_ffn_active = "inattention_ffn_active" in options
        self.window_size = options["window_size"]
        self.rag_window_size = options["rag_window_size"]
        self.prune_size = options["prune_size"]
        self.enc_size = prune_size // 4
        self.ragF_active = "ragF_active" in options
        self.ragG_active = "ragG_active" in options
        self.inattention_sum_active = "inattention_sum_active" in options
        self.window_pre = int(self.window_size / 2) - int(self.prune_size / 4) + 1 + self.rag_window_size
        self.window_post = self.window_pre + int(self.prune_size/2)
        self.block_size = int(self.prune_size / 2) - 1
        self.embeddings = Embeddings(config, options, mode).to(device)
        self.layers = nn.ModuleList([TransformerEncoderLayer(config, options, mode) for _ in range(config.num_hidden_layers)]).to(device)
        self.inattention = None
        self.inattention_threshold = None
        self.layer_norm_1 = None
        self.layer_norm_2 = None
        self.layer_norm_3 = None
        self.encoding_proj = nn.Linear(config.hidden_size, config.hidden_size // 4).to(device)
        self.encoding_proj2 = nn.Linear(config.hidden_size // 4, config.hidden_size).to(device)
        self.y_proj = nn.Linear(self.window_size, config.hidden_size).to(device)
        self.feed_forward_inattention = None
        if self.inattention_active:
            self.inattention = MultiHeadAttention(config, options, mode, True, False, False, 0)
            self.inattention_threshold = nn.Linear(config.hidden_size, 1).to(device)
            self.hidden_size = config.hidden_size
            self.layer_norm_1 = nn.LayerNorm(config.hidden_size).to(device)
            self.layer_norm_2 = nn.LayerNorm(config.hidden_size).to(device)
            if self.inattention_ffn_active:
                self.feed_forward_inattention = FeedForward(config)
                self.layer_norm_3 = nn.LayerNorm(config.hidden_size).to(device)
        self.encodings = None
        self.rag_attention = None
        self.rag_attentionA = None
        self.rag_threshold = None
        self.layer_norm_1r = None
        self.layer_norm_2r = None
        self.feed_forward = None

    def forward(self, x,mode,batch_i=999999):
        x = self.embeddings(x,mode)
        hidden_state = None
        r1 = 0
        if self.ragF_active or (self.inattention_active and x.size(1) != self.prune_size):
            hidden_state = self.layer_norm_1(x)
            r1 = x.size(0)

        if self.inattention_active and x.size(1) != self.prune_size:
            inattention1 = self.inattention(hidden_state, mode)
            if self.inattention_ffn_active:
                inattention1 = self.feed_forward_inattention(self.layer_norm_3(inattention1),mode)
            inattention2 = self.layer_norm_2(inattention1)
            y = self.inattention_threshold(inattention1)
            y = y.squeeze()
            if self.inattention_cls_active:
                x0 = torch.zeros_like(x)
                x0[:,0] = x0[:,0] + self.y_proj(y)
                x = x + x0
                x = x + inattention2
            elif self.inattention_cls1_active:
                x = x + y.unsqueeze(-1)
                x = x + inattention2
            elif self.inattention_cls2_active:
                x = x + y.unsqueeze(-1)
            else:
                x = x + inattention2
            y_block = torch.cat([y[:,1:self.window_pre],y[:,self.window_post:]], dim=1)
            x_block = torch.cat([x[:,1:self.window_pre,:],x[:,self.window_post:,:]], dim=1)

            values, indexes = torch.topk(y_block, self.block_size, dim=1, sorted=False)

            batch_seq = torch.arange(r1)[:, None]

            x = torch.cat([x[:,0,:].unsqueeze(1),x[:,self.window_pre:self.window_post,:],x_block[batch_seq, indexes]], dim=1)

        for layer in self.layers:
            x = layer(x, mode)
        return x

    def set_values(self,new_weights):
        self.embeddings.set_values(new_weights)

    def set_embeddings(self,embeddings):
        self.embeddings.set_embeddings(embeddings)

    def get_embeddings(self):
        return self.embeddings.get_embeddings()

    def print_embeddings(self):
        self.embeddings.print_embeddings()



class TransformerForContextTraining(nn.Module):
    def __init__(self, config,num_labels, options, mode):
        super().__init__()
        self.options = options
        self.encoder = TransformerEncoder(config, options, mode).to(device)
        self.dropout = nn.Dropout(config.hidden_dropout_prob).to(device)
        self.classifier = nn.Linear(config.hidden_size, num_labels).to(device)
        self.relu = nn.LeakyReLU().to(device)
        self.config = config
        self.optimizer = None

    def rag_encoding(self,inputs, mode):
        return self.encoder.rag_encoding(inputs, mode)

    def ragF_encoding(self,inputs, mode, rag_key,in_rag):
        return self.encoder.ragF_encoding(inputs, mode, rag_key,in_rag)

    def reset_classifier(self,config):
        pass

    def set_values(self,new_weights):
        self.encoder.set_values(new_weights)

    def set_embeddings(self,embeddings):
        self.encoder.set_embeddings(embeddings)

    def get_embeddings(self):
        return self.encoder.get_embeddings()

    def forward(self, x,batch_i=0,i=0,context_cache=None,mode=""):
        if True:
            if ("tune1" in mode or "reg_test1" in mode):
                with torch.no_grad():
                    x = self.encoder(x,mode)
                    x = x[:, 0, :] # select hidden state of [CLS] token
                    if "testset" not in mode:
                        x = self.dropout(x)
            else:
                x = self.encoder(x,mode,batch_i)
                x = x[:, 0, :] # select hidden state of [CLS] token
                if "testset" not in mode:
                    x = self.dropout(x)

            x = self.classifier(x)
            x = self.relu(x)
            if "zero_shotc"  in mode or "reg_test0c"  in mode:
                x = torch.nn.functional.sigmoid(x)
            elif "zero_shotm"  in mode or "reg_test0m"  in mode:
                x = torch.softmax(x,dim=1)
        return x


    def forward1(self, x,batch_i=0,i=0,context_cache=None,mode=""):
        #print("---2")
        x = self.encoder(x,mode)
        return x


    def testA(self,contexts,targets,blanket,targetsI,targetsM,test_ids,target_label_unions,mode,perplexity=None, y_dimension=1,batch_i=1, context_cache=None, f1=False, results2={}, uniq_correct2={}, uniq_correct3={}, last_set=False,valid_indexes={},valid_mask=None,index_to_word={},word_2_words={},uniq_words={}):
        total_loss = 0
        losses = []
        correct = incorrect = 0
        correct1 = incorrect1 = 0
        correct2 = incorrect2 = 0
        correct3 = incorrect3 = 0
        num_valid_outputs = 0
        target_label_len = {}
        target_label_indexes = {}
        word_prediction = False
        if uniq_words is None:
            uniq_words = {}
        output_list = []
        for w in ["zero_shotcw","zero_shotmw","zero_shotw","reg_test0w","reg_test0cw","reg_test0mw"]:
            if w in mode:
                word_prediction = True

        for t in target_label_unions:
            if t == "": continue
            target_label_len[t] = len(target_label_unions[t])
            target_label_indexes[t] = target_label_unions[t]

        with torch.no_grad():

            outputs = []
            if True:
                if "early_fusion" in mode:
                    outputs = self.forward(contexts,batch_i,0,context_cache,mode)
                else:
                    output = self.forward(contexts[i],batch_i,i,context_cache,mode).sum(0).to("cpu")
            for i in range(len(contexts)):
                output = outputs[i]
                target = targets[i]
                if early_fusion:
                    target = target.squeeze(0)
                if f1:
                    if not word_prediction:
                        test_id = test_ids[i]
                    if not word_prediction:
                        max_score_o = 0
                        max_o = ""
                        max_score_t = 0
                        max_t = ""
                        output_lst = output.tolist()
                        target_lst = target.tolist()
                        for t in target_label_indexes:
                            tmp_o = 0.0
                            score_o = 0.0
                            tmp_t = 0.0
                            score_t = 0.0
                            for u in target_label_indexes[t]:
                                if output_lst[u] >= 0.5:
                                    tmp_o += 1.0
                                if target_lst[u] >= 0.5:
                                    tmp_t += 1.0
                            score_o = tmp_o / target_label_len[t]
                            score_t = tmp_t / target_label_len[t]
                            if score_o > max_score_o:
                                max_score_o = score_o
                                max_o = t
                            if score_t > max_score_t:
                                max_score_t = score_t
                                max_t = t
                        if max_o == max_t and max_o != "":
                            correct+=1
                        else:
                            incorrect+=1

                    if word_prediction:
                        output_masked = output * valid_mask #torch.masked_select(output,valid_mask)
                        try:
                            loss = torch.nn.functional.cross_entropy(output_masked,target,reduction="sum")
                            total_loss += loss.detach().item()
                        except:
                            pass
                        max_o1 = torch.argmax(output_masked).item()
                        max_o2 = torch.max(output_masked).item()
                        max_t1 = list(targetsI[i].keys())
                        im0 = index_to_word[max_t1[0]]
                        if max_o1 == max_t1[0] and len(max_t1) > 0:
                            uniq_words[im0] = 0
                            correct+=1
                            output_list.append(index_to_word[max_o1]+","+im0+",correct")
                        else:
                            incorrect+=1
                            if len(max_t1) > 0:
                                output_list.append(index_to_word[max_o1]+","+im0+",inc")
                            else:
                                output_list.append(index_to_word[max_o1]+",UNK,inc")
                    if not word_prediction:
                        if output[list(targetsI[i].keys())[0]] >= 0.5:
                            correct1+=1
                            incorrect1-=1
                        incorr_mask = output_masked1 >= 0.5
                        incorrect1 += len(torch.masked_select(output_masked1,incorr_mask))
                    if not word_prediction:
                        output_lst = output.tolist()
                        if test_id not in results2:
                            target_lst = target.tolist()
                            #print(test_id)
                            results2[test_id] = {}
                            results2[test_id]["output"] = [0.0] * len(output)
                            results2[test_id]["target"] = [0.0] * len(output)
                            for u in range(len(target_lst)):
                                results2[test_id]["target"][u] = target_lst[u]

                        for u in range(len(output_lst)):
                            results2[test_id]["output"][u] += output_lst[u]
                if valid_mask is not None:
                    num_valid_outputs = valid_mask.sum(0).item()
        total_loss /= len(contexts)
        if f1:
            if last_set and not word_prediction:
                correct2 = incorrect2 = 0
                ii = 0
                for test_id2 in sorted(results2.keys()):
                    #print("-----------4")
                    max_score_o = -100000
                    max_score_o2 = -100000
                    max_o = ""
                    max_o2 = ""
                    max_score_t = 0
                    max_t = ""
                    ii+=1
                    tmpR2O = results2[test_id2]["output"]
                    tmpR2T = results2[test_id2]["target"]
                    for t in target_label_indexes:
                        tmp_o = 0.0
                        tmp_o2 = -100000
                        score_o = 0.0
                        tmp_t = 0.0
                        score_t = 0.0
                        for u in target_label_indexes[t]:
                            if tmpR2O[u] > tmp_o2:
                                tmp_o2 = tmpR2O[u]
                            if tmpR2O[u] >= 0.5:
                                tmp_o += 1.0
                            if tmpR2T[u] > 0.0:
                                tmp_t += 1.0
                        score_o = tmp_o / target_label_len[t]
                        score_t = tmp_t / target_label_len[t]
                        if tmp_o2 > max_score_o2:
                            max_score_o2 = tmp_o2
                            max_o2 = t
                        if score_o > max_score_o:
                            max_score_o = score_o
                            max_o = t
                        if score_t > max_score_t:
                            max_score_t = score_t
                            max_t = t
                    if max_o == max_t and max_o != "":
                        uniq_correct2[max_o] = 1
                        correct2+=1
                        print(ii,test_id2,max_o,max_t,"max_o max_t",correct2)
                    else:
                        incorrect2+=1

                    if max_o2 == max_t and max_o2 != "":
                        uniq_correct3[max_o2] = 1
                        correct3+=1
                        print("-------",ii,test_id2,max_o2,max_t,"max_o2 max_t",correct3)
                    else:
                        incorrect3+=1

            return total_loss, losses, correct, incorrect, correct1, incorrect1, correct2, incorrect2, correct3, incorrect3, len(uniq_correct2), len(uniq_correct3), num_valid_outputs, output_list
        return total_loss, losses

    def trainA(self,contexts,targets,blanket,batch_i=0,context_cache=None,mode="",label=""):
        if "tune1" in mode:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam([
                    {'params': self.classifier.parameters()},
                    {'params': self.fc2.parameters()},
                    {'params': self.fc3.parameters()}
                ])
        else:
            if self.optimizer is None:
                self.optimizer = torch.optim.Adam(self.encoder.parameters()) #,lr=0.001)
        has_def = "DEF" in label
        total_loss = 0
        outputs = None
        if True:
            self.optimizer.zero_grad()
            if "early_fusion" in mode:
                outputs = self.forward(contexts,batch_i,0,context_cache,mode)
                outputs = outputs.squeeze(1)
                if "rag_blanket2_active" in self.options:
                    for i in range(len(outputs)):
                        outputs[i] = outputs[i] * blanket[i]
            else:
                output = self.forward(contexts[i],batch_i,0,context_cache,mode).sum(0)

            if "tune" in mode:
                if ("tune1c" in mode or "tune0c" in mode or "transformers" in mode or "transformer0s" in mode or "transformer00s" in mode or "transformer10s" in mode or "transformer01s" in mode or "transformer11s" in mode) and not ("tune1m" in mode or "tune0m" in mode) and not ("mqx" in mode or "mrx" in mode):
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs,targets)
                else:
                    loss = torch.nn.functional.cross_entropy(outputs,targets)
            else:
                loss = torch.nn.functional.cross_entropy(outputs,targets)


            total_loss += loss.detach().item()

            loss.backward()

            self.optimizer.step()
            #optimizer.zero_grad()
            #print("___________",i,"of",len(contexts))

        #print(total_loss,"train total_loss")
        return total_loss

class TransformerForPreTraining(TransformerForContextTraining):
    def __init__(self, config, options, mode):
        super().__init__(config, config.num_labels, options, mode)



stop_words = set(stopwords.words('english'))
number_words = ["Number_Zero","Number_One","Number_Two","Number_Three","Number_Four","Number_Five","Number_Six","Number_Seven","Number_Eight","Number_Nine"]



def add_idioms(corpus, word_to_index, words_freq, idioms, feature_strings, mode):
    index = max(word_to_index.values()) + 1
    #print(word_to_index, index)
    idiom_map = {}
    for idiom in idioms:
        #print(idioms[idiom]["any"]["idiom"][0]["idiom_order"])
        words = idioms[idiom]["any"][IDIOMS_C][0]["idiom_order"].split("-")
        idiom_map[words[0]] = words
    #print(idiom_map)
    new_corpus = []
    for i,word in enumerate(corpus):
        if i >= len(corpus) - 10: continue
        #print(word)
        c3 = corpus[i:i+10]
        c2 = ""
        for c in c3:
            c2 += " "+c["STR"]
        if "remove" in word: continue
        if word["STR"] in idiom_map:
            words = idiom_map[word["STR"]]
            #print(words)
            #print(c2)
            match = True
            for j in range(1,len(words)):
                match = match and words[j] == c3[j]["STR"]
            if match and "idiommark" in mode:
                #print(words,c2)
                for j in range(0,len(words)):
                    primary_key2 = corpus[i+j]["primary_keys"][1]
                    str1 = corpus[i+j]["STR"]
                    str1 = "IDM_"+str1+"_STR"
                    corpus[i+j][str1] = "idiom"
                    #print(str1,primary_key2,corpus[i])
                    a,b,c = get_primary_keys(corpus[i+j],mode,3,True)
                    words_freq[b] = words_freq[primary_key2]
                    #print(corpus[i],a,b,c)
                    if word_to_index.get(str1) == None:
                        word_to_index.update ( {str1 : index})
                        feature_strings[str1] = 1
                        index  += 1
            if match and "idiomreplstr" in mode:
                primary_key2 = corpus[i]["primary_keys"][1]
                new_word = copy.deepcopy(word)
                del new_word["STR"]
                rem_f = ""
                for f in new_word:
                    if new_word[f] == "word":
                        rem_f = f
                if rem_f != "": del new_word[rem_f]
                strs = []
                for j in range(0,len(words)):
                    if "END" in corpus[i+j]:
                        new_word["END"] = corpus[i+j]["END"]
                    corpus[i+j]["remove"] = "remove"
                    strs.append(corpus[i+j]["STR"])
                new_str = "-".join(strs)
                new_word["STR"] = new_str
                new_word[new_str] = "word"
                new_word["idiom"] = "idiom"
                a,b,c = get_primary_keys(new_word,mode,3,True)
                words_freq[b] = words_freq[primary_key2]
                #print(new_str,new_word,a,b,c)
                if word_to_index.get(new_str) == None:
                    word_to_index.update ( {new_str : index})
                    feature_strings[new_str] = 1
                    index  += 1
                word = new_word
            if match and "idiomreplcnd" in mode:
                primary_key2 = corpus[i]["primary_keys"][1]
                for j in range(0,len(words)):
                    for f in list(corpus[i+j].keys()):
                        corpus[i][f] = corpus[i+j][f]
                        if corpus[i+j][f] == "word":
                            corpus[i]["IDM_"+f] = "idiom"
                            if word_to_index.get("IDM_"+f) == None:
                                word_to_index.update ( {"IDM_"+f : index})
                                feature_strings["IDM_"+f] = 1
                                index  += 1
                    if j>0:
                        corpus[i+j]["remove"] = "remove"
                a,b,c = get_primary_keys(corpus[i],mode,3,True)
                words_freq[b] = words_freq[primary_key2]
                #print(a,b,c,corpus[i])
        new_corpus.append(word)
    del corpus
    return new_corpus, word_to_index, words_freq, feature_strings

def add_proper_nouns(corpus, word_to_index, feature_strings, ppn_set, mode):
    index = max(word_to_index.values()) + 1
    new_corpus = []
    skip_until = -1
    for i,word in enumerate(corpus):
        if i >= len(corpus) - 10:
            new_corpus.append(word)
            continue
        if "propnmark" in mode and i<skip_until:
            continue
        c3 = corpus[i:i+10]
        c2 = ""
        for c in c3:
            c2 += " "+c["STR"]
        if word["STR"] in ["NUM",",",".","\'","a","an",""]: continue
        if "VERB" in word or "PUNCT" in word or "ADP" in word: continue
        if "remove" in word: continue
        # refactor this to require only one PROPN in the phrase and for all words to be upper_case except stop words
        if True: #"PROPN" in word:
            words = []
            words.append(word)
            str1 = word["STR"]
            hnn = "zzz"
            if "hnn" in word:
                hnn = word["hnn"]
            where_when_exclude = "asdf;lkj"
            if "Concept_where" in word: where_when_exclude = "Concept_when"
            if "Concept_when" in word: where_when_exclude = "Concept_where"
            j = 1
            ent_type = ""
            if True:
                for jj in c3[j]:
                    if "ent_type_" in c3[j][jj]:
                        ent_type = jj
            has_propn = "PROPN" in c3[0]
            match = True
            match = match and ("PROPN" in c3[j] or "Sentence_Case" in c3[j]) and (("hnn" in c3[j] and c3[j]["hnn"] == hnn) or ("compound" in c3[j] and (ent_type == "" or ent_type in c3[j]) ) ) and "label" not in c3[j]
            match = match and where_when_exclude not in c3[j]
            if match: has_propn = has_propn or "PROPN" in c3[j]
            while match and j<9:
                words.append(c3[j])
                str1 += " "+c3[j]["STR"]
                j += 1
                match = match and ("PROPN" in c3[j] or "Sentence_Case" in c3[j]) and (("hnn" in c3[j] and c3[j]["hnn"] == hnn) or ("compound" in c3[j] and (ent_type == "" or ent_type in c3[j]) ) ) and "label" not in c3[j]
                match = match and where_when_exclude not in c3[j]
                if match: has_propn = has_propn or "PROPN" in c3[j]

            if not has_propn:
                new_corpus.append(word)
                continue

            skip_until = i+j-1
            if True or "propnreplstr" in mode:

                new_word = copy.deepcopy(word)
                del new_word["STR"]
                rem_f = ""
                for f in new_word:
                    if new_word[f] == "word":
                        rem_f = f
                if rem_f != "": del new_word[rem_f]
                strs = []
                for j in range(0,len(words)):
                    if "END" in corpus[i+j]:
                        new_word["END"] = corpus[i+j]["END"]
                    if "propnreplstr" in mode:
                        corpus[i+j]["remove"] = "remove"
                    strs.append(corpus[i+j]["STR"])
                new_str = "*".join(strs)
                new_word["STR"] = new_str
                if "string" in mode or "parse" in mode:
                    if "_STR" not in new_str:
                        new_str = new_str+"_STR"
                new_word[new_str] = "word"
                ppn_set[new_str] = 1
                new_word["propn_phrase"] = "propn_phrase"
                new_word["PROPN"] = "pos"
                a,b,c = get_primary_keys(new_word,mode,3,True)
        new_corpus.append(word)
    if "propn" not in mode:
        new_corpus = []
        for i,word in enumerate(corpus):
            new_corpus.append(word)
    del corpus
    return new_corpus, word_to_index, feature_strings, ppn_set

def get_ordered_feature_vectors(words,y_dimension,y_limit_1,word_to_index,rag_mode,mode,masked=""):
    concept = {"Concept_male":10,"Concept_female":10,"Concept_neuter":10,"Concept_honorific":9,"Concept_unit":8,"Date_Time":7,"Clock_Time":6,"Concept_when":5,"Concept_where":4,"Concept_who":3,"Concept_state":2,"Concept_country":2,"Concept_continent":2,"Concept_day_of_week":2,"Concept_road":1,"Concept_month":1,"Concept_pii_profession":1,"Concept_post_nominal_title":1}

    pos2 = {"Present_Progressive":2,"Collective":2,"Verb_Is_A":3,"Verb_Has_A":3,"Auxiliary_Verb":4,"Auxiliary_Verb2":4,"Auxiliary_Verb_Past":4,"Auxiliary_Verb_Plural":4,"Auxiliary_Verb_Singular":4,"Past":4,"MD":4,"Infinitive_Verb":5,"Verb_Past":5,"Verb_Plural":6,"Verb_Singular":6,"VB":6,"VBD":6,"VBN":6,"VBZ":6,"Gerund":7,"VBG":7,"VBP":7,"Past_Participle":7,"Proper_Noun_Singular":7,"NNP":7,"NNPS":7,"Pronoun_Singular":7,"Pronoun_Plural":7,"PRP":7,"PRP$":7,"JJ":7,"JJR":7,"JJS":7,"Adjective_Superlative":7,"Adjective_Comparative":7,"CC":8,"RB":8,"RBR":8,"RBS":8,"RP":8,"WP":8,"WP$":8,"Ordinal_Number":8,"Cardinal_Number":8,"CD":8,"Number_Unit":8,"Number_Age":8,"UH":8,"Concept_pii_age":7,"Number":2,"LS":2,"Noun_Plural":9,"Noun_Singular":9,"NN":9,"NNS":9,"Abbreviation":10,"DT":10,"PDT":10,"IN":10,"NFP":10,"$":10,"ADD":10,"FW":10,"TO":10,"HYPH":10,"XX":10,"EX":10,"WRB":8,"WDT":8,"AFX":10}

    skip_types = ["ID","STR","Word_End","Noun","Verb", "Proper_Noun","Adjective", "Pronoun","Unknown_Pos","Adverb", "Preposition","Interjection", "Conjunction", "Topic_Medicine", "Word_Start","Word_End","Sentence_Start","Sentence_End","Paragraph_Start","Paragraph_End","propn_phrase","compound","hcompound","",",",".","POS","case","hcase",":","\'\'",'``',"_SP"]

    global ent_type_i_G
    ent_type_i_G = 8

    suf_i = 14
    suf_i_max = 19
    # missing 11,12
    dep_h_i = 20
    dep_h_i_max = 29
    dep_g_i = 30
    dep_g_i_max = 34
    pos2_i = 35
    pos2_i_max = 37
    cluster_i = 38
    cluster_i_max = 40
    morph_i = 41
    morph_i_max = 43
    global  morph_i_G
    global morph_i_max_G
    morph_i_G = morph_i
    morph_i_max_G = morph_i_max
    concept_i = 44
    concept_i_max = 45
    dep_h2_i = 46
    dep_h2_i_max = 46 # this was set to +11 but is set to +0 while artificial deps are on pause
    dep_g2_i = 47
    dep_g2_i_max = 47 #this was set to +6 but is set to +0 while artificial deps are on pause
    dep_r_i = 48
    dep_r_i_max = 48 #this was set to +4 but is set to +0 while artificial deps are on pause

    folding_i = 49
    folding_i_max = 50 #this was set to +6 when folding was not restricted to one only
    rag_i = 51
    fold_i = 52

    ppn_words = {}
    ppn_contexts = {}

    rag_key = 0 # str is 0
    #lemma
    if "1" in folding or "5" in folding:
        rag_key = 9
    #lemma-pos
    elif "2" in folding or "6" in folding:
        rag_key = 12
    #root
    elif "3" in folding or "7" in folding:
        #else:
        rag_key = 4

    base_word = [0] * y_dimension

    context_words = []
    orig_context_corefs = {}
    for i in range(len(words)):
        word = words[i]
        key_comp = word["primary_keys"][2].split("|")

        context_word = copy.deepcopy(base_word)

        c_i = 0
        word_missing = False
        ppn_set1 = {}
        word_str = ""
        for k in key_comp:
            if "_STR" in k and "PPN_" not in k and "DEF_" not in k and "LIT_" not in k and "IDM_" not in k and k in word_to_index:
                context_word[0] = word_to_index[k]
                word_str = k

        if "string" in mode or y_dimension <= 2 or y_limit_1:
            #return False,context_word
            context_words.append(context_word)
            continue

        has_coreference_word = None
        has_coreference_context = None

        suffixes = []
        sorted_list_suffix = []
        morphes = []
        sorted_list_morph = []
        dep_h_dict = {}
        sorted_list_dep_h = []
        dep_g_dict = {}
        sorted_list_dep_g = []
        dep_h2_dict = {}
        sorted_list_dep_h2 = []
        dep_g2_dict = {}
        sorted_list_dep_g2 = []
        pos2_dict = {}
        sorted_list_pos2 = []
        concept_dict = {}
        sorted_list_concept = []
        clusters = []
        sorted_list_cluster = []
        folds = []
        sorted_list_folds = []
        sorted_list_dep_r = []
        dep_r_dict = {}

        phrase_start_found = False
        clause_start_found = False
        word_pos = ""
        for k in key_comp:
            if word[k] == "pos":
                word_pos = k

        for k in key_comp:
            if k in skip_types: continue
            if "no_prediction_i" in k: continue
            elif w != "STR" and word[k] == "mod": continue
            elif len(k) > 0 and "#" == k[0]: continue
            elif "_STR" in k and "PPN_" not in k and "IDM_" not in k and "DEF_" not in k and "LIT_" not in k: continue
            elif k not in word_to_index or k not in word:
                continue
            elif word[k] == "pos" and k in word_to_index:
                context_word[1] = word_to_index[k]
            elif k in ["Singular","Plural","morph_Number=Sing","morph_Number=Plur"] and k in word_to_index:
                context_word[2] = word_to_index[k]
            elif "next_punct" in k or "last_punct" in k:
                if sent2_active:
                    context_word[3] = word_to_index[k]
            elif "Coreference" in k:
                if coref_active and "metadata" not in word[k] and word[k] in word_to_index:
                    if "PRON" in key_comp and word[k] in ppn_words and ppn_words[word[k]] is not None:
                        has_coreference_word = ppn_words[word[k]]
                        has_coreference_context = ppn_contexts[word[k]]
            elif "_STR" not in k and ( (word[k] == "word" and context_word[4] == 0) or (word[k] == "base") ) and k in word_to_index and "lemma_" not in k:
                #pass
                context_word[4] = word_to_index[k]
            elif word[k] == "base" and "lemma_" not in k:
                pass
            elif "Longest_Suffix_" in k and k in word_to_index:
                if suf_active:
                    context_word[5] = word_to_index[k]
            elif "Last_Suffix_" in k and k in word_to_index:
                if suf_active:
                    context_word[6] = word_to_index[k]
            elif y_dimension == 8:
                continue
            elif (("Prefix_" in k) and k in word_to_index):
                if suf_active:
                    context_word[7] = word_to_index[k]
            elif ("Case" in k and k in word_to_index):
                pass
            elif k in word_to_index and "ent_type_" in k:
                context_word[ent_type_i_G] = word_to_index[k]
            elif ("Phrase_" in k or ("lemma_" in k and "poslemma_" not in k and "fold_lemma_" not in k)) and k in word_to_index:
                if (not phrase_start_found or "lemma_" in k) and not segment2_lemma_active:
                    context_word[9] = word_to_index[k]
                if k == "Phrase_Start":
                    phrase_start_found = True
            elif "poslemma_" in k and "fold_poslemma_" not in k:
                if k in word_to_index:
                    if segment_lemma_active:
                        context_word[12] = word_to_index[k]
            elif False and "Clause_" in k and k in word_to_index:
                if not phrase_start_found:
                    context_word[10] = word_to_index[k]
                if k == "Clause_Start":
                    clause_start_found = True
            elif ("_character" in k) and k in word_to_index:
                context_word[10] = word_to_index[k]
            elif "punct" in k and len(k) == 5:
                if punct_active:
                    context_word[11] = word_to_index[k]
            elif "PPN_" in k and "_STR" in k and k in word_to_index:
                if context_word[13] != 0:
                    ppn_set1[k] = 1
                context_word[13] = word_to_index[k]
                #print(k,"-------------------ppn ordered vector")
            elif "IDM_" in k and "_STR" in k and k in word_to_index:
                context_word[13] = word_to_index[k]
            elif "DEF_" in k and "_STR" in k and k in word_to_index:
                context_word[13] = word_to_index[k]
            elif "LIT_" in k and "_STR" in k and k in word_to_index:
                context_word[13] = word_to_index[k]
            elif ("Suffix_" in k) and k in word_to_index:
                if suf_active:
                    suffixes.append(k)
                    sorted_list_suffix = sorted(suffixes, key=len)
                    m = suf_i
                    for l in range(len(sorted_list_suffix)):
                        if m+l <= suf_i_max:
                                context_word[m+l] = word_to_index[sorted_list_suffix[l]]
            elif ("morph_" in k) and k in word_to_index:
                if morph_active:
                    morphes.append(k)
                    sorted_list_morph = sorted(morphes)
                    m = morph_i
                    for l in range(len(sorted_list_morph)):
                        if m+l <= morph_i_max:
                                context_word[m+l] = word_to_index[sorted_list_morph[l]]
            elif (k in dep_h or k in dep_segment_h) and k in word_to_index:
                if dep_active and deph_active:
                    if k in dep_h:
                        dep_h_dict[k] = dep_h[k]
                    elif k in dep_segment_h:
                        dep_h_dict[k] = dep_segment_h[k]
                    sorted_list_dep_h = [item[0] for item in sorted(dep_h_dict.items(), key=lambda item: item[1], reverse=True)]
                    m = dep_h_i
                    for l in range(len(sorted_list_dep_h)):
                        if m+l <= dep_h_i_max:
                            context_word[m+l] = word_to_index[sorted_list_dep_h[l]]
            elif (k in dep_g or k in dep_segment_g) and k in word_to_index:
                if dep_active and depd_active:
                    if k in dep_g:
                        dep_g_dict[k] = dep_g[k]
                    elif k in dep_segment_g:
                        dep_g_dict[k] = dep_segment_g[k]
                    sorted_list_dep_g = [item[0] for item in sorted(dep_g_dict.items(), key=lambda item: item[1], reverse=True)]
                    m = dep_g_i
                    for l in range(len(sorted_list_dep_g)):
                        if m+l <= dep_g_i_max:
                            context_word[m+l] = word_to_index[sorted_list_dep_g[l]]
            elif k in pos2 and k in word_to_index:
                if not(("verb" in k.lower() and context_word[1] != "VERB") or ("noun" in k.lower() and context_word[1] != "NOUN")):
                    pos2_dict[k] = pos2[k]
                    sorted_list_pos2 = [item[0] for item in sorted(pos2_dict.items(), key=lambda item: item[1], reverse=True)]
                    m = pos2_i
                    for l in range(len(sorted_list_pos2)):
                        if m+l <= pos2_i_max:
                            context_word[m+l] = word_to_index[sorted_list_pos2[l]]
            elif k in concept and k in word_to_index:
                if gender_active:
                    concept_dict[k] = concept[k]
                    sorted_list_concept = [item[0] for item in sorted(concept_dict.items(), key=lambda item: item[1], reverse=True)]
                    m = concept_i
                    for l in range(len(sorted_list_concept)):
                        if m+l <= concept_i_max:
                            context_word[m+l] = word_to_index[sorted_list_concept[l]]
            elif ("fold_poslemma_" in k or "fold_lemma_" in k or k in folding_next) and k in word_to_index:
                if folding_active or folding_rag_active:
                    folds.append(k)
                    sorted_list_folds = sorted(folds)
                    m = folding_i
                    for l in range(len(sorted_list_folds)):
                        if m+l <= folding_i_max:
                                context_word[m+l] = word_to_index[sorted_list_folds[l]]

            elif k in folding_markers and k in word_to_index:
                context_word[fold_i] = word_to_index[k]

            elif k in word_to_index:
                #if word[k] == "pos2":
                print("unassigned",k,word[k],word)
                #exit()
                pass
            elif not ("0" in k or "1" in k or "2" in k or "3" in k or "4" in k or "5" in k or "6" in k or "7" in k or "8" in k or "9" in k):
                word_missing = True
            #if len(sorted_list)>3:
            #    print(sorted_list,word,context_word,"get_ordered")
        ppn_replaced = 0

        pron_set = []
        for p in ["he_STR","her_STR","hers_STR","him_STR","his_STR","she_STR"]:
            if p in word_to_index:
                pron_set.append(word_to_index[p])
        if context_word[1] == 0:
            context_word[1] = word_to_index["UNK_POS"]
        if y_dimension > 8:
            if context_word[7] == 0:
                context_word[7] = word_to_index["Prefix_"]
            if context_word[pos2_i] == 0:
                context_word[pos2_i] = word_to_index["UNK_POS2"]
            if context_word[dep_g_i] == 0:
                context_word[dep_g_i] = word_to_index["UNK_DEP"]
            if context_word[dep_h_i] == 0:
                context_word[dep_h_i] = word_to_index["UNK_HDEP"]
            if genderN_active and (context_word[1] not in [word_to_index["NOUN"],word_to_index["ADJ"]] or context_word[0] in pron_set):
                context_word[concept_i] = 0
                context_word[concept_i_max] = 0
            if genderPPN_active and (context_word[1] != word_to_index["PROPN"] or context_word[0] in pron_set):
                context_word[concept_i] = 0
                context_word[concept_i_max] = 0
            if genderPRN_active and context_word[1] != word_to_index["PRON"] and context_word[0] not in pron_set:
                context_word[concept_i] = 0
                context_word[concept_i_max] = 0
            if genderNPPN_active and (context_word[1] not in [word_to_index["NOUN"],word_to_index["ADJ"],word_to_index["PROPN"]]  or context_word[0] in pron_set):
                context_word[concept_i] = 0
                context_word[concept_i_max] = 0
            if genderNPRN_active and (context_word[1] not in [word_to_index["NOUN"],word_to_index["ADJ"],word_to_index["PRON"]] and context_word[0] not in pron_set):
                context_word[concept_i] = 0
                context_word[concept_i_max] = 0

        if (coref_active or coref2_active):
            if "PROPN" in key_comp:
                ppn_words[word_str] = word
                ppn_contexts[word_str] = context_word
            else:
                ppn_words[word_str] = None
                ppn_contexts[word_str] = None

        if pos_active and has_coreference_word is not None and (coref_active or coref2_active):
            orig_context_corefs[len(context_words)] = context_word
            if coref2_active:
                lemmah = ""
                poslemmah = ""
                for h in has_coreference_word:
                    if "lemma_" in h and "poslemma" not in h and "fold_lemma" not in h and h in word_to_index:
                        lemmah = h
                    if "poslemma_" in h and "fold_pos" not in h and h in word_to_index:
                        poslemmah = h
                if lemmah != "":
                    context_word[9] = word_to_index[lemmah]
                    if poslemmah != "":
                        context_word[12] = word_to_index[poslemmah]
                    if gender_prop_active:
                        if "Concept_male" in word:
                            has_coreference_word["Concept_male"] = "metadata"
                            has_coreference_context[concept_i_max] = word_to_index["Concept_male"]
                        if "Concept_female" in word:
                            has_coreference_word["Concept_female"] = "metadata"
                            has_coreference_context[concept_i_max] = word_to_index["Concept_female"]
            else:
                context_word = has_coreference_context
            context_word[11] = word_to_index["Coreference"]

        context_words.append(context_word)

    return context_words, orig_context_corefs

def mod_for_hypoth(i,j1,word,context_word,word_to_index,mod_freq,mod_freq0,mod_freq2,mod_freq0_rank,mod_freq2_rank,mod_freq0_std,mod_freq2_std, mod_freq0_avg, mod_freq2_avg,num_mod_added):
                    zero_features = {}
                    zero_features_neg = {}

                    pos_found = ""
                    for w in word:
                        if w != "STR" and word[w] == "mod":
                            fields = w.split("_")
                            if int(fields[2]) == j1:
                                if fields[0] == "0":
                                    zero_features[fields[1]+fields[3]] = 1
                                    if fields[1] == "-":
                                        zero_features_neg[fields[3]] = 1
                                        if fields[3] in word_to_index and word_to_index[fields[3]] in context_word:
                                            if fields[3] in postags:
                                                pos_found = fields[3]
                                            c_i = context_word.index(word_to_index[fields[3]])
                                            context_word[c_i] = 0
                    if pos_found != "":
                        for w in word:
                            if "poslemma" in w and pos_found in w:
                                zero_features_neg[w] = 1
                                if w in word_to_index: #  and word_to_index[w] in context_word:
                                    context_word[12] = 0

                    return zero_features_neg, {}


def generate_training_data3(corpus_file, corpus, word_to_index, label_to_index, words_freq, words_freq2_override, mod_freq, context_size,skip_size, X_dimension, y_dimension, mode, valid_indexes, max_position_embeddings, rag_position_embeddings, previous_predictions, output_strings, rag_blanket, batch_size, corpus_effective_len, loader, rag_mode=False):
    data = []
    data.append([])
    data.append([])
    data.append([])
    data.append([])
    data.append([])
    data.append([])
    rag_blanket2 = []
    rag_blanket_len = 0
    trgt_cache = []
    ctxt_cache = []
    blkt_cache = []
    prediction_map = {}
    freq_total = 1
    filtered_count = 0
    filtered_count_unk = 0
    filtered1_count = 0
    pop_count = 0
    output_strings2 = {}
    print("start gen3 -->",corpus_file,len(corpus),len(word_to_index),rag_mode,max_position_embeddings,rag_position_embeddings,context_size,"<-- start gen3")
    index_to_word = {}
    for w in word_to_index:
        index_to_word[word_to_index[w]] = w

    this_folding_active = folding_active
    this_folding_echo_active = folding_echo_active
    this_folding_bidirectional_active = folding_bidirectional_active
    this_folding_once_active = folding_once_active
    if rag_mode:
        this_folding_active = folding_rag_active
        this_folding_echo_active = folding_rag_echo_active
        this_folding_bidirectional_active = folding_rag_bidirectional_active
        this_folding_once_active = folding_rag_once_active


    index_features = {}
    feature_labels = {}
    min_iii = max_iii = 0
    mod_freq0 = {}
    mod_freq2 = {}
    mod_freq0_rank = {}
    mod_freq2_rank = {}
    mod_freq0_std = 0
    mod_freq2_std = 0
    mod_freq0_avg = 0
    mod_freq2_avg = 0
    num_mod_freq0 = num_mod_freq2 = num_mod_added = 0

    i = max_position_embeddings + 1
    start_i = i
    end = len(corpus) - max_position_embeddings - 23

    pickle_trgt_word_vectors = None
    pickle_ctxt_word_vectors = None
    if enable_pickle:
        if not rag_mode:
            #print("open start_ctxt",mode+"_start_ctxt_word_v.pickle")
            pickle_trgt_word_vectors = open(mode+"_start_trgt_word_v.pickle", "wb")
            pickle_ctxt_word_vectors = open(mode+"_start_ctxt_word_v.pickle", "wb")
            if rag_blanket_active:
                pickle_blkt_word_vectors = open(mode+"_start_blkt_word_v.pickle", "wb")
        else:
            #print("open rag_ctxt",mode+"_rag_ctxt_word_v.pickle")
            print("init rag file=============",mode+"_rag_word_v.pickle")
            pickle_ctxt_word_vectors = open(mode+"_rag_word_v.pickle", "wb")

    target_label = ""
    target_label_index = -1
    ascii_increment = 4
    label_at = -1
    ascii_at = -1
    last_label_at = -1
    last_ascii_at = -1
    label_index = 0
    zero_shot_union = {}
    zero_shot_unions = {}
    test_id = 0
    word_prediction = False
    x_limit_1 = False
    if "zero_shotf" in mode or "zero_shotcf" in mode or "zero_shotmf" in mode or "fine_tune0f" in mode or "fine_tune1cf" in mode or "fine_tune1f" in mode or "reg_test0f" in mode or "reg_test0mf" in mode or "reg_test0cf" in mode or "reg_test1cf" in mode or "reg_test1f" in mode or x_dimension_1:
        x_limit_1 = True
    has_words_count = not_has_words_count = 0
    for w in ["zero_shotcw","zero_shotmw","zero_shotw","fine_tune0w","fine_tune0cw","fine_tune0mw","reg_test0w","reg_test0cw","reg_test0mw"]:
        if w in mode:
            word_prediction = True
    words_freq2 = {}
    ordered_words, orig_context_corefs = get_ordered_feature_vectors(corpus,X_dimension,x_limit_1,word_to_index,rag_mode,mode)

    sent_markers = []
    phrase_markers = []
    phrase_markers_pos = []
    phrase_markers_neg = []
    phrase_markers_sets = []
    word_markers_pos = []
    word_markers_neg = []
    sent_phrase_totals = []
    phrase_word_totals = []
    phrase_contents = []
    sent_i = 1
    phrase_i = 1
    phrase_j = 0
    last_sent_start = i
    word_j = 0
    last_phrase_start = i
    below_avg_freq = above_avg_freq = below_prd_freq = above_prd_freq = below_avg_num = above_avg_num = 0
    norm_word_seq = {}
    word_seq_repeats = 0

    sent_breaks = ["last_punct_EXCLAMATION_MARK","last_punct_PERIOD","last_punct_QUESTION_MARK"]
    phrase_breaks = ["last_punct_CLOSE_PARENTHESIS","last_punct_COLON","last_punct_DASH_DASH","last_punct_DASH","last_punct_DOUBLE_QUOTE","last_punct_OPEN_PARENTHESIS","last_punct_PERIOD_PERIOD","last_punct_PERIOD_PERIOD_PERIOD","last_punct_SEMI_COLON","last_punct_SINGLE_QUOTE"]

    sent_num = 1
    phrase_num = 1

    i = start_i
    while i < end:
        word = corpus[i]
        for w in word:
            if w in sent_breaks:
                sent_num += 1
                phrase_num += 1
            if w in phrase_breaks:
                phrase_num += 1
        i+=1

    if not rag_mode and (sent_active or sent3_active or clause_active or clause2_active or missing_active):
        sent_markers = [0] * (end+max_position_embeddings+5)
        phrase_markers = [0] * (end+max_position_embeddings+5)
        phrase_markers_sets = [["","",""]] * (end+max_position_embeddings+5)
        phrase_markers_pos = [0] * (end+max_position_embeddings+5)
        sent_phrase_totals = [0] * (sent_num+5)
        phrase_word_totals = [0] * (phrase_num+5)
        phrase_contents = []
        for p in range(phrase_num+5):
            phrase_contents.append({})

        word_markers_pos = [0] * (end+max_position_embeddings+5)

    phrase_min = phrase_max * -1
    word_min = word_max * -1

    pos_heirarchy = ["VERB", "PREP", "NOUN", "PRON", "PROPN"]
    clause_deps = ["advcl","appos","ccomp","parataxis","pcomp","prepc","ROOT","xcomp"]
    phrase_tmp = {}
    last_phrase_tmp = {}
    phrase_set = [""] * 3
    last_phrase_set = [""] * 3
    coref_index = -1
    if coref_active or coref2_active:
        if "Coreference" in word_to_index:
            coref_index = word_to_index["Coreference"]
    compounds = {}
    word_indexes = {}
    id_indexes = {}

    i = start_i
    while i < end:
        word = corpus[i]

        if word["ID"] not in id_indexes:
            id_indexes[word["ID"]] = []
        id_indexes[word["ID"]].append(i)

        if word["STR"] not in word_indexes:
            word_indexes[word["STR"]] = []
        word_indexes[word["STR"]].append(i)
        for j in pos_heirarchy:
            if j in word:
                phrase_tmp[j] = 1
        for j in clause_deps:
            if j in word:
                phrase_tmp[j] = 1
            if "h"+j in word:
                phrase_tmp["h"+j] = 1
        if compound_active and ("compound" in word or "hcompound" in word or "PROPN" in word):
            has_propn = False
            for j in range(8):
                if i+j >= end: break
                word2 = corpus[i+j]
                if "PROPN" in word2:
                    has_propn = True
                    break
                if "compound" not in word2 and "hcompound" not in word2: break
            for j in range(-1,-9,-1):
                if i+j <= start_i: break
                word2 = corpus[i+j]
                if "PROPN" in word2:
                    has_propn = True
                    break
                if "compound" not in word2 and "hcompound" not in word2: break
            if has_propn:
                compounds[i] = word["STR"]
        if False and "propnreplstr" in mode and ("string" in mode or "parse" in mode):
            key_comp = word["primary_keys"][2]
        else:
            key_comp = word["primary_keys"][2].split("|")

        for k in key_comp:
            if k in word_to_index:
                if word_prediction or y_dimension_1:
                    if k in word and word[k] != "word":
                        continue
        word_primary_key0 = word["primary_keys"][1]
        if word_primary_key0 not in words_freq2:
            words_freq2[word_primary_key0] = 0
        words_freq2[word_primary_key0] += 1

        if not rag_mode and (sent_active or sent3_active or clause_active or clause2_active or missing_active):
            has_sent_break = False
            for j in sent_breaks:
                if j in word:
                    has_sent_break = True
            if has_sent_break:
                sent_i += 1
                phrase_i += 1
                phrase_j = 0
                word_j = 0
                pos_set = ""
                for j in pos_heirarchy:
                    if j in phrase_tmp:
                        pos_set = j
                        break
                phrase_set[0] = pos_set
                cls_dep = ""
                last_cls_dep = ""
                for j in clause_deps:
                    if j[0] == "h":
                        k = j[1:]
                        if k in last_phrase_tmp:
                            last_cls_dep = k
                    else:
                        if "h"+j in last_phrase_tmp:
                            cls_dep = j
                if cls_dep != "":
                    phrase_set[1] = cls_dep
                    last_phrase_set[2] = "h"+cls_dep
                if last_phrase_set[1] == "" and last_cls_dep != "":
                    last_phrase_set[1] = last_cls_dep
                    phrase_set[2] = "h"+last_cls_dep
                last_sent_start = i
                last_phrase_start = i
                last_phrase_tmp = phrase_tmp
                phrase_tmp = {}
                last_phrase_set = phrase_set
                phrase_set = [""] * 3
            else:
                has_phrase_break = False
                for j in phrase_breaks:
                    if j in word:
                        has_phrase_break = True
                if has_phrase_break:
                    phrase_i += 1
                    phrase_j += 1
                    word_j = 0

                    pos_set = ""
                    for j in pos_heirarchy:
                        if j in phrase_tmp:
                            pos_set = j
                            break
                    phrase_set[0] = pos_set
                    cls_dep = ""
                    last_cls_dep = ""
                    for j in clause_deps:
                        if j[0] == "h":
                            k = j[1:]
                            if k in last_phrase_tmp:
                                last_cls_dep = k
                        else:
                            if "h"+j in last_phrase_tmp:
                                cls_dep = j
                    if cls_dep != "":
                        phrase_set[1] = cls_dep
                        last_phrase_set[2] = "h"+cls_dep
                    if last_phrase_set[1] == "" and last_cls_dep != "":
                        last_phrase_set[1] = last_cls_dep
                        phrase_set[2] = "h"+last_cls_dep

                    last_phrase_start = i
                    last_phrase_tmp = phrase_tmp
                    last_phrase_set = phrase_set
                    phrase_tmp = {}
                    phrase_set = [""] * 3
            sent_markers[i] = sent_i
            phrase_markers[i] = phrase_i
            sent_phrase_totals[sent_i] = phrase_j
            phrase_word_totals[phrase_i] = word_j
            phrase_markers_pos[i] = phrase_j
            phrase_markers_sets[i] = phrase_set
            word_markers_pos[i] = word_j
            word_j += 1

        i+=1

    print("\n",list(words_freq2.keys())[:16],"words_freq\n")
    freq_total = 0
    freq_total_valid = 0
    uniq_words_valid = {}
    if rag_mode and words_freq2_override is not None:
        print("--> override words_freq2 <--")
        words_freq2 = words_freq2_override
    for w in words_freq2:
        if "UNK_STR" in w or "NUM_STR" in w: continue
        if "_STR" in w:
            freq_total += words_freq2[w]
            if w in word_to_index:
                freq_total_valid += words_freq2[w]
                uniq_words_valid[w] = 1
    threshold = 0.5
    avg_freq_valid = freq_total_valid / (len(uniq_words_valid))
    if len(words_freq2) > 0:
        threshold = 1 / (len(words_freq2) / 2)
    print("X_dim",X_dimension,x_dimension_1,"y_dim",y_dimension,y_dimension_1,"len corpus",len(corpus),"len freq",len(words_freq2),"freq_total",freq_total,"freq_total_valid",freq_total_valid,"len uniq valid",len(uniq_words_valid)," training_Data3 pre subsampling")

    word_total = 0
    if enable_memmap:
        if "newsgroup" in corpus_file or "_n20_" in corpus_file:
            in_label = False
            i = max_position_embeddings + 3
            while i < end:
                word = corpus[i]
                if corpus[i]["STR"] == "label":
                    in_label = True
                if corpus[i]["STR"] == "label":
                    in_label = False
                if "UNK_STR" in word_primary_key0 or "NUM_STR" in word_primary_key0 or "last_punct" in corpus[i]["STR"]:
                    i+=1
                    continue
                word_str = word["STR"]
                if "W" in folding and word_str not in output_strings:
                    i+=1
                    continue
                if not in_label:
                    word_total += 1
                i+=1
        else:
            i = max_position_embeddings + 3
            while i < end:
                word = corpus[i]
                if "UNK_STR" in word_primary_key0 or "NUM_STR" in word_primary_key0 or "last_punct" in corpus[i]["STR"]:
                    i+=1
                    continue
                word_str = word["STR"]
                if "W" in folding and word_str not in output_strings:
                    i+=1
                    continue
                word_total += 1
                i+=1
        print(word_total,"word_total gen data3")

    fi_memmap = None
    if enable_memmap:
        shape = (word_total,max_position_embeddings,X_dimension)
        print(shape,"shape")
        fi_memmap = np.memmap(mode+"_start_ctxt_word_v.dat", dtype=np.int32, mode='w+', shape=shape)

    labels_at = []
    asciis_at = []
    label_ascii_pairs = []
    if "newsgroup" in corpus_file or "_n20_" in corpus_file:
        i = max_position_embeddings + 3
        while i < end:
            if corpus[i]["STR"] == "label":
                labels_at.append(i)
            if corpus[i]["STR"] == "ascii":
                asciis_at.append(i)
            i += 1
        label_ascii_pairs = []
        for i in range(len(labels_at)):
            for j in range(len(asciis_at)):
                if asciis_at[j] > labels_at[i]:
                    if asciis_at[j] - labels_at[i] < 12:
                        label_ascii_pairs.append( [ labels_at[i] , asciis_at[j] ] )
        print(len(label_ascii_pairs),"len label_ascii pairs")
        print(label_ascii_pairs[-5:])

        label_ascii_i = 0
        label_at = label_ascii_pairs[label_ascii_i][0]
        ascii_at = label_ascii_pairs[label_ascii_i][1]

    del labels_at
    del asciis_at
    offsets = []
    for o in range(3):
        offsets.append([-1,-1])

    i = max_position_embeddings + 1
    slice_i = 0
    target_i = 0
    prediction_i = 0
    target_j = 0
    print("start data3 folding_full",folding_full,len(previous_predictions),corpus_effective_len,len(output_strings))
    while i < end:
        if "testset" not in mode and len(previous_predictions) > 0 and "W" in folding_full and has_words_count > len(previous_predictions):
            print("break at",has_words_count,"1")
            break
        if "testset" not in mode and corpus_effective_len > 0 and "W" in folding_full and has_words_count > corpus_effective_len:
            print("break at",has_words_count,"2")
            break
        sent_i = 1
        phrase_i = 1
        if not rag_mode and (sent_active or sent3_active):
            sent_i = sent_markers[i]
            phrase_i = phrase_markers[i]

        if "newsgroup" in corpus_file or "_n20_" in corpus_file:
            if i >= label_at - 1 or target_label == "":
                label_ascii_i += 1
                if label_ascii_i == len(label_ascii_pairs):
                    break
                last_label_at = label_at
                last_ascii_at = ascii_at
                label_at = label_ascii_pairs[label_ascii_i][0]
                ascii_at = label_ascii_pairs[label_ascii_i][1]
                if True:
                    zero_shot_union = {}
                    for f in corpus[label_at+1]:
                        if corpus[label_at+1][f] == "word":
                            target_label = f
                            target_label_index = label_at+1
                            test_id = target_label_index
                    k = label_at+1
                    while k < ascii_at:
                        for f in corpus[k]:
                            if f not in ["STR","ID","primary_keys"]:
                                zero_shot_union[f] = corpus[k][f]
                        k+=1
            if label_to_index.get(target_label) == None:
                label_to_index.update ( {target_label : label_index})
                label_index += 1
            if i <= last_ascii_at:
                i = last_ascii_at + 1 + int(skip_size * np.random.uniform(0.0,0.75))
                continue
            if corpus[i]["STR"] in ["UNK_STR","UNK"] or "UNK" in corpus[i]:
                i += 1
                continue

        if i > label_at and label_at != -1:
            print(i,end,label_at,ascii_at,last_label_at,last_ascii_at,label_ascii_pairs[-5:],"ERROR_END")
            break

        word = corpus[i]
        blanket_words = {}
        word_primary_key0 = word["primary_keys"][1]

        pop_count += 1

        if "UNK_STR" in word_primary_key0 or "NUM_STR" in word_primary_key0 or "last_punct" in corpus[i]["STR"]:
            i+=1
            continue

        word_str = word["STR"]
        word_id = word["ID"]
        if "W" in folding and word_str not in output_strings:
            i+=1
            continue
        output_strings2[word_str] = 1
        key_comp = word["primary_keys"][2].split("|")

        word_deps = {}
        word_pos = ""
        word_lemma = ""
        for w in word:
            feature_labels[w] = word[w]
            if word[w] == "pos":
                word_pos = w
            if len(w) > 6 and w[:6] == "lemma_" and "_STR" not in w:
                word_lemma = w
            if clause2_active:
                if w[0] == "h":
                    if w in clause_deps:
                        word_deps[w] = 1
                    if w[1:] in clause_deps:
                        word_deps[w[1:]] = 1
                else:
                    if w in clause_deps:
                        word_deps[w] = 1
                    if "h"+w in clause_deps:
                        word_deps["h"+w] = 1

        invalid_context_from_word_lemma = {}
        invalid_context_from_word_lemma = add_dict_if_dict("fold_"+word_str,word_to_index,invalid_context_from_word_lemma)
        invalid_context_from_word_lemma = add_dict_if_dict("fold_"+word_lemma,word_to_index,invalid_context_from_word_lemma)
        invalid_context_from_word_lemma = add_dict_if_dict("fold_"+"pos"+word_lemma+"_"+word_pos,word_to_index,invalid_context_from_word_lemma)

        filtered_count_unk += 1
        filtered_count += 1

        has_words = len(word["STR"]) >= 2 and word["STR"].isalpha()

        context_words = []
        context_words2 = []
        remove_contexts_features = []

        # in tuning and testing there can be words that are not recognized from the pretrained model
        valid_word = False
        key_comp_word = key_comp
        for k in key_comp:
            if k in word_to_index:
                #print(key_comp,k,word)
                if k not in word:
                    print(word,k,"ERROR missing k in word")
                if k in word and word[k] == "word":
                    valid_word = True
        # when predicting labels, it doesn't matter that the center word is out of vocabulary
        if not word_prediction:
            valid_word = True

        # transformer uses masked model
        if "transformer" in mode and valid_word:
            target_word= {}
            target_id = word["ID"]
            context_ids = []
            target_ordered = []
            mask_index = 0
            zero_features = {}
            filtered1_count += 1
            if ("testset" in mode or "tuning" in mode) and not word_prediction:
                target_word= {}
                test_label = ""
                if zero_shot_forced or "zero_shot" in mode:
                    for k in zero_shot_union:
                        if k in word_to_index:
                            if k in labels_test:
                                test_label = k
                            if "zero_shotf" in mode or "zero_shotcf" in mode or "zero_shotmf" in mode:
                                if zero_shot_union[k] != "word": continue
                            target_word[word_to_index[k]] = 1.0
                    if test_label not in zero_shot_unions:
                        zero_shot_unions[test_label] = {}
                    for t in target_word:
                        zero_shot_unions[test_label][t] = 1
                #overwrite if not in zero shot mode
                if not zero_shot_forced and "zero_shot" not in mode:
                    target_word[label_to_index.get(target_label)] = 1.0
                    if test_label not in zero_shot_unions:
                        zero_shot_unions[test_label] = {}
                    for t in target_word:
                        zero_shot_unions[test_label][t] = 1
            else:
                target_word= {}
                for k in key_comp_word:
                    if k in word_to_index:
                        if True or "zero_shotw" in mode or "zero_shotcw" in mode or "zero_shotmw" in mode or "reg_test0w" in mode or "reg_test0cw" in mode or "reg_test0mw" in mode:
                            if is_target_feature(k,output_strings2):
                                target_word[word_to_index[k]] = 1.0
                        else:
                            target_word[word_to_index[k]] = 1.0
                if len(target_word.keys()) == 0:
                    target_word[word_to_index["UNK"]] = 1.0
                target_ordered = ordered_words[i]


            cls_word = [0] * X_dimension
            cls_word[0] = word_to_index["CLS"]
            context_words.append(cls_word)
            context_ids.append("-1")
            remove_contexts_features.append([])


            if not rag_mode:
                r2 = rag_position_embeddings
                if ragV_active or ragU_active:
                    r2 = 0
                rand_win = np.random.randint(int(-16.0 * 0.33),int(16.0 * 0.33))
                window_offset_pre = rand_win + int(window_sizeG/2)
                window_offset_post = window_sizeG - 3 - window_offset_pre # - r2
                if i < start_i + 3:
                    print(rand_win,window_offset_pre,window_offset_post,window_sizeG,max_position_embeddings,"data3 windows ctxt")
            else: # rag_mode
                window_offset_pre = int(rag_position_embeddings / 2) - 1
                window_offset_post = int(rag_position_embeddings / 2)
                if window_offset_pre+window_offset_post+2 > rag_position_embeddings:
                  window_offset_post -= 1
                if window_offset_pre+window_offset_post+2 > rag_position_embeddings:
                  window_offset_pre -= 1
                if window_offset_pre+window_offset_post+2 > rag_position_embeddings:
                  window_offset_post -= 1
                if i < start_i + 3:
                    print(window_offset_pre,window_offset_post,window_sizeG,max_position_embeddings,"data3 windows rag")

            start = window_offset_pre
            start1 = start
            if True or "pretrain" not in mode:
                for j in range(start1,-1,-1):
                    if corpus[i - j - 1]["STR"] == "ascii":
                        start1 = j-1
            if not rag_mode and not ragV_active and not ragU_active:
                if i < start_i + 3:
                    print(rag_position_embeddings,"data3 rag pre")

                for j in range(rag_position_embeddings):
                    context_word = [0] * X_dimension
                    context_words.append(context_word)
                    context_ids.append("-1")
                    remove_contexts_features.append([])

            for j in range(start,start1,-1):
                context_word = [0] * X_dimension
                context_words.append(context_word)
                context_ids.append("-1")
                remove_contexts_features.append([])

            end_pre = -1

            for j in range(start1,end_pre,-1):
                has_words = has_words or (len(corpus[i - j - 1]["STR"]) >= 2 and corpus[i - j - 1]["STR"].isalpha())
                context_word = list(ordered_words[i - j - 1])
                if coref_index != -1 and coref_index in context_word and word["STR"] in word_to_index and context_word[0] == word_to_index[word["STR"]]:
                    context_word = list(orig_context_corefs[i - j - 1])

                if not rag_mode and hypoth_active and corpus[i - j - 1]["STR"] not in ["UNK","NUM"]:
                    zero_features_neg, features_added = mod_for_hypoth(i,-j-1,corpus[i - j - 1],context_word,word_to_index,mod_freq,mod_freq0,mod_freq2,mod_freq0_rank, mod_freq2_rank, mod_freq0_std, mod_freq2_std, mod_freq0_avg, mod_freq2_avg, num_mod_added)
                    for z in zero_features_neg:
                        zero_features[z] = 1

                remove_features = []
                for wj in corpus[i-j-1]:
                    if "-" in corpus[i-j-1][wj]:
                        fields = corpus[i-j-1][wj].split("-")
                        if word_id == fields[1]:
                            remove_features.append(wj)

                context_words.append(context_word)
                context_ids.append(corpus[i - j - 1]["ID"])
                remove_contexts_features.append(remove_features)

            if not rag_mode:
                if True:
                    mask_word = [0] * X_dimension
                    mask_word[0] = word_to_index["MASK"]

                    target_i += 1
                    if "no_prediction_i" not in word:
                        prediction_i += 1

                mask_index = len(context_words)
                context_words.append(mask_word)
                context_ids.append("-1")
                remove_contexts_features.append([])

            context_size3 = window_offset_post
            if context_size3 < 0: context_size3 = 0

            for j in range(context_size3):
                if len(context_words) >= max_position_embeddings: break
                if corpus[i+j+1]["STR"] == "label":
                    break
                has_words = has_words or (len(corpus[i + j + 1]["STR"]) >= 2 and corpus[i + j + 1]["STR"].isalpha())
                context_word = list(ordered_words[i + j + 1])
                if coref_index != -1 and coref_index in context_word and word["STR"] in word_to_index and context_word[0] == word_to_index[word["STR"]]:
                    #context_word = copy.deepcopy(orig_context_corefs[i + j + 1])
                    context_word = list(orig_context_corefs[i + j + 1])

                if not rag_mode and hypoth_active and corpus[i + j + 1]["STR"] not in ["UNK","NUM"]:
                    zero_features_neg, features_added = mod_for_hypoth(i,j+1,corpus[i + j + 1],context_word,word_to_index,mod_freq, mod_freq0,mod_freq2, mod_freq0_rank, mod_freq2_rank, mod_freq0_std, mod_freq2_std, mod_freq0_avg, mod_freq2_avg, num_mod_added)
                    for z in zero_features_neg:
                        zero_features[z] = 1

                remove_features = []
                for wj in corpus[i+j+1]:
                    if "-" in corpus[i+j+1][wj]:
                        fields = corpus[i+j+1][wj].split("-")
                        #print("",fields[1],wj,corpus[i+j+1][wj])
                        if word_id == fields[1]:
                            #print(" ",wj)
                            remove_features.append(wj)

                context_words.append(context_word)
                context_ids.append(corpus[i + j + 1]["ID"])
                remove_contexts_features.append(remove_features)
            while len(context_words) < max_position_embeddings: #((context_size * 2) + 8): # (context_size + upto 2) + 1 + padding
                context_words.append([0]*X_dimension)
                context_ids.append("-1")
                remove_contexts_features.append([])

            if i < start_i + 3:
                print(i,len(context_words),"data3 ctxt")
            if "sqs" in mode or "sqx" in mode or "fqx" in mode or "fx" in mode or "mqx" in mode:
                for iii,c_word in enumerate(context_words):
                    #add position encoding as a regular feature
                    iiii = iii - mask_index
                    word2 = corpus[i + iiii]
                    if not rag_mode:
                        c_word[-1] = word_to_index["POSITION_"+str(iii)]
                    for c in c_word:
                        if c not in index_features:
                            index_features[c] = 0
                        index_features[c] += 1
                    #remove dependency components that link to target word
                    if not rag_mode and iiii != 0 and iii != 0:
                        for w2 in word2:
                            if w2 in word_to_index:
                                val = word2[w2]
                                t = word_to_index[w2]
                                if "metadata-" in val and t in c_word:
                                    fields = val.split("-")
                                    for v1 in fields[1:]:
                                        if v1 == word["ID"]:
                                            t_i = c_word.index(t)
                                            c_word[t_i] = 0
                                            #print("FOUND2")
                    #remove folded features that link to target word
                    for c in range(len(c_word)):
                        if c_word[c] in invalid_context_from_word_lemma:
                            #print(c,c_word[c],"context word in invalid_context_from_word_lemma",invalid_context_from_word_lemma)
                            #exit()
                            c_word[c] = 0
                    #remove features that link to target word through remove_contexts_features
                    for r in remove_contexts_features[iii]:
                        if r in word_to_index and word_to_index[r] in c_word:
                            ind1 = c_word.index(word_to_index[r])
                            c_word[ind1] = 0

                if dep_missing_active:
                    dep_words = {}
                    if i < end - 1:
                        word2 = corpus[i+1]
                        if word2["STR"] in word_to_index:
                            has_dep = False
                            for w in word2:
                                if w in dep_g:
                                    has_dep = True
                                    break
                            if not has_dep:
                                dep_words[word2["STR"]] = -1
                    if True:
                        word2 = corpus[i-1]
                        if word2["STR"] in word_to_index:
                            has_dep = False
                            for w in word2:
                                if w in dep_g:
                                    has_dep = True
                                    break
                            if not has_dep:
                                dep_words[word2["STR"]] = 1
                    dep_covered = {}
                    dep_links = {}
                    if len(dep_words) > 0:
                        for iii,c_word in enumerate(context_words):
                            dep_covered[c_word[0]] = 1
                    for word2_str in dep_words:
                        closest_jj = {}
                        for jj in word_indexes[word2_str]:
                            if abs(i - jj) < 100 and abs(i - jj) > 8:
                                closest_jj[jj] = abs(i - jj)
                        for jjj in closest_jj:
                            closeness = closest_jj[jjj]
                            word3 = corpus[jjj]
                            for w in word3:
                                if w in dep_g:
                                    if "-" in word3[w]:
                                        fields_id = word3[w].split("-")
                                        for id1f in fields_id[1:]:
                                            if id1f in id_indexes:
                                                for id_i in id_indexes[id1f]:
                                                    if id_i == i: continue
                                                    if abs(id_i - jjj) == 1:
                                                        word4 = corpus[id_i]
                                                        if "UNK" not in word4["STR"] and "NUM" not in word4["STR"] and word4["STR"] in word_to_index and word_to_index[word4["STR"]] not in dep_covered:
                                                            dep_links[word_to_index[word4["STR"]]] = closeness
                    if len(dep_links) > 0:
                        dep_sort = sorted(dep_links, key=dep_links.get)
                        len_d = len(dep_sort)
                        if len_d > 3:
                            len_d = 3
                        for d in range(len_d):
                            context_words[0][d+1] = dep_sort[d]

                if not rag_mode and pos_active:
                    for iii in range(mask_index-1,0,-1):
                        iiii = iii - mask_index
                        context_words[mask_index+iiii][-2] = word_to_index["RELPOSITION_"+str(iiii)]
                    for iii in range(mask_index+1,len(context_words)):
                        iiii = iii - mask_index
                        context_words[mask_index+iiii][-2] = word_to_index["RELPOSITION_"+str(iiii)]
                    #VERB ADV ADP ADJ PART NOUN PRON PROPN CCONJ DET AUX
                    if pos_active2:
                        pos_counters = {}
                        neg_pos_counters = {}
                        pos_index = {}
                        pos_positions = {}
                        for pos in ["VERB","ADV","ADP","ADJ","PART","NOUN","PRON","PROPN","CCONJ","DET","AUX"]:
                            pos_counters[pos] = 0
                            neg_pos_counters[pos] = 0
                            pos_index[pos] = pos
                            pos_positions[pos] = {}
                        #for pos in ["VERB","ADV","ADP","ADJ","PART","NOUN","PRON","PROPN","CCONJ","DET","AUX"]:
                        #    for i in range(1,int(max_position_embeddings/4)):
                        #

                        for iii in range(mask_index-1,0,-1):
                            iiii = iii - mask_index
                            pos = index_to_word[context_words[mask_index+iiii][1]]
                            if pos in neg_pos_counters:
                                neg_pos_counters[pos] += 1
                                pos_positions[pos][iii] = 1
                                if neg_pos_counters[pos] >= int(max_position_embeddings/4): continue
                                context_words[mask_index+iiii][-3] = word_to_index[pos_index[pos]+"_RELPOSITION_-"+str(neg_pos_counters[pos])]
                                #print(pos_index[pos]+"_RELPOSITION_-"+str(neg_pos_counters[pos]),word_to_index[pos_index[pos]+"_RELPOSITION_-"+str(neg_pos_counters[pos])])
                        for iii in range(mask_index+1,len(context_words)):
                            iiii = iii - mask_index
                            pos = index_to_word[context_words[mask_index+iiii][1]]
                            if pos in pos_counters:
                                pos_counters[pos] += 1
                                pos_positions[pos][iii] = 1
                                if pos_counters[pos] >= int(max_position_embeddings/4): continue
                                context_words[mask_index+iiii][-3] = word_to_index[pos_index[pos]+"_RELPOSITION_"+str(pos_counters[pos])]
                        if pos_pointers_active:
                            for pos in pos_positions:
                                pos_str = pos #index_to_word[pos]
                                for iii1 in pos_positions[pos]:
                                    for iii2 in pos_positions[pos]:
                                        if iii1 == iii2: continue
                                        if context_words[iii2][-14] == 0:
                                            context_words[iii2][-14] = word_to_index[pos_str+"_POINT_POSITION_"+str(iii1)]
                                        elif context_words[iii2][-15] == 0:
                                            context_words[iii2][-15] = word_to_index[pos_str+"_POINT_POSITION_"+str(iii1)]
                                        elif context_words[iii2][-16] == 0:
                                            context_words[iii2][-16] = word_to_index[pos_str+"_POINT_POSITION_"+str(iii1)]

                    if dep_pointers_active:
                        for iii,c_word in enumerate(context_words):
                            id1 = context_ids[iii]
                            if id1 == "-1":
                                continue
                            iiii = iii - mask_index
                            word2 = corpus[i + iiii]
                            for w in word2:
                                if w in dep_g or (w[0] != "h"):
                                    if "-" in word2[w]:
                                        ids = word2[w].split("-")
                                        id2 = ids[1]
                                        if ((w[0] != "h")) and len(ids) > 2:
                                            id2 = ids[2]
                                        if id2 in context_ids:
                                            index2 = context_ids.index(id2)
                                            c_word[-10] = word_to_index["DEP_POINT_POSITION_"+str(index2)]
                                if w in dep_h or (w[0] == "h"):
                                    if "-" in word2[w]:
                                        ids = word2[w].split("-")
                                        id2 = ids[1]
                                        if ((w[0] == "h")) and len(ids) > 2:
                                            id2 = ids[2]
                                        if id2 in context_ids:
                                            index2 = context_ids.index(id2)
                                            if c_word[-11] == 0:
                                                c_word[-11] = word_to_index["HEAD_POINT_POSITION_"+str(index2)]
                                            elif c_word[-12] == 0:
                                                c_word[-12] = word_to_index["HEAD_POINT_POSITION_"+str(index2)]
                                            elif c_word[-13] == 0:
                                                c_word[-13] = word_to_index["HEAD_POINT_POSITION_"+str(index2)]

                    if rep_pointers_active:
                        lemmapos_at = {}
                        for iii,c_word in enumerate(context_words):
                            if c_word[0] == 0: continue
                            if c_word[12] not in lemmapos_at:
                                lemmapos_at[c_word[12]] = []
                            lemmapos_at[c_word[12]].append(iii)
                        for l in lemmapos_at:
                            if len(lemmapos_at[l]) > 1 and lemmapos_at[l] != "":
                                for l1 in lemmapos_at[l]:
                                    for l2 in lemmapos_at[l]:
                                        if l1 == l2: continue
                                        if context_words[l2][-17] == 0:
                                            context_words[l2][-17] = word_to_index["REP_POINT_POSITION_"+str(l1)]
                                        elif context_words[l2][-18] == 0:
                                            context_words[l2][-18] = word_to_index["REP_POINT_POSITION_"+str(l1)]
                                        if rep2_pointers_active:
                                            k = l1 - mask_index
                                            if k == 0 or abs(k) > 5:
                                                continue
                                            for j in range(1,6):
                                                if l2 + j >= max_position_embeddings or l1 + j >= max_position_embeddings or l1 + j == mask_index:
                                                    continue
                                                if context_words[l2+j][-19] == 0:
                                                    context_words[l2+j][-19] = word_to_index["REP_REL_"+str(1*j)+"_REL_POINT_POSITION_"+str(k)]
                                                elif context_words[l2+j][-20] == 0:
                                                    context_words[l2+j][-20] = word_to_index["REP_REL_"+str(1*j)+"_REL_POINT_POSITION_"+str(k)]

                                                if l2 - j <= 1 or l1 - j <= 1 or l1 - j == mask_index:
                                                    continue
                                                if context_words[l2-j][-19] == 0:
                                                    context_words[l2-j][-19] = word_to_index["REP_REL_"+str(-1*j)+"_REL_POINT_POSITION_"+str(k)]
                                                elif context_words[l2-j][-20] == 0:
                                                    context_words[l2-j][-20] = word_to_index["REP_REL_"+str(-1*j)+"_REL_POINT_POSITION_"+str(k)]

                missing_folds = {}
                if not rag_mode and (sent_active or clause_active or clause2_active or missing_active):
                    phrase_min = -1 * phrase_max
                    missing_folds = copy.deepcopy(missing_options)
                    sent_i = sent_markers[i]
                    phrase_i = phrase_markers[i]

                    for iii in range(mask_index-1,0,-1):
                        iiii = iii - mask_index
                        if context_words[mask_index+iiii][0] == 0:
                            continue
                        word2 = corpus[i + iiii]
                        sent_j = sent_markers[i+iiii]
                        phrase_j = phrase_markers[i+iiii]
                        if sent_i > 5 and sent_j == 0:
                            sent_j = sent_i
                        if phrase_i > 5 and phrase_j == 0:
                            phrase_j = phrase_i
                        rel_sent_i = sent_j - sent_i
                        abs_phrase_pos_i = phrase_markers_pos[i+iiii]
                        abs_phrase_neg_i = abs_phrase_pos_i - sent_phrase_totals[sent_j] - 1
                        if rel_sent_i == 0:
                            rel_phrase_i = phrase_j - phrase_i
                        else:
                            rel_phrase_i = abs_phrase_neg_i
                        if rel_phrase_i >= phrase_max:
                            rel_phrase_i = phrase_max
                        if rel_phrase_i <= phrase_min:
                            rel_phrase_i = phrase_min
                        if abs_phrase_pos_i >= phrase_max:
                            abs_phrase_pos_i = phrase_max
                        if abs_phrase_neg_i <= phrase_min:
                            abs_phrase_neg_i = phrase_min

                        relsent_abs_pos = "RELPOSITION_SENT_"+str(rel_sent_i)+"_ABS_PHRASE_"+str(abs_phrase_pos_i)
                        relsent_abs_neg = "RELPOSITION_SENT_"+str(rel_sent_i)+"_ABS_PHRASE_"+str(abs_phrase_neg_i)
                        relsent_rel = "RELPOSITION_SENT_"+str(rel_sent_i)+"_REL_PHRASE_"+str(rel_phrase_i)
                        if sent_active:
                            context_words[mask_index+iiii][-4] = word_to_index[relsent_abs_pos]
                            context_words[mask_index+iiii][-5] = word_to_index[relsent_abs_neg]
                        if rel_sent_i == 0:
                            if sent_active:
                                context_words[mask_index+iiii][-6] = word_to_index[relsent_rel]
                            if rel_phrase_i == 0 and missing_active:
                                for w2 in word2:
                                    if w2 in missing_folds:
                                        missing_folds[w2] += 1
                                        break

                        if clause_active:
                            # do not mark the phrase that the target word is in
                            #if not(rel_sent_i == 0 and rel_phrase_i == 0):
                            cls1 = phrase_markers_sets[i+iiii][0]
                            if rel_sent_i == 0:
                                if word_pos in cls1:
                                    cls1 = ""
                            if cls1 != "":
                                context_words[mask_index+iiii][-9] = word_to_index[cls1+"_clause"]

                        if clause2_active:
                            # do not mark the phrase that the target word is in
                            #if not(rel_sent_i == 0):
                            cls2 = phrase_markers_sets[i+iiii][1]
                            cls2h = phrase_markers_sets[i+iiii][2]
                            if rel_sent_i == 0:
                                if len(word_deps) > 0:
                                    cls2 = ""
                                    cls2h = ""
                            if cls2 != "":
                                context_words[mask_index+iiii][-10] = word_to_index[cls2+"_clause"]
                            if cls2h != "" and cls2h != "h":
                                context_words[mask_index+iiii][-11] = word_to_index[cls2h+"_clause"]

                    for iii in range(mask_index+1,len(context_words)):
                        iiii = iii - mask_index
                        if context_words[mask_index+iiii][0] == 0:
                            continue
                        word2 = corpus[i + iiii]
                        sent_j = sent_markers[i+iiii]
                        phrase_j = phrase_markers[i+iiii]
                        if sent_i > 5 and sent_j == 0:
                            sent_j = sent_i
                        if phrase_i > 5 and phrase_j == 0:
                            phrase_j = phrase_i
                        rel_sent_i = sent_j - sent_i
                        abs_phrase_pos_i = phrase_markers_pos[i+iiii]
                        abs_phrase_neg_i = abs_phrase_pos_i - sent_phrase_totals[sent_j] - 1
                        if rel_sent_i == 0:
                            rel_phrase_i = phrase_j - phrase_i
                        else:
                            rel_phrase_i = abs_phrase_pos_i
                        if rel_phrase_i >= phrase_max:
                            rel_phrase_i = phrase_max
                        if rel_phrase_i <= phrase_min:
                            rel_phrase_i = phrase_min
                        if abs_phrase_pos_i >= phrase_max:
                            abs_phrase_pos_i = phrase_max
                        if abs_phrase_neg_i <= phrase_min:
                            abs_phrase_neg_i = phrase_min
                        relsent_abs_pos = "RELPOSITION_SENT_"+str(rel_sent_i)+"_ABS_PHRASE_"+str(abs_phrase_pos_i)
                        relsent_abs_neg = "RELPOSITION_SENT_"+str(rel_sent_i)+"_ABS_PHRASE_"+str(abs_phrase_neg_i)
                        relsent_rel = "RELPOSITION_SENT_"+str(rel_sent_i)+"_REL_PHRASE_"+str(rel_phrase_i)
                        if sent_active:
                            context_words[mask_index+iiii][-4] = word_to_index[relsent_abs_pos]
                            context_words[mask_index+iiii][-5] = word_to_index[relsent_abs_neg]
                        if rel_sent_i == 0:
                            if sent_active:
                                context_words[mask_index+iiii][-6] = word_to_index[relsent_rel]
                            if rel_phrase_i == 0 and missing_active:
                                for w2 in word2:
                                    if w2 in missing_folds:
                                        missing_folds[w2] += 1
                                        break

                        if clause_active:
                            # do not mark the phrase that the target word is in
                            #if not(rel_sent_i == 0 and rel_phrase_i == 0):
                            cls1 = phrase_markers_sets[i+iiii][0]
                            if rel_sent_i == 0:
                                if word_pos in cls1:
                                    cls1 = ""
                            if cls1 != "":
                                context_words[mask_index+iiii][-9] = word_to_index[cls1+"_clause"]

                        if clause2_active:
                            # do not mark the phrase that the target word is in
                            #if not(rel_sent_i == 0):
                            cls2 = phrase_markers_sets[i+iiii][1]
                            cls2h = phrase_markers_sets[i+iiii][2]
                            if rel_sent_i == 0:
                                if len(word_deps) > 0:
                                    cls2 = ""
                                    cls2h = ""
                            if cls2 != "":
                                context_words[mask_index+iiii][-10] = word_to_index[cls2+"_clause"]
                            if cls2h != "" and cls2h != "h":
                                context_words[mask_index+iiii][-11] = word_to_index[cls2h+"_clause"]

                if not rag_mode and missing_active:
                    for m in missing_folds:
                        if missing_folds[m] == 1:
                            for index in range(X_dimension):
                                if context_words[mask_index][index] == 0:
                                    context_words[mask_index][index] = word_to_index["missing_"+m]
                                    #print("missing_"+m)
                                    break

                if not rag_mode and sent3_active:
                    for iii in range(mask_index-1,0,-1):
                        iiii = iii - mask_index
                        if context_words[mask_index+iiii][0] == 0:
                            continue
                        phrase_j = phrase_markers[i+iiii]
                        rel_word_pos_i = word_markers_pos[i+iiii]
                        rel_word_neg_i = rel_word_pos_i - phrase_word_totals[phrase_j] - 1
                        if rel_word_pos_i > word_max:
                            rel_word_pos_i = word_max
                        if rel_word_neg_i < word_min:
                            rel_word_neg_i = word_min
                        #rel_word_neg_i = word_markers_neg[i+iiii]
                        relword_pos = "RELPOSITION_WORD_"+str(rel_word_pos_i)
                        relword_neg = "RELPOSITION_WORD_"+str(rel_word_neg_i)
                        context_words[mask_index+iiii][-8] = word_to_index[relword_neg]
                        context_words[mask_index+iiii][-7] = word_to_index[relword_pos]

                    for iii in range(mask_index+1,len(context_words)):
                        iiii = iii - mask_index
                        if context_words[mask_index+iiii][0] == 0:
                            continue
                        phrase_j = phrase_markers[i+iiii]

                        rel_word_pos_i = word_markers_pos[i+iiii]
                        rel_word_neg_i = rel_word_pos_i - phrase_word_totals[phrase_j] - 1
                        if rel_word_pos_i > word_max:
                            rel_word_pos_i = word_max
                        if rel_word_neg_i < word_min:
                            rel_word_neg_i = word_min
                        #rel_word_neg_i = word_markers_neg[i+iiii]
                        relword_pos = "RELPOSITION_WORD_"+str(rel_word_pos_i)
                        relword_neg = "RELPOSITION_WORD_"+str(rel_word_neg_i)
                        context_words[mask_index+iiii][-8] = word_to_index[relword_neg]
                        context_words[mask_index+iiii][-7] = word_to_index[relword_pos]

                if not rag_mode and dep2_active:
                    for iii in range(mask_index-1,0,-1):
                        iiii = iii - mask_index
                        dep_mods = []
                        word2 = corpus[i+iiii]
                        for w2 in word2:
                            if w2 in dep_h:
                                #connecting with the target word is not allowed
                                if w2[1:] in key_comp_word or "h"+w2 in key_comp_word:
                                    continue
                                #print("------------------")
                                for j in range(int(max_position_embeddings/2)):
                                    word3 = corpus[i+iiii-1-j]
                                    linked = ""
                                    #print("------------------===")
                                    for w3 in word3:
                                        if w3 in dep_g:

                                            #print("------------------========",w2,w3)
                                            if w2[1:] == w3 or w3[1:] == w2:
                                                linked = w3
                                    if linked != "":
                                        #print("------------------==========================")
                                        for w3 in word3:
                                            if w3 != linked:
                                                if w3 in dep_h:
                                                    dep_mods.append(w2+"-"+w3)
                                                if dep2a_active and w3 in dep_g:
                                                    #print("------------------==========================")
                                                    dep_mods.append(w2+"-"+w3)
                        for j in range(len(dep_mods)):
                            #print(dep_mods[j])
                            if j >= 1 and not dep3_active: break
                            if j >= 3: break
                            context_words[mask_index+iiii][-12-j] = word_to_index[dep_mods[j]]

                    for iii in range(mask_index+1,len(context_words)):
                        iiii = iii - mask_index
                        dep_mods = []
                        word2 = corpus[i+iiii]
                        for w2 in word2:
                            if w2 in dep_g:
                                #connecting with the target word is not allowed
                                if w2[1:] in key_comp_word or "h"+w2 in key_comp_word:
                                    continue
                                for j in range(int(max_position_embeddings/2)):
                                    word3 = corpus[i+iiii+1+j]
                                    linked = ""
                                    for w3 in word3:
                                        if w3 in dep_h:
                                            if w2[1:] == w3 or w3[1:] == w2:
                                                linked = w3
                                    if linked != "":
                                        for w3 in word3:
                                            if w3 != linked:
                                                if w3 in dep_g:
                                                    dep_mods.append(w2+"-"+w3)
                                                if dep2a_active and w3 in dep_h:
                                                    #print("------------------=====================================")
                                                    dep_mods.append(w2+"-"+w3)
                        for j in range(len(dep_mods)):
                            #print(dep_mods[j])
                            if j >= 1 and not dep3_active: break
                            if j >= 3 and not dep3a_active: break
                            if j >= 5: break
                            context_words[mask_index+iiii][-12-j] = word_to_index[dep_mods[j]]

                if not rag_mode and dep4_active:
                    index_sub_dep_start = 35
                    index_sub_dep_end = 40
                    sub_deps_indexes = {}
                    sub_depsH_indexes = {}
                    for s in sub_deps:
                        if s in word_to_index:
                            sub_deps_indexes[word_to_index[s]] = sub_deps[s]
                        if "h"+s in word_to_index:
                            sub_depsH_indexes[word_to_index["h"+s]] = sub_deps[s]

                    for iii,c_word in enumerate(context_words):
                        depC_words = {}
                        depH_words = {}
                        for w in c_word:
                            #print(iii,w)
                            if w in sub_deps_indexes:
                                for d1 in sub_deps_indexes[w]:
                                    depC_words[word_to_index[d1]] = 1
                            if not dep5_active:
                                continue
                            if w in sub_depsH_indexes:
                                for d1 in sub_depsH_indexes[w]:
                                    depH_words[word_to_index["Head_"+d1]] = 1
                        index_sub = index_sub_dep_start
                        for d1 in depC_words:
                            c_word[index_sub] = d1
                            index_sub += 1
                        if dep5_active:
                            for d1 in depH_words:
                                if index_sub <= index_sub_dep_end:
                                    c_word[index_sub] = d1
                                    index_sub += 1
                if nulls1_active:
                    for iii,c_word in enumerate(context_words):
                        #only assign NULL_ in context words for words, not for CLS or MASK
                        if not rag_mode and (iii == 0 or iii == mask_index): continue
                        for y_i in range(X_dimension):
                            if c_word[y_i] == 0:
                                if "NULL_"+str(y_i) in word_to_index:
                                    c_word[y_i] = word_to_index["NULL_"+str(y_i)]
            num_words = 0
            total_freq = 0
            freq_product = 1.0
            word_strs = []
            for iii in range(len(context_words)):
                if context_words[iii][0] in index_to_word:
                    w0 = index_to_word[context_words[iii][0]]
                    if "NUM" not in w0 and "UNK" not in w0 and w0 in words_freq2:
                        f0 = words_freq2[w0]
                        if f0 < avg_freq_valid:
                            word_strs.append(w0)
                            total_freq += avg_freq_valid - f0
                            num_words+=1
                        #total_freq += f0
                        freq_product *= np.sqrt(f0 / avg_freq_valid)
            n_avg_freq = num_words * avg_freq_valid
            #norm_seq = "".join(sorted(word_strs))
            norm_seq = "".join(word_strs)
            if norm_seq in norm_word_seq:
                word_seq_repeats+=1
                if rag_mode:
                    has_words = False
            else:
                norm_word_seq[norm_seq] = 0
            norm_word_seq[norm_seq] += 1


            if not has_words:
                #print("no words found: skipping",i)
                not_has_words_count += 1
            else:
                if total_freq > avg_freq_valid/16.0:
                    below_avg_freq += 1
                else:
                    above_avg_freq += 1
                if freq_product < 256.0:
                    below_prd_freq += 1
                else:
                    above_prd_freq += 1
                if num_words > 3:
                    below_avg_num += 1
                else:
                    above_avg_num += 1

                # Mask Model
                if "_def_" in mode:
                    data[0].append(word)
                if rag_blanket_active:
                    context_blanket = {}
                    for bw in blanket_words:
                        if bw+"_STR" in word_to_index and is_target_feature(bw+"_STR",output_strings2):
                            context_blanket[word_to_index[bw+"_STR"]] = blanket_words[bw]
                    rag_blanket_len += len(context_blanket)

                if enable_pickle:
                    if not rag_mode:
                        trgt_cache.append(target_word)
                        ctxt_cache.append(context_words)
                        if rag_blanket_active:
                            blkt_cache.append(context_blanket)
                        if len(trgt_cache) == batch_size:
                            pickle_trgt_word_vectors.write(pickle.dumps(trgt_cache))
                            if enable_dataloader1:
                                loader.dataset.append(torch.tensor(ctxt_cache, dtype=torch.long, device="cpu"))
                            elif enable_memmap:
                                slice_i1 = slice_i * batch_size
                                slice_i2 = slice_i1 + batch_size
                                #print(i,slice_i,slice_i1,slice_i2)
                                fi_memmap[slice_i1:slice_i2, :, :] = np.array(ctxt_cache,dtype=np.int32)
                                slice_i += 1
                            else:
                                pickle_ctxt_word_vectors.write(pickle.dumps(np.array(ctxt_cache,dtype=np.int32)))
                            trgt_cache = []
                            ctxt_cache = []
                            if rag_blanket_active:
                                pickle_blkt_word_vectors.write(pickle.dumps(blkt_cache))
                                blkt_cache = []
                    else:
                        #if i < 24:
                        #    print("write rag file")
                        ctxt_cache.append(context_words)
                        if len(ctxt_cache) == batch_size:
                            if enable_dataloader1:
                                loader.dataset.append(torch.tensor(ctxt_cache, dtype=torch.long, device="cpu"))
                            elif enable_memmap:
                                slice_i1 = slice_i * batch_size
                                slice_i2 = slice_i1 + batch_size
                                #print(i,slice_i,slice_i1,slice_i2)
                                fi_memmap[slice_i1:slice_i2, :, :] = np.array(ctxt_cache,dtype=np.int32)
                                slice_i += 1
                            else:
                                pickle_ctxt_word_vectors.write(pickle.dumps(np.array(ctxt_cache,dtype=np.int32)))
                            ctxt_cache = []
                else:
                    if rag_blanket_active:
                        rag_blanket2.append(list(context_blanket))
                    #print(target_word,"target word B")
                    #print(data[1][:10],"target words AAAA")
                    data[1].append(target_word)
                    #print(data[1][:10],"target words AAAA")
                    #if len(data[1]) > 12: exit()
                    data[2].append(context_words) # includes MASK word
                if "_def_" in mode:
                    data[3].append(context_words2)
                if "testset" in mode:
                    data[4].append(test_id)
                # this is required for reshuffle
                #if "pretrain" in mode:
                #    data[4].append(target_ordered)

                data[5].append(word["STR"])
                has_words_count += 1

        if "transformer" in mode:
            #if valid_word:
            #    i += int(skip_size * 1.5)
            #else:
            i+=1
        else:
            if "tuning" in mode or "testset" in mode:
                i+=(context_size*2)
                #print(i,target_label,target_label_index,corpus[i]["ID"],corpus[i]["STR"],"-----------9")
            else:
                i+=1
                #print(i,target_label,target_label_index,corpus[i]["ID"],corpus[i]["STR"],"-----------10")
    print("fil1",filtered1_count,"of fil_unk",filtered_count_unk,"of fil",filtered_count,"of pop",pop_count,"of", end,"has_words", has_words_count,"not_has",not_has_words_count,"training_Data3 post subsampling")
    print(below_avg_freq,above_avg_freq,below_prd_freq,above_prd_freq,below_avg_num,above_avg_num,"----",avg_freq_valid,"freqs")
    print(len(norm_word_seq),"norm_word_seq len",word_seq_repeats,"repeats")


    del label_ascii_pairs
    del norm_word_seq
    del sent_markers
    del phrase_markers
    del phrase_markers_sets
    del phrase_markers_pos
    #del phrase_markers_neg
    del word_markers_pos
    #del word_markers_neg
    del sent_phrase_totals
    del phrase_word_totals
    del phrase_contents

    print(target_i,len(previous_predictions),"gen tra3")


    if enable_pickle:
        if len(ctxt_cache) != 0:
            if not rag_mode and not rag_blanket_active:
                pickle_trgt_word_vectors.write(pickle.dumps(trgt_cache))
            if enable_dataloader1:
                loader.dataset.append(torch.tensor(ctxt_cache, dtype=torch.long, device="cpu"))
            elif enable_memmap:
                slice_i1 = slice_i * batch_size
                slice_i2 = slice_i1 + len(ctxt_cache)
                #print(i,slice_i,slice_i1,slice_i2)
                fi_memmap[slice_i1:slice_i2, :, :] = np.array(ctxt_cache,dtype=np.int32)
                #slice_i += 1
            else:
                pickle_ctxt_word_vectors.write(pickle.dumps(np.array(ctxt_cache,dtype=np.int32)))
            if rag_blanket_active:
                pickle_blkt_word_vectors.write(pickle.dumps(blkt_cache))

        if not rag_mode:
            pickle_trgt_word_vectors.close()
        if enable_memmap:
            fi_memmap.flush()
            del fi_memmap
        pickle_ctxt_word_vectors.close()
        if rag_blanket_active:
            pickle_blkt_word_vectors.close()
        data[1] = []
        data[2] = []

    del compounds
    compounds = {}
    del ordered_words
    ordered_words = []
    del orig_context_corefs
    orig_context_corefs = {}
    del rag_blanket
    rag_blanket = {}
    had_base = False
    had_lemma = False
    had_last = False
    for i in sorted(index_features.keys()):
        if "lemma_" in index_to_word[i]: had_lemma = True
        if "Last_Suffix" in index_to_word[i]: had_last = True
        if "STR" not in index_to_word[i] and "lemma_" not in index_to_word[i]  and "Longest_Suffix_" not in index_to_word[i]  and "Last_Suffix_" not in index_to_word[i]:
            tag = "NONE"
            if index_to_word[i] in feature_labels:
                tag = feature_labels[index_to_word[i]]
            if tag != "base":
                #print(i,index_to_word[i],tag,index_features[i],"features")
                pass
            else:
                had_base = True
    print("gen3",len(word_to_index),had_base,had_last,had_lemma,pos_active,pos_active2,rag_blanket_len)
    #exit()
    return data, len(word_to_index), label_to_index, zero_shot_unions, prediction_map, output_strings2, rag_blanket2, word_total, words_freq2

def list_to_num_list(list1):
    num_list = []
    num1 = 0
    for i in range(len(list1)):
        if i > 0 and i % 60 == 0:
            #print(i,num1,num_list)
            num_list.append(num1)
            num1 = 0
        num1 += int(list1[i]) << (i%60)
        #print(i,list1[i],num1)
    if(len(list1) % 60 != 0):
        num_list.append(num1)
    return num_list

def num_list_to_list(num_list1):
    list1 = []
    for i in range(len(num_list1)):
        for j in range(60):
            list1.append((num_list1 >> j) & 1)
    return list1

def get_corpus_metadata(corpus_file, corpus,ppn_set,mode,X_dimension=1,max_position_embeddings=0,first=False, rag_mode=False,rag_blanket_mode=False, freqs={}, freqs_h={}):
    global punct_active
    global sent2_active
    feature_strings = {}
    word_to_index= dict()
    index_to_word = dict()
    output_strings = {}
    mod_freq={}
    #vocab = {}
    index = 0
    corpus_max_len = 99999999

    this_folding_active = folding_active
    this_folding_echo_active = folding_echo_active
    this_folding_bidirectional_active = folding_bidirectional_active
    this_folding_once_active = folding_once_active
    this_folding_full = folding_full
    if rag_mode:
        this_folding_active = folding_rag_active
        this_folding_echo_active = folding_rag_echo_active
        this_folding_bidirectional_active = folding_rag_bidirectional_active
        this_folding_once_active = folding_rag_once_active
        this_folding_full = rag_folding_full

    feature_strings["STR"] = "1"
    word_to_index["EXPERIMENT_SPACE"] = index
    word_to_index["CLS"] = index+1
    word_to_index["MASK"] = index+2
    word_to_index["UNION_adjective"] = index+3
    word_to_index["UNION_adverb"] = index+4
    word_to_index["UNION_noun"] = index+5
    word_to_index["UNION_verb"] = index+6
    word_to_index["UNION_proper_noun"] = index+7
    word_to_index["UNION_any"] = index+8
    word_to_index["mask"] = index+9
    word_to_index["mask_STR"] = index+10
    word_to_index["REPLACEDDD"] = index+11
    word_to_index["Suffix_"] = index+12
    word_to_index["Longest_Suffix_"] = index+13
    word_to_index["Prefix_"] = index+14
    word_to_index["UNK"] = index+15
    word_to_index["UNK_STR"] = index+16
    word_to_index["UNK_POS"] = index+17
    word_to_index["UNK_POS2"] = index+18
    word_to_index["UNK_DEP"] = index+19
    word_to_index["UNK_HDEP"] = index+20
    word_to_index["lemma_UNK"] = index+21
    word_to_index["poslemma_UNK"] = index+21
    word_to_index["fold_lemma_UNK"] = index+22
    word_to_index["fold_poslemma_UNK"] = index+23
    word_to_index["PROPNcompound"] = index+24
    #word_to_index["EXTENDED_CONTEXT"] = index+22
    #vocab["EXPERIMENT_SPACE"] = 1
    #vocab["UNION_adjective"] = 1
    #vocab["UNION_adverb"] = 1
    #vocab["UNION_noun"] = 1
    #vocab["UNION_verb"] = 1
    #vocab["UNION_proper_noun"] = 1
    #vocab["UNION_any"] = 1
    feature_strings["EXPERIMENT_SPACE"] = 1
    feature_strings["CLS"] = 1
    feature_strings["MASK"] = 1
    feature_strings["UNION_adjective"] = 1
    feature_strings["UNION_adverb"] = 1
    feature_strings["UNION_noun"] = 1
    feature_strings["UNION_verb"] = 1
    feature_strings["UNION_proper_noun"] = 1
    feature_strings["UNION_any"] = 1
    feature_strings["mask"] = 1
    feature_strings["mask_STR"] = 1
    feature_strings["REPLACEDDD"] = 1
    feature_strings["Suffix_"] = 1
    feature_strings["Longest_Suffix_"] = 1
    feature_strings["Prefix_"] = 1
    feature_strings["UNK"] = 1
    feature_strings["UNK_STR"] = 1
    feature_strings["UNK_POS"] = 1
    feature_strings["UNK_POS2"] = 1
    feature_strings["UNK_DEP"] = 1
    feature_strings["UNK_HDEP"] = 1
    feature_strings["lemma_UNK"] = 1
    feature_strings["poslemma_UNK"] = 1
    feature_strings["fold_lemma_UNK"] = 1
    feature_strings["fold_poslemma_UNK"] = 1
    feature_strings["PROPNcompound"] = 1
    #feature_strings["EXTENDED_CONTEXT"] = 1
    index += 26
    if "sqs" in mode or "sqx" in mode or "qx" in mode or "fx" in mode:
        print(mode,max_position_embeddings)

        word_to_index["POSTAGS_UNK"] = index
        feature_strings["POSTAGS_UNK"] = 1
        index += 1
        word_to_index["BTAGS_UNK"] = index
        feature_strings["BTAGS_UNK"] = 1
        index += 1
        word_to_index["DEP_PRED_UNK"] = index
        feature_strings["DEP_PRED_UNK"] = 1
        index += 1
        if "pos" in input_mode:
            for p in postags:
                word_to_index["POSTAGS_"+p] = index
                feature_strings["POSTAGS_"+p] = 1
                index += 1
        if "btag" in input_mode:
            for p in btags:
                word_to_index["BTAGS_"+p] = index
                feature_strings["BTAGS_"+p] = 1
                index += 1
        if "dep" in input_mode:
            for p in dep_g:
                word_to_index["DEP_PRED_"+p] = index
                feature_strings["DEP_PRED_"+p] = 1
                index += 1

        for i in range(max_position_embeddings):
            word_to_index["POSITION_"+str(i)] = index
            feature_strings["POSITION_"+str(i)] = 1
            index += 1

        if nulls1_active:
            for i in range(X_dimension):
                word_to_index["NULL_"+str(i)] = index
                feature_strings["NULL_"+str(i)] = 1
                index += 1

        #if dep_positions_active:
        #    for i in range(max_position_embeddings):
        #        word_to_index["DEP_POSITION_"+str(i)] = index
        #        feature_strings["DEP_POSITION_"+str(i)] = 1
        #        index += 1
        #    for i in range(max_position_embeddings):
        #        word_to_index["HEAD_POSITION_"+str(i)] = index
        #        feature_strings["HEAD_POSITION_"+str(i)] = 1
        #        index += 1



        if pos_active:
            for i in range(1,max_position_embeddings):
                word_to_index["RELPOSITION_"+str(i)] = index
                feature_strings["RELPOSITION_"+str(i)] = 1
                index += 1
            for i in range(1,max_position_embeddings):
                word_to_index["RELPOSITION_-"+str(i)] = index
                feature_strings["RELPOSITION_-"+str(i)] = 1
                index += 1

            if dep_pointers_active:
                for i in range(1,max_position_embeddings):
                    word_to_index["DEP_POINT_POSITION_"+str(i)] = index
                    feature_strings["DEP_POINT_POSITION_"+str(i)] = 1
                    index += 1
                    word_to_index["HEAD_POINT_POSITION_"+str(i)] = index
                    feature_strings["HEAD_POINT_POSITION_"+str(i)] = 1
                    index += 1

            if rep_pointers_active:
                for i in range(0,max_position_embeddings):
                    word_to_index["REP_POINT_POSITION_"+str(i)] = index
                    feature_strings["REP_POINT_POSITION_"+str(i)] = 1
                    index += 1

            if rep2_pointers_active:
                #for i in range(0,max_position_embeddings):
                for j in range(1,6):
                    for k in range(-5,6):
                        word_to_index["REP_REL_"+str(j)+"_REL_POINT_POSITION_"+str(k)] = index
                        feature_strings["REP_REL_"+str(j)+"_REL_POINT_POSITION_"+str(k)] = 1
                        index += 1
                        word_to_index["REP_REL_"+str(-1*j)+"_REL_POINT_POSITION_"+str(k)] = index
                        feature_strings["REP_REL_"+str(-1*j)+"_REL_POINT_POSITION_"+str(k)] = 1
                        index += 1

            if dep_rel_positions_active:
                for i in range(1,max_position_embeddings):
                    word_to_index["DEP_RELPOSITION_"+str(i)] = index
                    feature_strings["DEP_RELPOSITION_"+str(i)] = 1
                    index += 1
                for i in range(1,max_position_embeddings):
                    word_to_index["DEP_RELPOSITION_-"+str(i)] = index
                    feature_strings["DEP_RELPOSITION_-"+str(i)] = 1
                    index += 1
                for i in range(1,max_position_embeddings):
                    word_to_index["HEAD_RELPOSITION_"+str(i)] = index
                    feature_strings["HEAD_RELPOSITION_"+str(i)] = 1
                    index += 1
                for i in range(1,max_position_embeddings):
                    word_to_index["HEAD_RELPOSITION_-"+str(i)] = index
                    feature_strings["HEAD_RELPOSITION_-"+str(i)] = 1
                    index += 1

            if pos_active2:
                #VERB ADV ADP ADJ PART NOUN PRON PROPN CCONJ DET AUX
                for pos in ["VERB","ADV","ADP","ADJ","PART","NOUN","PRON","PROPN","CCONJ","DET","AUX"]:
                    if pos not in word_to_index:
                        word_to_index[pos] = index
                        feature_strings[pos] = 1
                        index += 1

                    #for pos in ["VERB","ADV","ADP","ADJ","PART","NOUN","PRON","PROPN","CCONJ","DET","AUX","nsubj","hnsubj","nsubjpass","hnsubjpass","dobj","hdobj","ROOT"]:
                    for i in range(1,int(max_position_embeddings/4)):
                        word_to_index[pos+"_RELPOSITION_"+str(i)] = index
                        feature_strings[pos+"_RELPOSITION_"+str(i)] = 1
                        index += 1
                    for i in range(1,int(max_position_embeddings/4)):
                        word_to_index[pos+"_RELPOSITION_-"+str(i)] = index
                        feature_strings[pos+"_RELPOSITION_-"+str(i)] = 1
                        index += 1
                    if pos_pointers_active:
                        for i in range(1,max_position_embeddings):
                            word_to_index[pos+"_POINT_POSITION_"+str(i)] = index
                            feature_strings[pos+"_POINT_POSITION_"+str(i)] = 1
                            index += 1

        if sent3_active:
            for i in range(-1*word_max,word_max+1):
                relpos = "RELPOSITION_WORD_"+str(i)
                word_to_index[relpos] = index
                feature_strings[relpos] = 1
                index += 1

        if sent_active:
            for i in range(0,phrase_max+1):
                for j in range(0,phrase_max+1):
                    if i == 0:
                        relpos = "RELPOSITION_SENT_"+str(i)+"_REL_PHRASE_"+str(j)
                        word_to_index[relpos] = index
                        feature_strings[relpos] = 1
                        index += 1
                        if j != 0:
                            relpos = "RELPOSITION_SENT_"+str(i)+"_REL_PHRASE_-"+str(j)
                            word_to_index[relpos] = index
                            feature_strings[relpos] = 1
                            index += 1
                    relpos = "RELPOSITION_SENT_"+str(i)+"_ABS_PHRASE_"+str(j)
                    word_to_index[relpos] = index
                    feature_strings[relpos] = 1
                    index += 1
                    if j != 0:
                        relpos = "RELPOSITION_SENT_"+str(i)+"_ABS_PHRASE_-"+str(j)
                        word_to_index[relpos] = index
                        feature_strings[relpos] = 1
                        index += 1
                    #relpos = "RELPOSITION_SENT_-"+str(i)+"_REL_PHRASE_"+str(j)
                    #word_to_index[relpos] = index
                    #feature_strings[relpos] = 1
                    #index += 1
                    if i == 0: continue
                    #if j != 0:
                    #    relpos = "RELPOSITION_SENT_-"+str(i)+"_REL_PHRASE_-"+str(j)
                    #    word_to_index[relpos] = index
                    #    feature_strings[relpos] = 1
                    #    index += 1
                    relpos = "RELPOSITION_SENT_-"+str(i)+"_ABS_PHRASE_"+str(j)
                    word_to_index[relpos] = index
                    feature_strings[relpos] = 1
                    index += 1
                    if j != 0:
                        relpos = "RELPOSITION_SENT_-"+str(i)+"_ABS_PHRASE_-"+str(j)
                        word_to_index[relpos] = index
                        feature_strings[relpos] = 1
                        index += 1
        if freq_active:
            for i in freq_2s:
                freq = "FREQ_1/"+str(i)
                word_to_index[freq] = index
                feature_strings[freq] = 1
                index += 1
        if freq2_active:
            for i in freq_2s_short:
                freq = "FREQ_HEAD_1/"+str(i)
                word_to_index[freq] = index
                feature_strings[freq] = 1
                index += 1

        pos_heirarchy = ["VERB", "PREP", "NOUN", "PRON", "PROPN"]
        clause_deps = ["advcl","appos","ccomp","parataxis","pcomp","prepc","ROOT","xcomp"]

        if clause_active:
            for p in pos_heirarchy:
                word_to_index[p+"_clause"] = index
                feature_strings[p+"_clause"] = 1
                index += 1

        if clause2_active:
            for p in clause_deps:
                word_to_index[p+"_clause"] = index
                feature_strings[p+"_clause"] = 1
                index += 1
                word_to_index["h"+p+"_clause"] = index
                feature_strings["h"+p+"_clause"] = 1
                index += 1

        if dep2_active:
            #deps_all = []
            #deps_all.extend(list(dep_g.keys()))
            #deps_all.extend(list(dep_h.keys()))
            #print(deps_all)
            for d1 in dep_g.keys():
                for d2 in dep_g.keys():
                    d3 = d1+"-"+d2
                    word_to_index[d3] = index
                    feature_strings[d3] = 1
                    index += 1
                if dep2a_active:
                    for d2 in dep_h.keys():
                        d3 = d1+"-"+d2
                        word_to_index[d3] = index
                        feature_strings[d3] = 1
                        index += 1
            for d1 in dep_h.keys():
                for d2 in dep_h.keys():
                    d3 = d1+"-"+d2
                    word_to_index[d3] = index
                    feature_strings[d3] = 1
                    index += 1
                if dep2a_active:
                    for d2 in dep_g.keys():
                        d3 = d1+"-"+d2
                        word_to_index[d3] = index
                        feature_strings[d3] = 1
                        index += 1


        if dep4_active:
            for d in sub_deps:
                for d1 in sub_deps[d]:
                    if d1 not in word_to_index:
                        word_to_index[d1] = index
                        feature_strings[d1] = 1
                        index += 1

        if dep5_active:
            for d in sub_deps:
                for d1 in sub_deps[d]:
                    if "Head_"+d1 not in word_to_index:
                        word_to_index["Head_"+d1] = index
                        feature_strings["Head_"+d1] = 1
                        index += 1

        if missing_active:
            for m in missing_options:
                if m not in ["CCONJ","SCONJ","csubj","csubjpass","nsubj","nsubjpass"]:
                    if "missing_"+m not in word_to_index:
                        word_to_index["missing_"+m] = index
                        feature_strings["missing_"+m] = 1
                        index += 1

        if this_folding_active:
            for fold in folding_markers:
                word_to_index[fold] = index
                feature_strings[fold] = 1
                index += 1


            label_ascii_blocked = {}
            if "newsgroup" in corpus_file or "_n20_" in corpus_file:
                labels_at = []
                asciis_at = []
                for i in range(len(corpus)):
                    #print(corpus2[i]["STR"])
                    if corpus[i]["STR"] == "label":
                        labels_at.append(i)
                    if corpus[i]["STR"] == "ascii":
                        asciis_at.append(i)
                for i in range(len(labels_at)):
                    for j in range(len(asciis_at)):
                        if asciis_at[j] > labels_at[i] and asciis_at[j] - labels_at[i] < 12:
                            #print(labels_at[i],asciis_at[j]+1)
                            for k in range(labels_at[i],asciis_at[j]+1,1):
                                label_ascii_blocked[k] = 1
                                #if len(label_ascii_blocked) < 10:
                                #    print(k)


            dep_ws = {}
            for fold in this_folding_full:
                if fold in folding_options1:
                    folds = folding_options1[fold].split(":")
                    for f in folds:
                        for i in range(len(corpus)):
                            if i in label_ascii_blocked:
                                continue
                            word = corpus[i]
                            if f in word:
                                dep_ws[i] = 1
                elif fold in folding_options:
                    folds = folding_options[fold].split(":")
                    for f in folds:
                        for i in range(len(corpus)):
                            if i in label_ascii_blocked:
                                continue
                            word = corpus[i]
                            if f in word:
                                dep_ws[i] = 1

            corpus_effective_len = 0
            sim_corpus = []
            for i in range(len(corpus)):
                #if i in label_ascii_blocked:
                #    continue
                if i not in dep_ws:
                    corpus_effective_len += 1
                    #sim_corpus.append(corpus[i])
            '''
            for i in range(len(sim_corpus)):
                    #if sim_corpus[i]["STR"] not in output_strings:
                    #    continue

                    if "UNK" in sim_corpus[i]["STR"] or "NUM" in sim_corpus[i]["STR"] or "last_punct" in sim_corpus[i]["STR"]:
                        continue
                    has_words = (len(sim_corpus[i]["STR"]) >= 2 and sim_corpus[i]["STR"].isalpha())
                    for j in range(4):
                        has_words = has_words or (len(sim_corpus[i + j + 1]["STR"]) >= 2 and sim_corpus[i + j + 1]["STR"].isalpha())
                        has_words = has_words or (len(sim_corpus[i - j - 1]["STR"]) >= 2 and sim_corpus[i - j - 1]["STR"].isalpha())
                    if has_words:
                        corpus_effective_len += 1
            '''
            print("effective len",corpus_effective_len)
            sim_corpus = []
            corpus_max_len = corpus_effective_len

    print(len(word_to_index)," num metadata tokens - corpus metadata")
    if renorm_active:
        for r in renorm_options:
            if r not in word_to_index:
                word_to_index[r] = index
                feature_strings[r] = 1
                index += 1
        if renorm3_active:
            for r in renorm_options2:
                if r not in word_to_index:
                    word_to_index[r] = index
                    feature_strings[r] = 1
                    index += 1

    #for i in range(X_dimension):
    #    word_to_index["MASK_"+str(i)] = index
    #    feature_strings["MASK_"+str(i)] = 1
    #    index += 1
    #print(word_to_index,"corpus_metadata init")

    for l in labels_test:
        if "base" in mode and "_STR" in l: continue
        if "parse" in mode and "_STR" not in l: continue
        if "string" in mode and "_STR" not in l: continue
        word_to_index[l] = index
        index+=1

    corpus2 = []
    #last_propn = None


    if "propn" in mode:
        if word_to_index.get("zzzzzz_ppn_phrase_STR") == None:
            word_to_index.update ( {"zzzzzz_ppn_phrase_STR" : index})
            index  += 1
            feature_strings["zzzzzz_ppn_phrase_STR"] = "1"
            #vocab["zzzzzz_ppn_phrase_STR"] = 1

    write_freq = False
    if len(freqs) == 0:
        write_freq = True

    ids = {}
    for i in range(len(corpus)):
        word = corpus[i]
        if word["ID"] not in ids:
            ids[word["ID"]] = []
        ids[word["ID"]].append(i)

    fold_ws = {}
    for fold in this_folding_full:
        if fold in folding_options:
            folds = folding_options[fold].split(":")
            for f in folds:
                fold_ws[f] = 1
    print(fold_ws,"fold_ws")

    #print("corpus_metadata: ",hypoth1_active,hypoth_active)
    end = len(corpus)
    for iii in range(end):
        word = corpus[iii]
        i0 = iii
        id0 = word["ID"]
        longest = ""
        skip_longest = False
        del_w = {}
        add_w = {}
        add_poslemma = {}
        skip_propn = False
        #if "UNK" in word or "UNK_STR" in word:
        #    continue
        has_propn = False
        propn_base = ""
        propn_word = ""
        propn_str = False
        word_pos = ""

        if rag_mode:
            word["rag_1"] = "metadata"
        if rag_active and not rag_mode:
            word["rag_0"] = "metadata"

        if "PUNCT" in word:
            continue

        for w in word:
            has_propn = has_propn or word[w] == "pos" and w == "PROPN"
            if word[w] == "base":
                propn_base = w
            if word[w] == "word":
                propn_word = w
                if "STR" in w:
                    propn_str = True
            if word[w] == "pos":
                word_pos = w

        if segment_dep_active and not rag_blanket_mode and word_pos != "":
            add_word = {}
            for w in word:
                if w in dep_g or w in dep_h:
                    i1,id1 = get_linked_index(corpus,i0,w,ids)
                    word1 = corpus[i1]
                    word1_pos = ""
                    for w1 in word1:
                        if word1[w1] == "pos":
                            word1_pos = w1
                    if word1_pos != "":
                        word_dep = w+"_"+word_pos+"_"+word1_pos
                        if word_to_index.get(word_dep) == None:
                            word_to_index.update ( {word_dep : index})
                            index  += 1
                            if word_dep[0] == "h":
                                dep_segment_h[word_dep] = dep_h[w] + 30
                            else:
                                dep_segment_g[word_dep] = dep_g[w] + 30
                        add_word[word_dep] = "metadata-"+id1
            for w in add_word:
                word[w] = add_word[w]

        if hypoth_active:
            rem_features = {}
            for w in word:
                if w != "STR" and word[w] == "mod":
                    fields = w.split("_")
                    if fields[0] != "0":
                        rem_features[w] = 1
            for r in rem_features:
                del word[r]
        else:
            rem_features = {}
            for w in word:
                if w != "STR" and word[w] == "mod":
                    rem_features[w] = 1
            for r in rem_features:
                del word[r]


        fold_ok = False
        for w in word:
            if w in fold_ws:
                fold_ok = True
                break

        if word_pos != "" and not rag_blanket_mode:
            if segment_lemma_active:
                for w in word:
                    if "lemma_" in w and "poslemma_" not in w:
                        if word_to_index.get("pos"+w+"_"+word_pos) == None:
                            word_to_index.update ( {"pos"+w+"_"+word_pos : index})
                            index  += 1
                        add_poslemma["pos"+w+"_"+word_pos] = 1
                        if (this_folding_active or folding_vocab_active) and fold_ok:
                            if word_to_index.get("fold_pos"+w+"_"+word_pos) == None:
                                #print("fold_pos"+w+"_"+word_pos)
                                word_to_index.update ( {"fold_pos"+w+"_"+word_pos : index})
                                index  += 1
                for w in add_poslemma:
                    word[w] = "metadata"
                if len(add_poslemma) > 0:
                    a,b,c = get_primary_keys(word,mode,3,True)
            else:
                for w in word:
                    if "lemma_" in w and "poslemma_" not in w:
                        if (this_folding_active or folding_vocab_active) and fold_ok:
                            if word_to_index.get("fold_"+w) == None:
                                word_to_index.update ( {"fold_"+w : index})
                                index  += 1
                        if ragV_active and "5" in folding:
                            if word_to_index.get("rag_"+w) == None:
                                word_to_index.update ( {"rag_"+w : index})
                                index  += 1

        longest_suffix = ""
        has_longest_suffix = False
        has_suffix = False
        str_word = word["STR"]
        str_base = ""
        str_last_suffix = ""
        has_head = False
        for w in word:
            if iii < corpus_max_len and w == "STR":
                    output_strings[word[w]] = 1

            if w != "ID" and w != "STR":
                #has_propn = has_propn or ("propn" in mode and w == "PROPN")
                if word[w] == "base" or (word[w] == "word" and "STR" not in w):
                    str_base = w
                    #print(str_base)
                if "##" in w:
                    offset_s = -1*(len(w)-2)
                    if w[offset_s:] == str_word[offset_s:]:
                        str_last_suffix = w
                if w == "Coreference":
                    if "_STR" not in word[w]:
                        while len(word[w]) > 1 and not word[w][-1].isalpha(): word[w] = word[w][:-1]
                        non_alpha =False
                        for c in word[w]:
                            if not c.isalpha():
                                non_alpha = True
                        if non_alpha:
                            word[w] = "metadata"
                        else:
                            word[w] = word[w]+"_STR"
                        if word_to_index.get(word[w]) == None:
                            word_to_index.update ( {word[w] : index})
                            index  += 1
                        #print(word[w])

                if w in dep_h:
                    has_head = True

                if word[w] != "word" and w not in del_w:
                    feature_strings[w] = "1"

                if w != "STR" and word[w] == "mod":
                    if hypoth1_active:
                        fields = w.split("_")
                        if fields[0] != "0":
                            new_feature = fields[0]+fields[1]+fields[3]
                            if new_feature not in mod_freq:
                                mod_freq[new_feature] = 0
                            mod_freq[new_feature] += 1
                            #print(new_feature,"get_corpus_metadata")
                            if word_to_index.get(new_feature) == None:
                                word_to_index.update ( {new_feature : index})
                                index  += 1

                elif word_to_index.get(w) == None:
                    word_to_index.update ( {w : index})
                    index  += 1

                if "PPN_" in w or "IDM_" in w or "DEF_" in w or "LIT_" in w or "gls_" in w: continue

                if word[w] == "sfx" and "Suffix" in w:
                    has_suffix = True
                    if "Longest_" in w:
                        has_longest_suffix = True
                    if "Last_" not in w and len(w) > len(longest_suffix):
                        longest_suffix = w
        if "string" not in mode:
            if str_last_suffix != "":
                longest_suffix = "Suffix_"+str_word[len(str_base):]
                word["Last_Suffix_"+str_last_suffix] = "sfx"
                has_suffix = True
            for w in word:
                if "##" in w and not has_suffix:
                    print(word,"corpus metadata spacy???? WARNING")
                    exit()
            if not has_suffix:
                word["Suffix_"] = "sfx"
                word["Longest_Suffix_"] = "sfx"
                word["Last_Suffix_"] = "sfx"
            if not has_longest_suffix:
                if longest_suffix == "":
                    longest_suffix = "Longest_Suffix_"
                else:
                    longest_suffix = "Longest_"+longest_suffix
                word[longest_suffix] = "sfx"
                feature_strings[longest_suffix] = "1"
                if word_to_index.get(longest_suffix) == None:
                    word_to_index.update ( {longest_suffix : index})
                    index  += 1
        for w in del_w:
            del word[w]
        for w in add_w:
            word[w] = add_w[w]

        if not skip_propn:
            word2 = {}
            for w in word:
                word2[w] = word[w]
            w2 = list(word.keys())
            for w in w2:
                del word[w]
            del word

            corpus2.append(word2)

    del corpus
    corpus = None

    total_freq = 0
    for f in freqs:
        for p in freqs[f]:
            total_freq += freqs[f][p]

    if first:
        corpus, word_to_index, feature_strings, ppn_set = add_proper_nouns(corpus2, word_to_index, feature_strings, ppn_set, mode)
        vocab_size = len(word_to_index)

        corpus2 = []

        for word in corpus:
            del_w = {}
            for w in word:
                if w != "ID" and w != "STR":
                    if "string" in mode:
                        if word[w] in ["s/p","sfx","pos","pos2","base"]:
                            del_w[w] = 1
                        if not isinstance(word[w],list) and word[w].isnumeric() and int(word[w]) > 10:
                            del_w[w] = 1
            for w in del_w:
                del word[w]
            corpus2.append(word)

    if punct_active:
        corpus = corpus2
        corpus2 = []

        previous_word = {}
        for word in corpus:
            for w in word:
                if "last_punct" in w and "punct" not in previous_word and "punct" not in word:
                    new_word = {w+"_STR":"word","STR":w,"punct":"metadata"}
                    corpus2.append(new_word)
                    break
            previous_word = word
            corpus2.append(word)

    if this_folding_active and not first and not rag_blanket_mode:
        #folding_options = {".":"punct", "f":"NUM", "k":"UNK", "j":"ADJ", "d":"ADV", "t":"DET", "e":"PROPN", "n":"NOUN", "x":"AUX", "v":"VERB", "p":"PREP:ADP", "r":"pobj", "o":"dobj", "i":"iobj", "u":"PRON", "s":"csubj:csubjpass:nsubj:nsubjpass", "b":"CCONJ:SCONJ","!":"INTJ","y":"X:SYM","z":"PART","c":"compound"}
        #folding_options1 = {".":"punct", "f":"NUM", "k":"UNK","!":"INTJ","y":"SYM:X"}
        #folding_next = ["last_punct","next_punct","last_NUM","next_NUM","last_UNK","next_UNK","last_INTJ","next_INTJ","last_X","next_X","last_SYM","next_SYM","last_PART","next_PART"]

        if True:
            corpus_effective_len = 0
            sim_corpus = []
            for i in range(len(corpus2)):
                if i in label_ascii_blocked:
                    continue
                if i not in dep_ws:
                    sim_corpus.append(corpus2[i])
            for i in range(len(sim_corpus)):
                    if sim_corpus[i]["STR"] not in output_strings:
                        continue

                    if "UNK" in sim_corpus[i]["STR"] or "NUM" in sim_corpus[i]["STR"] or "last_punct" in sim_corpus[i]["STR"]:
                        continue
                    has_words = (len(sim_corpus[i]["STR"]) >= 2 and sim_corpus[i]["STR"].isalpha())
                    for j in range(4):
                        if i + j + 1 >= len(sim_corpus) or i - j - 1 <= 0: continue
                        has_words = has_words or (len(sim_corpus[i + j + 1]["STR"]) >= 2 and sim_corpus[i + j + 1]["STR"].isalpha())
                        has_words = has_words or (len(sim_corpus[i - j - 1]["STR"]) >= 2 and sim_corpus[i - j - 1]["STR"].isalpha())
                    if has_words:
                        corpus_effective_len += 1
            corpus_max_len = corpus_effective_len
            for k in range(4):
                corpus_effective_len = 0
                output_strings2 = {}
                for iii in range(len(sim_corpus)):
                    word = sim_corpus[iii]
                    for w in word:
                        if iii < corpus_max_len and w == "STR":
                                output_strings2[word[w]] = 1
                for i in range(len(sim_corpus)):
                        if sim_corpus[i]["STR"] not in output_strings2 or sim_corpus[i]["STR"] not in output_strings:
                            continue

                        if "UNK" in sim_corpus[i]["STR"] or "NUM" in sim_corpus[i]["STR"] or "last_punct" in sim_corpus[i]["STR"]:
                            continue
                        has_words = (len(sim_corpus[i]["STR"]) >= 2 and sim_corpus[i]["STR"].isalpha())
                        for j in range(4):
                            if i + j + 1 >= len(sim_corpus) or i - j - 1 <= 0: continue
                            has_words = has_words or (len(sim_corpus[i + j + 1]["STR"]) >= 2 and sim_corpus[i + j + 1]["STR"].isalpha())
                            has_words = has_words or (len(sim_corpus[i - j - 1]["STR"]) >= 2 and sim_corpus[i - j - 1]["STR"].isalpha())
                        if has_words:
                            corpus_effective_len += 1
                corpus_max_len = corpus_effective_len
            corpus_max_len = int(corpus_max_len * 0.91)
            sim_corpus = []
            output_strings2 = {}

        fold_key = "lemma_"
        if segment_lemma_active:
            fold_key = "poslemma_"

        after_pred_marker = False
        for fold in this_folding_full:
            if fold == "^":
                after_pred_marker = True

            if fold in folding_options1:
                if fold == ".":
                    punct_active = False
                    sent2_active = True
                folds = folding_options1[fold].split(":")
                for f in folds:

                    label_ascii_blocked = {}
                    if "newsgroup" in corpus_file or "_n20_" in corpus_file:
                        labels_at = []
                        asciis_at = []
                        for i in range(len(corpus2)):
                            if corpus2[i]["STR"] == "label":
                                labels_at.append(i)
                            if corpus2[i]["STR"] == "ascii":
                                asciis_at.append(i)
                        for i in range(len(labels_at)):
                            for j in range(len(asciis_at)):
                                if asciis_at[j] > labels_at[i] and asciis_at[j] - labels_at[i] < 12:
                                    #print(labels_at[i],asciis_at[j]+1)
                                    for k in range(labels_at[i],asciis_at[j]+1,1):
                                        label_ascii_blocked[k] = 1

                    corpus = corpus2
                    corpus2 = []
                    dep_ws = {}
                    for i in range(1,len(corpus)-1):
                        if i in label_ascii_blocked:
                            continue
                        word = corpus[i]
                        fold_word = ""
                        for w in word:
                            if f == w:
                                dep_ws[i] = 1
                                fold_word = w
                                break
                        if fold_word == "" or after_pred_marker:
                            continue
                        has_folding = False
                        for w in word:
                            if is_fold_feature(w) and not this_folding_once_active:
                                if this_folding_echo_active and ("punct_" in w or "NUM" in w or "UNK" in w):
                                    continue
                                corpus[i-1][w] = word[w]
                                corpus[i+1][w] = word[w]
                                if w in folding_markers:
                                    corpus[i-1][folding_markers[w]] = "metadata"
                                    corpus[i+1][folding_markers[w]] = "metadata"
                                    has_folding = True
                        word["folding_-1"] = "metadata"
                        if not has_folding:
                            corpus[i-1]["folding_1"] = "metadata"
                            corpus[i+1]["folding_1"] = "metadata"
                        if i > 0:
                            corpus[i-1]["last_"+f] = "metadata="+f
                        if i < len(corpus)-1:
                            corpus[i+1]["next_"+f] = "metadata="+f
                    for i in range(len(corpus)):
                        if i in label_ascii_blocked:
                            corpus2.append(corpus[i])
                            #continue
                        elif i in dep_ws and after_pred_marker:
                            #print(i,"no_prediction_i","1")
                            corpus[i]["folding_-1"] = "metadata"
                            if this_folding_echo_active:
                                corpus2.append(corpus[i])
                        elif i not in dep_ws or after_pred_marker:
                            corpus2.append(corpus[i])
                        else:
                            if this_folding_echo_active:
                                corpus2.append(corpus[i])
                    corpus = []

            elif fold in folding_options:
                folds = folding_options[fold].split(":")
                for f in folds:

                    label_ascii_blocked = {}
                    if "newsgroup" in mode or "_n20_" in mode:
                        labels_at = []
                        asciis_at = []
                        for i in range(len(corpus2)):
                            if corpus2[i]["STR"] == "label":
                                labels_at.append(i)
                            if corpus2[i]["STR"] == "ascii":
                                asciis_at.append(i)
                        for i in range(len(labels_at)):
                            for j in range(len(asciis_at)):
                                if asciis_at[j] > labels_at[i] and asciis_at[j] - labels_at[i] < 12:
                                    for k in range(labels_at[i],asciis_at[j]+1,1):
                                        label_ascii_blocked[k] = 1

                    print(fold,"fold",len(corpus2),f,len(label_ascii_blocked),after_pred_marker,"2")
                    corpus = corpus2
                    corpus2 = []
                    ids = {}
                    dep_ws = {}
                    dep_rev = {}
                    for i in range(len(corpus)):
                        if i in label_ascii_blocked:
                            continue
                        word = corpus[i]
                        if word["ID"] not in ids:
                            ids[word["ID"]] = []
                        ids[word["ID"]].append(i)
                        fold_w = ""
                        dep_w = ""
                        for w in word:
                            if w == f:
                                fold_w = w
                        for w in word:
                            w1 = word[w]
                            if w in dep_g:
                                if dep_w == "" and "-" in w1:
                                    dep_w = w1.split("-")[1]
                                    if dep_w == str(i):
                                        dep_w = "-1"
                                    else:
                                        if dep_w not in dep_rev:
                                            dep_rev[dep_w] = []
                                        dep_rev[dep_w].append(i)
                            if this_folding_bidirectional_active:
                                dep_wh = ""
                                if w in dep_h:
                                    if dep_wh == "" and "-" in w1:
                                        dep_wh = w1.split("-")[1]
                                        if dep_wh == str(i):
                                            dep_wh = "-1"
                                        else:
                                            if dep_wh not in dep_rev:
                                                dep_rev[dep_wh] = []
                                            dep_rev[dep_wh].append(i)
                                        if i not in dep_ws:
                                            dep_ws[i] = []
                                        dep_ws[i].append("h"+dep_wh)

                        if fold_w != "":
                            if dep_w == "":
                                dep_w = "-1"
                            if i not in dep_ws:
                                dep_ws[i] = []
                            dep_ws[i].append(dep_w)
                    for i in range(len(corpus)):
                        if i in label_ascii_blocked:
                            continue
                        word = corpus[i]
                        id1 = word["ID"]
                        j = -99
                        j1 = -99
                        if i in dep_ws and not after_pred_marker:
                            for id_ws in dep_ws[i]:
                                id2 = id_ws
                                has_h = False
                                if id2[0] == "h":
                                    id2 = id2[1:]
                                    has_h = True
                                j = i+1
                                if id2 != "-1" and id2 in ids:
                                    for i2 in ids[id2]:
                                        if abs(i2-i) < 30:
                                            j = i2
                                if j >= len(corpus) or j < 0:
                                    continue
                                id3 = corpus[j]["ID"]
                                fold_word = ""
                                for w in word:
                                    if fold_key in w:
                                        fold_word = w
                                        break
                                if fold_word == "":
                                    continue
                                j1 = i
                                if id1 in dep_rev:
                                    for j2 in dep_rev[id1]:
                                        if abs(j2 - i) < 30:
                                            j1 = j2
                                if False and j1 != i and not has_h:
                                    word2 = corpus[j1]
                                    dep_w2 = ""
                                    dep_w3 = ""
                                    for w in word2:
                                        w1 = word2[w]
                                        if w in dep_g:
                                            if dep_w2 == "" and "-" in w1:
                                                dep_w2 = w1.split("-")[1]
                                                dep_w3 = w
                                    if dep_w2 == id1:
                                        word2[dep_w3] = "metadata-"+id3
                                if j == -99 or j == i:
                                    continue
                                fold_word1 = fold_word
                                if "fold_" != fold_word1[:5]:
                                    fold_word1 = "fold_"+fold_word1
                                word[fold_word1] = "metadata="+f
                                has_folding = False
                                for w in word:
                                    if is_fold_feature(w) and not this_folding_once_active:
                                        if this_folding_echo_active and ("punct_" in w or "NUM" in w or "UNK" in w):
                                            continue
                                        corpus[j][w] = word[w]
                                        if w in folding_markers:
                                            corpus[j][folding_markers[w]] = "metadata"
                                            has_folding = True
                                word["folding_-1"] = "metadata"
                                corpus[j][fold_word1] = "metadata="+f
                                if not has_folding:
                                    corpus[j]["folding_1"] = "metadata"
                    for i in range(len(corpus)):
                        if i in label_ascii_blocked:
                            corpus2.append(corpus[i])
                        elif i in dep_ws and after_pred_marker:
                            corpus[i]["folding_-1"] = "metadata"
                            if this_folding_echo_active:
                                corpus2.append(corpus[i])
                        elif i not in dep_ws or after_pred_marker:
                            corpus2.append(corpus[i])
                        else:
                            if this_folding_echo_active:
                                corpus2.append(corpus[i])
                    corpus = []

        for i in range(len(corpus2)):
            a,b,c = get_primary_keys(corpus2[i],mode,3,True)

    blanket2 = {}

    #move all target features to the beginning
    word_to_index2 = {}
    index2 = 0
    for w in sorted(word_to_index.keys()):
        if is_target_feature(w,output_strings):
            if word_to_index2.get(w) == None:
                word_to_index2.update ( {w : index2})
                index2  += 1
    #now add the rest
    for w in sorted(word_to_index.keys()):
            if word_to_index2.get(w) == None:
                word_to_index2.update ( {w : index2})
                index2  += 1
    del word_to_index


    vocab_size = len(word_to_index2)
    length_of_corpus = len(corpus2)
    return word_to_index2,index_to_word,corpus2,vocab_size,length_of_corpus,feature_strings, ppn_set, freqs, freqs_h, mod_freq,output_strings,corpus_max_len,blanket2

def generate_dictionary_data2(data,token_map,short_map,corpus_file,ppn_set,mode,X_dimension,max_position_embeddings,in_rag=False,in_rag_blanket=False):
    feature_strings = {}
    corpus = []
    print("start dictionary data2")
    if corpus_file:
        with open(corpus_file, 'r') as f:
            for item in json_stream.load(f):
                # Convert the transient dictionary to a real Python dict
                word = dict(item.items())
                tmp = list(word.keys())
                for w in tmp:
                    if w in short_map:
                        t = word[w]
                        del word[w]
                        word[short_map[w]] = t
                tmp = list(word.keys())
                del_list = []
                for w in tmp:
                    if "boths" in mode.lower() or "parse" in mode.lower():
                        if w == "STR":
                            if word[w][0] in "0123456789" or word[w][-1] in "0123456789":
                                word["NUM_STR"] = "word"
                                if word[w] in word:
                                    #word["UNK"] = word[w]
                                    del_list.append(word[w])
                                word[w] = "NUM"
                            elif word[w] in ["g", "y","f","k","n","j","l","e","w", "u","r","o","v","q", "b","p","x","c","h","z"]:
                                word["UNK_STR"] = "word"
                                if word[w] in word:
                                    #word["UNK"] = word[w]
                                    del_list.append(word[w])
                                word[w] = "UNK"
                            else:
                                word[word[w]+"_STR"] = "word"
                        elif word[w] == "word":
                            if word[w][0] in "0123456789" or word[w][-1] in "0123456789":
                                word["NUM"] = "base"
                                del_list.append(w)
                            elif word[w] in ["g", "y","f","k","n","j","l","e","w", "u","r","o","v","q", "b","p","x","c"]:
                                word["UNK"] = "base"
                                del_list.append(w)
                            else:
                                word[w] = "base"
                        elif "lemma_" in w:
                            if w[6] in "0123456789" or w[-1] in "0123456789":
                                del_list.append(w)
                            elif len(w) == 7 and w[6] in ["g", "y","f","k","n","j","l","e","w", "u","r","o","v","q", "b","p","x","c"]:
                                del_list.append(w)
                for d in del_list:
                    del word[d]

                if hypoth_active and not in_rag and not in_rag_blanket:
                    rem_features = {}
                    for w in word:
                        if w != "STR" and word[w] == "mod":
                            fields = w.split("_")
                            #print(fields)
                            if fields[0] != "0":
                                rem_features[w] = 1
                    for r in rem_features:
                        del word[r]
                else:
                    rem_features = {}
                    for w in word:
                        if w != "STR" and word[w] == "mod":
                            rem_features[w] = 1
                    for r in rem_features:
                        del word[r]
                word3 = {}
                for w in word:
                    word3[w] = word[w]
                w2 = list(word.keys())
                for w in w2:
                    del word[w]
                del word
                word = None

                dilemma = {"dilemma":"dilemna","dilemma_STR":"dilemna_STR","lemma_dilemma":"lemma_dilemna","lemma_dilemma_STR":"lemma_dilemna_STR","trilemma":"trilemna","trilemma_STR":"trilemna_STR","lemma_trilemma":"lemma_trilemna","lemma_trilemma_STR":"lemma_trilemna_STR"}
                for s in dilemma:
                    if s in word3:
                        word3[dilemma[s]] = word3[s]
                        del word3[s]
                        #print(s,word3,"1")
                    if s == word3["STR"]:
                        word3["STR"] = dilemma[s]
                        #print(s,word3,"2")

                corpus.append(word3)

    else:
        corpus = []
        j = 0
        for row in data:
            word = {}
            #print(row)
            word["ID"] = row[1]
            if int(row[2]) >= 1234560000000:
                word["UNK"] = "word"
                word["STR"] = "UNK"
            elif len(row) > 8 and (row[7] != '0' or row[2] != '0'):
                str_word = ""
                str_word2 = ""
                if row[7] != '0':
                    str_word = token_map[row[7]]
                if row[2] != '0':
                    str_word2 = token_map[row[2]]
                if "Number_" in str_word and str_word in number_words:
                    word["Number"] = "metadata"
                if "s2l_" in str_word: str_word = str_word[4:]
                if "s2l_" in str_word2: str_word2 = str_word2[4:]
                if "WordPos_" in str_word: str_word = str_word[8:]
                if "Base_Word_" in str_word: str_word = str_word[10:]
                #print(str_word,str_word2)
                if True or str_word2 not in stop_words:
                    if str_word != "":
                        word[str_word] = "word"
                    else:
                        word[str_word2] = "word"
                    if str_word2 != "":
                        word["STR"] = str_word2
                    else:
                        word["STR"] = str_word
                else:
                    continue
            else:
                continue

            if row[3] != '0':
                word[token_map[row[3]]] = "pos" # 3 is B_Tag   5 is POS
            if row[9] != '0':
                word[token_map[row[9]]] = "s/p"
            word["END"] = row[8]
            i = 10
            while i < len(row) and row[i] != "99999996":  # complex pos
                if row[i] != '0':
                    if "Concept" in token_map[row[i]] or "_Case" in token_map[row[i]]:
                        word[token_map[row[i]]] = "metadata" # I think that these missing
                    else:
                        word[token_map[row[i]]] = "pos2" # I think that these missing
                i += 1
            i += 1
            while i < len(row) and row[i] != "99999997":  # suffixes
                if row[i] != '0':
                    if row[i] in token_map:
                        word[token_map[row[i]]] = "sfx"
                i += 1
            i += 1
            while i < len(row) and row[i] != "99999998":  # stanford dependencies
                if row[i] != '0':
                    word[token_map[row[i]]] = row[i+1]
                i += 2
            i += 1

            while i < len(row) and row[i] != "99999999":  # metadata
                if row[i] != '0':
                    if row[i] in token_map:
                        word[token_map[row[i]]] = "metadata"
                i += 1
            corpus.append(word)
            j+=1

        for word in corpus:
            tmp = list(word.keys())
            for w in tmp:
                if w in short_map:
                    t = word[w]
                    del word[w]
                    word[short_map[w]] = t
                #else:
                #    print(w)

        for word in corpus:
            tmp = list(word.keys())
            del_list = []
            for w in tmp:
                if "boths" in mode.lower() or "parse" in mode.lower():
                    if w == "STR":
                        if word[w][0] in "0123456789" or word[w][-1] in "0123456789":
                            word["NUM_STR"] = "word"
                            if word[w] in word:
                                #word["UNK"] = word[w]
                                del_list.append(word[w])
                            word[w] = "NUM"
                        elif word[w] in ["g", "y","f","k","n","j","l","e","w", "u","r","o","v","q", "b","p","x","c","h","z"]:
                            word["UNK_STR"] = "word"
                            if word[w] in word:
                                del_list.append(word[w])
                            word[w] = "UNK"
                        else:
                            word[word[w]+"_STR"] = "word"
                    elif word[w] == "word":
                        if word[w][0] in "0123456789" or word[w][-1] in "0123456789":
                            word["NUM"] = "base"
                            del_list.append(w)
                        elif word[w] in ["g", "y","f","k","n","j","l","e","w", "u","r","o","v","q", "b","p","x","c"]:
                            word["UNK"] = "base"
                            del_list.append(w)
                        else:
                            word[w] = "base"
                    elif "lemma_" in w:
                        if w[6] in "0123456789" or w[-1] in "0123456789":
                            del_list.append(w)
                        elif len(w) == 7 and w[6] in ["g", "y","f","k","n","j","l","e","w", "u","r","o","v","q", "b","p","x","c"]:
                            del_list.append(w)
            for d in del_list:
                del word[d]

    if in_rag and not in_rag_blanket:
        for word in corpus:
            a,b,c = get_primary_keys(word,mode,3,True)

    if not in_rag and not in_rag_blanket:
        with open('text_vectors.json', 'w') as f:
            json.dump(corpus, f, indent='\t')
    word_to_index2,index_to_word2,corpus2,vocab_size2,length_of_corpus2,feature_strings2,ppn_set2,freq1,freq2,mod_freq,output_strings,corpus_effective_len,blanket_drop = get_corpus_metadata(corpus_file, corpus,ppn_set,mode,X_dimension, max_position_embeddings,True,in_rag,in_rag_blanket)

    del corpus
    del blanket_drop
    corpus = []
    blanket_drop = []

    return word_to_index2,index_to_word2,corpus2,vocab_size2,length_of_corpus2,feature_strings2,ppn_set2,mod_freq,output_strings,corpus_effective_len


def get_parse_links(word,corpus_id_2_word):
    heads = {}
    complements = {}
    word_id = word["ID"]
    for f in word:
        if f in ["ID","END"]:
            continue
        if not isinstance(word[f],list) and word[f].isnumeric() and word[f] in corpus_id_2_word and word[f] != word_id:
            if f[0] == "h":
                heads[word[f]] = 1
            else:
                complements[word[f]] = 1
    return heads,complements

def generate_training_data2(label,corpus,window_size,vocab_size,word_to_index,label_to_index,words_freq,mode):

    torch.manual_seed(seed)
    np.random.seed(seed)

    corpus_id_2_parse = {}
    corpus_parse_2_id = {}
    corpus_id_2_word = {}
    total_context = None
    print("gen_training_data2 ",vocab_size,len(word_to_index))

    for word in corpus:
        word_id = word["ID"]
        if int(word_id) < 10:
            continue
        corpus_id_2_word[word_id] = word

    if enable_pickle and "torch" in mode:
        pickle_target_words = open(label+"target_words.pickle", "wb")
        pickle_trgt_word_vectors = open(mode+"_"+label+"trgt_word_v.pickle", "wb")
        pickle_ctxt_word_vectors = open(mode+"_"+label+"ctxt_word_v.pickle", "wb")
    training_data =  []
    target_words = []
    trgt_word_vectors = []
    ctxt_word_vectors = []
    out_dim = 0
    freq_total = 1
    filtered_count = 0
    for w in words_freq:
        freq_total += words_freq[w]
    #avg_freq = freq_total / len(words_freq)
    threshold = 1 / len(words_freq)
    print(len(corpus)," training_Data2 pre subsampling")
    for i,word in enumerate(corpus):
        #print(i,word,"training_data2")
        index_target_word = i
        word_id = word["ID"]
        word_primary_key2 = word["primary_keys"][1]
        #print(word_primary_key2)
        #print(words_freq[word_primary_key2])

        #cut this off at the ends so that every set of vectors is the same size, which is the window*2
        if i <= window_size or i + window_size >= len(corpus): continue

        # subsampling from 'Distributed Representaions of Word and Phrases and their Compositionality'
        #prob_sample = ( 1 - np.sqrt( threshold / ( words_freq[word_primary_key2] / freq_total ) ) )
        #print(prob_sample,word_primary_key2,words_freq[word_primary_key2],words_freq[word_primary_key2] / freq_total,freq_total,avg_freq)
        #if prob_sample > np.random.uniform(0.0,1.0):
            #print(word_primary_key2,words_freq[word_primary_key2])
        #    continue
        #    pass
        filtered_count += 1
        #print(word_primary_key2,words_freq[word_primary_key2])

        if "context" in mode or "combined" in mode:
            target_word = word
            context_words = []
            context_words2 = []
            if True:
                #Before the middle target word
                before_target_word_index = index_target_word - 1
                for x in range(before_target_word_index, before_target_word_index - window_size , -1):
                    if x >=0:
                        context_words.extend([corpus[x]])

                # the middle word is an empty set
                #if "double" in mode:
                #    context_words.append({})

                #After the middle target word
                after_target_word_index = index_target_word + 1
                for x in range(after_target_word_index, after_target_word_index + window_size):
                    if x < len(corpus):
                        context_words.extend([corpus[x]])

            trgt_word_vector,ctxt_word_vector,ctxt_word_vector_list = get_feature_vectors(target_word,context_words,vocab_size,word_to_index,mode,0)
            #print("T",target_word,*trgt_word_vector[:80])
            #print("C",context_words,*ctxt_word_vector[:80])
            if len(ctxt_word_vector_list) != (window_size * 2): print(len(ctxt_word_vector_list))
            if "combined" not in mode:
                if "torchsmpl" in mode:
                    out_dim = len(ctxt_word_vector_list[0])
                    for ctxt in ctxt_word_vector_list:
                        #training_data.append([target_word,trgt_word_vector,ctxt_word_vector])
                        target_words.append(target_word)
                        trgt_word_vectors.append(trgt_word_vector)
                        ctxt_word_vectors.append(ctxt)
                        #print(*trgt_word_vector,"11")
                        #print(list_to_num_list(trgt_word_vector),"22")
                else:
                    out_dim = len(ctxt_word_vector)
                    #training_data.append([target_word,trgt_word_vector,ctxt_word_vector])
                    target_words.append(target_word)
                    trgt_word_vectors.append(trgt_word_vector)
                    ctxt_word_vectors.append(ctxt_word_vector)
                    #print(*trgt_word_vector,"11")
                    #print(list_to_num_list(trgt_word_vector),"22")
                if enable_pickle and "torch" in mode and len(target_words) > 4000:
                    for p_i in range(0,len(target_words)):
                        pickle.dump(target_words[p_i], pickle_target_words)
                        pickle.dump(trgt_word_vectors[p_i], pickle_trgt_word_vectors)
                        pickle.dump(ctxt_word_vectors[p_i], pickle_ctxt_word_vectors)
                    target_words = []
                    trgt_word_vectors = []
                    ctxt_word_vectors = []

        if "parse" in mode or "combined" in mode:
            target_word = word
            context_words = []
            #context_words2 = []

            #print(corpus_id_2_parse[word["ID"]])
            heads,complements = get_parse_links(word,corpus_id_2_word)
            orig_contexts = copy.deepcopy(heads)
            for c in complements: orig_contexts[c] = 1

            # make a pass through all heads and complements to find their corresponding complements and heads and add to list of context words
            j = 1
            while j<(window_size/2) and j<=3:
                heads0 = copy.deepcopy(heads)
                complements0 = copy.deepcopy(complements)
                for h in heads0:
                    heads1,complements1 = get_parse_links(corpus_id_2_word[h],corpus_id_2_word)
                    for h1 in heads1: heads[h1] = 1
                    for c1 in complements1: complements[c1] = 1
                for c in complements0:
                    heads1,complements1 = get_parse_links(corpus_id_2_word[c],corpus_id_2_word)
                    for h1 in heads1: heads[h1] = 1
                    for c1 in complements1: complements[c1] = 1
                j+=1

            all_context = heads
            for c in complements:
                if c != word:
                    all_context[c] = 1

            # add parse elements that are parallel
            if i>50 and i<len(corpus)-50:
                for f in word:
                    if (window_size/2) < 4: continue
                    corpus_list_pre = []
                    corpus_list_post = []
                    if not isinstance(word[f],list) and word[f].isnumeric() and word[f] in corpus_id_2_word and word[f] != word_id:
                        for j in range(i-50,i):
                            if f in corpus[j]: corpus_list_pre.append(corpus[j][f])
                        for j in range(i,i+50):
                            if f in corpus[j]: corpus_list_post.append(corpus[j][f])
                        j = 4
                        while j<=6 and len(corpus_list_pre) > j-4 and j <= (window_size/2):
                            all_context[corpus_list_pre[-1*(j-3)]] = 1
                            j+=1
                        j = 4
                        while j<=6 and len(corpus_list_post) > j-4 and j <= (window_size/2):
                            all_context[corpus_list_post[(j-4)]] = 1
                            j+=1

                # look at surrounding subj-verb pairs
                for f in ["nsubj","nsubjpass","hnsubj","hnsubjpass"]:
                    if (window_size/2) < 7: continue
                    corpus_list_pre = []
                    corpus_list_post = []
                    if not isinstance(word[f],list) and word[f].isnumeric() and word[f] in corpus_id_2_word and word[f] != word_id:
                        for j in range(i-50,i):
                            if f in corpus[j]: corpus_list_pre.append(corpus[j][f])
                        for j in range(i,i+50):
                            if f in corpus[j]: corpus_list_post.append(corpus[j][f])
                        j = 7
                        while j<=9 and len(corpus_list_pre) > j-7 and j <= (window_size/2):
                            all_context[corpus_list_pre[-1*(j-6)]] = 1
                            j+=1
                        j = 7
                        while j<=9 and len(corpus_list_post) > j-7 and j <= (window_size/2):
                            all_context[corpus_list_post[(j-7)]] = 1
                            j+=1
                '''
                # look for time / place context
                for f in ["Concept_time","Concept_where"]:
                    if (window_size/2) < 10: continue
                    corpus_list_pre = []
                    corpus_list_post = []
                    if not isinstance(word[f],list) and word[f].isnumeric() and word[f] in corpus_id_2_word and word[f] != word_id:
                        for j in range(i-50,i):
                            if f in corpus[j]: corpus_list_pre.append(corpus[j][f])
                        for j in range(i,i+50):
                            if f in corpus[j]: corpus_list_post.append(corpus[j][f])
                        j = 10
                        while j<=12 and len(corpus_list_pre) > j-10 and j <= (window_size/2):
                            all_context[corpus_list_pre[-1*(j-9)]] = 1
                            j+=1
                        j = 10
                        while j<=12 and len(corpus_list_post) > j-10 and j <= (window_size/2):
                            all_context[corpus_list_post[(j-10)]] = 1
                            j+=1
                '''
            for a in all_context:
                context_words.append(corpus_id_2_word[a])
            #print(word,context_words,"\n--------------\n")
            #context_words.append({})
            #context_words.extend(context_words2)
            if len(context_words) <= 1:
                continue
            offset = 0
            if "combined" in mode:
                offset=vocab_size*2
            trgt_word_vector,ctxt_word_vector2,ctxt_word_vector2_list = get_feature_vectors(target_word,context_words,vocab_size,word_to_index,mode,offset)
            if "torchsmpl" in mode:
                if "combined" in mode:
                    ctxt_word_vector2_list.extend(ctxt_word_vector_list)
                out_dim = len(ctxt_word_vector2_list[0])
                for ctxt in ctxt_word_vector2_list:
                    #print("T",target_word,*trgt_word_vector[:800])
                    #print("C",context_words,*ctxt_word_vector[:800])
                    #training_data.append([target_word,trgt_word_vector,ctxt_word_vector2])
                    target_words.append(target_word)
                    trgt_word_vectors.append(trgt_word_vector)
                    ctxt_word_vectors.append(ctxt)
                    #print(i,len(target_words),"training_data2")

            else:
                if "combined" in mode:
                    ctxt_word_vector2 += ctxt_word_vector
                out_dim = len(ctxt_word_vector2)
                #print("T",target_word,*trgt_word_vector[:800])
                #print("C",context_words,*ctxt_word_vector[:800])
                #training_data.append([target_word,trgt_word_vector,ctxt_word_vector2])
                target_words.append(target_word)
                trgt_word_vectors.append(trgt_word_vector)
                ctxt_word_vectors.append(ctxt_word_vector2)
                #print(i,len(target_words),"training_data2")
            if enable_pickle and "torch" in mode and len(target_words) > 4000:
                for p_i in range(0,len(target_words)):
                    pickle.dump(target_words[p_i], pickle_target_words)
                    pickle.dump(trgt_word_vectors[p_i], pickle_trgt_word_vectors)
                    pickle.dump(ctxt_word_vectors[p_i], pickle_ctxt_word_vectors)
                target_words = []
                trgt_word_vectors = []
                ctxt_word_vectors = []

            #if total_context is None:
            #    total_context = ctxt_word_vector2
            #total_context += ctxt_word_vector2
            #print("G",*total_context[:800])
    print(filtered_count," training_Data2 post subsampling")

    if enable_pickle and "torch" in mode:
        for p_i in range(0,len(target_words)):
            pickle.dump(target_words[p_i], pickle_target_words)
            pickle.dump(trgt_word_vectors[p_i], pickle_trgt_word_vectors)
            pickle.dump(ctxt_word_vectors[p_i], pickle_ctxt_word_vectors)
        target_words = []
        trgt_word_vectors = []
        ctxt_word_vectors = []

        pickle_target_words.close()
        pickle_trgt_word_vectors.close()
        pickle_ctxt_word_vectors.close()
        #exit(1)
        return label, out_dim, label_to_index
    else:
        return [target_words,trgt_word_vectors,ctxt_word_vectors], out_dim, label_to_index


def transformer_traintest(trgt_word_vectors,ctxt_word_vectors,trgt_label,ctxt_label,len_X,batch_size,model,y_dimension, word_total, max_position_embeddings, X_dimension, model_tune, test_ids, target_label_unions, mode, loader, data_words, words_freq, valid_indexes, valid_mask, output_strings, rag_blanket, indexes_masked, context_cache=None,train=False,f1=False,word_to_index={},word_2_words={},model_label=0):
    if enable_pickle:
        in_pickle_trgt_word_vectors = open(mode+"_"+trgt_label+"trgt_word_v.pickle", "rb")
        in_pickle_ctxt_word_vectors = open(mode+"_"+ctxt_label+"ctxt_word_v.pickle", "rb")
        if rag_blanket_active:
            in_pickle_blkt_word_vectors = open(mode+"_"+label+"blkt_word_v.pickle", "rb")
    #sub_batch_size = 100
    #sub_batches = int(batch_size / sub_batch_size)
    if enable_memmap:
        fi_read = np.memmap(mode+"_start_ctxt_word_v.dat", dtype=np.int32, mode='r', shape=(word_total,max_position_embeddings,X_dimension))

    len_train = int(len_X * 0.95)
    total_loss = 0
    count = 1
    start = 0
    end = len_train
    correct = incorrect = correct1 = incorrect1 = 0
    correct2 = incorrect2 = correct3 = incorrect3 = 0
    valid_outputs = 0
    uniq2 = uniq3 = 0
    uniq_words = {}
    results_obj = {}
    uniq_correct2 = {}
    uniq_correct3 = {}
    loss = 0
    losses = []
    output_list = []
    index_to_word = {value: key for key, value in word_to_index.items()}
    #batch_size = 10
    load_iter = None
    if enable_dataloader1:
        load_iter = iter(loader)

    freq_total = 1
    for w in words_freq:
        freq_total += words_freq[w]
        #print(w)
    threshold = 0.5
    if len(words_freq) > 0:
        threshold = 1 / len(words_freq)

    #uniq_ids = {}
    #for t in test_ids:
    #    uniq_ids[t] = 1
    #print(uniq_ids,"uniq_ids")

    if not train and "testset" not in mode and "tuning" not in mode:
        start = len_train
        end = len_X

        ### TEST
        if enable_pickle:
            for i in range(0,len_train):
                a = pickle.load(in_pickle_trgt_word_vectors)
                a = pickle.load(in_pickle_ctxt_word_vectors)

    if "testset" in mode or "tuning" in mode:
        end = len_X

    if "testset" in mode and "w_" in mode:
        perplexity = True
    else:
        perplexity = None
    last_i = -10000000

    #len_w = len(word_to_index)
    len_w = 0
    for w in word_to_index:
        if is_target_feature(w,output_strings):
            if word_to_index[w] > len_w:
                len_w = word_to_index[w]
            #print(w,word_to_index[w])
    len_w += 1
    #for w in word_to_index:
    #    if not is_target_feature(w):
    #        print("NOT",w,word_to_index[w])

    #for w in sorted(word_to_index.keys()):
    #    if word_to_index[w] < len_w:
    #        print(w,word_to_index[w])

    #for i in range(start,end,batch_size):
    i = start
    num_sampled = 0
    while i < end:
        start_i = i
        test_ids_batch = []
        Xa = []
        ya = []
        batch_blanket = []
        #ii = i
        j = 0
        last_set = i+batch_size >= end
        #if not enable_pickle:
        #    Xa = trgt_word_vectors[i:i+batch_size]
        #    ya = ctxt_word_vectors[i:i+batch_size]
        if enable_pickle:
            try:
                Xa = pickle.load(in_pickle_trgt_word_vectors)
                if enable_dataloader1:
                    contexts = next(load_iter).squeeze(0)
                else:
                    if enable_memmap:
                        ya = fi_read[i:i+len(Xa), :, :]
                    else:
                        ya = pickle.load(in_pickle_ctxt_word_vectors)
                if rag_blanket_active:
                    batch_blanket = pickle.load(in_pickle_blkt_word_vectors)
                i += batch_size
                num_sampled += batch_size
            except:
                i = end + 1
                continue
        else:
            while j<batch_size and i < end:
                try:
                    word_i = data_words[i]+"_STR"
                    if False and train and word_i in words_freq:
                        prob_sample = ( 1 - np.sqrt( threshold / ( words_freq[word_i] / freq_total ) ) )
                        if prob_sample > np.random.uniform(0.0,1.0):
                            i+=1
                            continue

                    if enable_pickle:
                        Xa.append(pickle.load(in_pickle_trgt_word_vectors))
                        ya.append(pickle.load(in_pickle_ctxt_word_vectors))
                        if rag_blanket_active:
                            batch_blanket.append(pickle.load(in_pickle_blkt_word_vectors))
                    else:
                        Xa.append(trgt_word_vectors[i])
                        ya.append(ctxt_word_vectors[i])
                        if rag_blanket_active:
                            batch_blanket.append(rag_blanket[i])
                    if f1 and i<end:
                        test_ids_batch.append(test_ids[i])
                    i+=1
                    j+=1
                    num_sampled += 1
                except EOFError:
                    i+=1
                    j+=1
                    break
        #print(targets.shape,"targets shape")
        #print(contexts.shape,"contexts shape")
        #na = []
        #for x in Xa:
        #    na.append(get_neg_samples1(x,neg_multiplier))
        #negative_samples = torch.LongTensor(na)
        #if start_i < 400:
        #    tmp_y = []
        #    for iy in range(len(ya[0])):
        #        tmp_y.append(ya[0][iy][0])
        #    print(start_i,tmp_y,"ya 0")

        #for sub_batch_i in range(sub_batches):
        #    sub_start = sub_batch_i * sub_batch_size
        #    sub_end = (sub_batch_i + 1) * sub_batch_size
        #    Xa = Xa0[sub_start:sub_end]
        #    ya = ya0[sub_start:sub_end]
        #    batch_blanket = batch_blanket0[sub_start:sub_end]
        #    if len(Xa) == 0:
        #        continue


        if train:
            #if i < 500:
            #    print("----A",i,len(Xa))
            if "tuning" in mode and not "tune0" in mode and not("tune1" in mode):
                #print("----A",i)
                #targets = torch.FloatTensor(Xa).to(device)
                contexts = torch.from_numpy(np.array(ya)).to(device=device, dtype=torch.int64)
                #contexts = torch.tensor(ya, dtype=torch.long, device=device)
                #print(contexts[:4],"contexts traintest")
                total_loss += model_tune.train(contexts,Xa,i,context_cache,mode,label)
                del contexts
            else:
                #print("----B",i)

                #targets = torch.tensor(Xa, dtype=torch.float, device=device)
                #print(Xa[:10],len(Xa))
                #print(batch_blanket[0],"batch blanket 0")
                targets = torch.zeros(len(Xa), len_w, dtype=torch.float, device="cpu")
                blanket = None
                if rag_blanket2_active:
                    blanket = torch.ones(len(Xa), len_w, dtype=torch.float, device="cpu")
                    value = 4.0
                    XXa_r0 = []
                    XXa_c0 = []
                    Xb = []
                    last_iii = 0
                    #print(Xa[:10],"Xa 10")
                    for ii in range(len(batch_blanket)):
                        #print(ii,Xa[ii],"Xa ii")
                        #print(Xa[ii],"traintest Xa")
                        iii = list(batch_blanket[ii].keys())
                        #print(iii,"iii")
                        if len(iii) == 0:
                            continue
                        #print(iii,"batch_blanket iii")
                        #iii = list(batch_blanket[ii].keys())
                        if iii[0] > len_w:
                            if iii[0] in index_to_word:
                                print(iii,index_to_word[iii[0]])
                            else:
                                print(iii,len_w,"out of range for index to word")
                            continue
                        XXa_r0.extend([ii]*len(iii))
                        XXa_c0.extend(iii)
                        last_iii = ii
                    #print(last_iii,len_w,len(XXa_r0),len(XXa_c0))
                    #print(XXa_r0,XXa_c0)
                    #print(blanket.shape)
                    if hard_rag_blanket:
                        value = 16.0
                        #blanket[torch.tensor(XXa_r0, dtype=torch.long, device="cpu"),torch.tensor(XXa_c0, dtype=torch.long, device="cpu")] = 16.0
                    elif rag_blanket0_active:
                        value = 1.0
                    else:
                        value = 1.1
                    blanket[torch.tensor(XXa_r0, dtype=torch.long, device="cpu"),torch.tensor(XXa_c0, dtype=torch.long, device="cpu")] = value
                    #print(blanket.shape)
                    #print(blanket[last_iii-1],"blanket iii-1")
                    #print(blanket[last_iii],"blanket iii")
                    #print(blanket[last_iii+1])
                    #print(blanket)
                #XXa = []
                #for ii in range(len(Xa)):
                #    for iii in Xa[ii].keys():
                #        XXa.append([ii,iii])
                #print(XXa[:10],"XXa",len(XXa))
                #print(targets.shape,"targets shape")
                #XXa_r = torch.tensor([x[0] for x in XXa], dtype=torch.long, device="cpu")
                #XXa_c = torch.tensor([x[1] for x in XXa], dtype=torch.long, device="cpu")
                #targets[XXa_r,XXa_c] = 1.0

                XXa_r0 = []
                XXa_c0 = []
                Xb = []
                #print(Xa[:10],"Xa 10")
                for ii in range(len(Xa)):
                    #print(ii,Xa[ii],"Xa ii")
                    #print(Xa[ii],"traintest Xa")
                    iii = list(Xa[ii].keys())
                    if iii[0] > len_w:
                        print(iii,index_to_word[iii[0]])
                    XXa_r0.extend([ii]*len(iii))
                    XXa_c0.extend(iii)
                    #Xb.append(indexes_masked[iii[0]])
                    #print(Xa[ii],Xb[ii],"Xa Xb")

                ######### try creating a list of lists in memory and then creating the tenson on device in 1 go ##########
                targets[torch.tensor(XXa_r0, dtype=torch.long, device="cpu"),torch.tensor(XXa_c0, dtype=torch.long, device="cpu")] = 1.0
                del XXa_r0
                del XXa_c0
                #print(targets[1],"targets")
                targets=targets.to(device, non_blocking=True)
                if rag_blanket2_active:
                    blanket = blanket.to(device, non_blocking=True)

                #print(targets[:10,:100],"targets")
                #if early_fusion:
                #    targets = targets.squeeze(1)
                if enable_dataloader1:
                    contexts = contexts.to(device, non_blocking = True)
                    #print(contexts)
                    #print(contexts.shape)
                else:
                    #contexts = torch.tensor(ya, dtype=torch.long, device=device)
                    contexts = torch.from_numpy(ya).to(device=device, dtype=torch.int64)
                    del ya
                #print(targets)
                #print(targets.shape)
                total_loss += model.trainA(contexts,targets,blanket,i,context_cache,mode,trgt_label)
                del contexts
                del targets
                torch.cuda.empty_cache()
            #optimizer.zero_grad()
            #loss,losses = model(target, context, True)
            #loss.backward()
            #optimizer.step()
        else:
            #targets = torch.tensor(Xa, dtype=torch.float)
            #print(Xa[:10],len(Xa))

            blanket = None
            if rag_blanket_active:
                blanket = torch.ones(len(Xa), len_w, dtype=torch.float, device="cpu")
                XXa_r0 = []
                XXa_c0 = []
                for ii in range(len(batch_blanket)):
                    iii = list(batch_blanket[ii].keys())
                    if len(iii) == 0:
                        continue
                    if iii[0] > len_w:
                        if iii[0] in index_to_word:
                            print(iii,index_to_word[iii[0]])
                        else:
                            print(iii,len_w,"out of range for index to word")
                        continue
                    XXa_r0.extend([ii]*len(iii))
                    XXa_c0.extend(iii)

                if hard_rag_blanket:
                    value = 16.0
                    #blanket[torch.tensor(XXa_r0, dtype=torch.long, device="cpu"),torch.tensor(XXa_c0, dtype=torch.long, device="cpu")] = 16.0
                elif rag_blanket0_active:
                    value = 1.0
                else:
                    value = 1.1
                blanket[torch.tensor(XXa_r0, dtype=torch.long, device="cpu"),torch.tensor(XXa_c0, dtype=torch.long, device="cpu")] = value

            targets = torch.zeros(len(Xa), len_w, dtype=torch.float, device="cpu")
            #print(indexes_masked)
            XXa = []
            XXa_r0 = []
            XXa_c0 = []
            Xb = []
            for ii in range(len(Xa)):
                iii = list(Xa[ii].keys())
                XXa_r0.extend([ii]*len(iii))
                XXa_c0.extend(iii)
                #Xb.append(indexes_masked[iii[0]])
                #print(Xa[ii],Xb[ii],"Xa Xb")
                #for iii in Xa[ii].keys():
                #    XXa.append([ii,iii])
            #exit()
            #print(XXa[:10],"XXa")
            #print(targets.shape,"targets shape")
            #XXa_r = torch.tensor([x[0] for x in XXa], dtype=torch.long, device=device)
            #XXa_c = torch.tensor([x[1] for x in XXa], dtype=torch.long, device=device)
            targets[torch.tensor(XXa_r0, dtype=torch.long, device="cpu"),torch.tensor(XXa_c0, dtype=torch.long, device="cpu")] = 1.0
            del XXa_r0
            del XXa_c0
            #print(Xa[:10],"Xa")
            #print(targets[:10,:100],"targets")
            #print(targets[:10,:100],"targets")
            targets=targets.to(device)
            if rag_blanket_active:
                blanket = blanket.to(device, non_blocking=True)

            if enable_dataloader1:
                contexts = contexts.to(device, non_blocking = True)
            else:
                #contexts = torch.tensor(ya, dtype=torch.long, device=device)
                contexts = torch.from_numpy(ya).to(device=device, dtype=torch.int64)
                del ya
            if f1:
                #print("-----------9a")
                if "zero_shot" in mode or "reg_test0" in mode or "reg_test1" in mode:
                    #print("----xC",i)

                    loss,losses, correctA, incorrectA, correctB, incorrectB, correctC, incorrectC, correctD, incorrectD, uniqC, uniqD, num_valid_outputs, outA =   model.testA(contexts,targets,blanket,Xa,Xb,test_ids_batch, target_label_unions, mode,perplexity,y_dimension,i,context_cache,f1, results_obj, uniq_correct2, uniq_correct3, last_set, valid_indexes,valid_mask,index_to_word,word_2_words,uniq_words)
                    output_list.extend(outA)
                    #print("-----------9x",loss)
                else:
                    #print("----xD",i)
                    loss,losses, correctA, incorrectA, correctB, incorrectB, correctC, incorrectC, correctD, incorrectD, uniqC, uniqD, num_valid_outputs = model_tune.test(contexts, targets, test_ids_batch, target_label_unions, mode, perplexity, y_dimension,i, context_cache, f1, results_obj, uniq_correct2, uniq_correct3,last_set)
                    #print("-----------9y")
                #print("-----------9aa")
                correct += correctA
                incorrect += incorrectA
                correct1 += correctB
                incorrect1 += incorrectB
                correct2 = correctC
                incorrect2 = incorrectC
                correct3 = correctD
                incorrect3 = incorrectD
                uniq2 = uniqC
                uniq3 = uniqD
                valid_outputs = num_valid_outputs
                #print("-----------9ab")

            else:
                if "pretrain" in mode:
                    #print("----E",i)
                    loss,losses = model.testA(contexts,targets,Xa,Xb,test_ids,target_label_unions,mode,perplexity,y_dimension,i,context_cache,f1)
                    #print("----Ea",i)
                elif "tuning" not in mode:
                    #print("----F",i)
                    loss,losses = model_tune.test(contexts,targets,test_ids,target_label_unions,mode,perplexity,y_dimension,i,context_cache,f1)
                    #print("----Fa",i)

            #print("-----------9c")
            total_loss += loss * len(contexts)
            del contexts
            del targets
            #print(total_loss)
            #loss,losses = model(target, context, True)
        #print("-----------9d")

        #total_loss += loss.item() * len(target)
        #print(total_loss,end-start)
        count += len(Xa)
        del Xa
        if start_i - last_i > (end/4):
            last_i = start_i
            print("------------------------------------------------------transformer-train batch",start_i,"of",start,"to",end)
    print("------------------------------------------------------transformer-train batch total sampled",num_sampled,"of",end)

    avg_loss = total_loss/(1+end-start)
    if "testset" in mode:
        print(avg_loss,"avg_loss",np.exp(avg_loss),"exp loss",total_loss,1+end-start)
        uniq0 = sorted(list(uniq_words.keys()))
        if len(uniq0) > 50:
            uniq0 = uniq0[:50]
        print(uniq0,len(uniq_words),"uniq words")
        uniq2 = len(uniq_words)
    if enable_pickle:
        in_pickle_trgt_word_vectors.close()
        in_pickle_ctxt_word_vectors.close()
        if rag_blanket_active:
            in_pickle_blkt_word_vectors.close()

    if f1:
        #perplexity_f1 = str(round(np.exp(avg_loss),4))
        #print("------10",perplexity)
        #print("------11",perplexity.shape)
        perplexity_f1 = "0.0"
        if perplexity is not None:
            #perplexity_f1 = str(round(perplexity.compute().item(),4))
            perplexity_f1 = str(round(np.exp(avg_loss),4))
        #print("------12",perplexity_f1)

        end_time = time.time()
        diff_time = str(int(end_time - start_time))
        str1 = str(correct)+","+str(incorrect)+","+str(correct2)+","+str(incorrect2)+","+str(correct3)+","+str(incorrect3)+","+str(uniq2)+","+str(uniq3)+","+str(correct1)+","+str(incorrect1)+","+perplexity_f1+","+str(valid_outputs)+","+diff_time+","+mode+","+str(model_label)
        print(str1)
        fo = open(mode+"_ms.log","a")
        fo.write(str1+"\n")
        fo.close()
        fo = open("test_results.csv","a")
        fo.write(str1+"\n")
        fo.close()
        #print("-----------90")
        return round(total_loss / count,6), correct, incorrect, output_list
    #print("-----------91")

    return round(total_loss / count,6)


def test(training_dataR,training_data,model_file_in,config,word_to_index,test_ids,target_label_unions,mode,valid_indexes,word_2_words, words_freq, previous_predictions, previous_predictions_file_out,rag_position_embeddings, prediction_map, output_strings, rag_blanket, batch_size, word_total, max_position_embeddings, X_dimension, loader):

    model_fields = model_file_in.split("#")
    model_max = int(model_fields[1])
    model_i = 4
    while model_i <= model_max:
        model_file_in_i = model_fields[0]+"#"+str(model_i)+"#"+model_fields[2]
        y_dimension = len(word_to_index)
        print("opening model file in test",model_file_in_i)
        try:
            in_pickle_model = open(model_file_in_i, "rb")
            word_to_index = pickle.load(in_pickle_model)
            output_strings = pickle.load(in_pickle_model)
            label_to_index = pickle.load(in_pickle_model)
            definition_picks = pickle.load(in_pickle_model)
            freqs = pickle.load(in_pickle_model)
            freqs_h = pickle.load(in_pickle_model)
            words_freq2_override = pickle.load(in_pickle_model)
            mod_freq = pickle.load(in_pickle_model)
            model = pickle.load(in_pickle_model)
            model_tune = pickle.load(in_pickle_model)
            in_pickle_model.close()
        except:
            model_i+=4
            continue
        model.to(device)

        ctxt_label = "start_"
        if rag_active:
            with torch.no_grad():
                torch.cuda.empty_cache()
                encode_retrieve(training_dataR,training_data[2],len(training_data[5]),word_to_index,rag_position_embeddings,batch_size,model,mode,False)
                torch.cuda.empty_cache()
                gc.collect()
            ctxt_label = "rag_"

        valid_indexes = {}
        #index_to_word = {}
        #for o in output_strings:
        #    o_str = o+"_STR"
        #    if o_str in word_to_index and is_target_feature(o_str):
        #        valid_indexes[word_to_index[o_str]] = 1

        for w in word_to_index:
            #index_to_word[word_to_index[w]] = w
            if is_target_feature(w,output_strings):
                valid_indexes[word_to_index[w]] = 1
        print("test num valid_indexes",len(valid_indexes),"num output strings",len(output_strings))
        max_valid = 0
        for v in valid_indexes:
            if v > max_valid: max_valid = v
        #print(valid_indexes)
        #for i in range(max_valid+1):
        #    print(i,index_to_word[i])

        len_w = 0
        for w in word_to_index:
            if is_target_feature(w,output_strings):
                if word_to_index[w] > len_w: len_w = word_to_index[w]
        len_w += 1

        valid2 = [False] * len_w

        for u in valid_indexes:
            valid2[u] = True
        valid_mask = torch.tensor(valid2, dtype=torch.bool).to(device)
        #print(*valid_mask)

        #indexes1 = torch.arange(len(word_to_index), dtype=torch.long, device=device)
        #print(indexes1,indexes1.shape,"indexes1")
        #indexes2 = torch.masked_select(indexes1,valid_mask)
        #indexes3 = indexes2.tolist()
        #print(indexes3)
        indexes_masked = []
        #for i in range(len(word_to_index)):
        #    if i in indexes3:
        #        j = indexes3.index(i)
        #        indexes_masked.append(j)
        #    else:
        #        indexes_masked.append(-1)
        #print(indexes2,indexes2.shape,"indexes2")
        #target_p2[0][0] = (torch.nonzero(torch.masked_select(target,valid_mask)))[0]


        if "negsmpl" in mode and "zero_shot" in mode:
            model.reset_classifier(y_dimension)
        #elif "transformer" in mode:
        #    model.reset_classifier(config,y_dimension)

        #batch_size = 25
        len_X = len(training_data[5])

        #print(training_data[1][:10],"training_data C")


        if "negsmpl" in mode:
            test_loss, correct, incorrect = negsmpl_traintest1("start_",len_X,batch_size, model, None, target_label_unions, mode, False, True)
        elif "transformer" in mode:
            test_loss, correct, incorrect, output_list = transformer_traintest(training_data[1],training_data[2],"start_",ctxt_label,len_X,batch_size, model,y_dimension, word_total, max_position_embeddings, X_dimension, model_tune, test_ids, target_label_unions, mode, loader, training_data[5], words_freq, valid_indexes,valid_mask, output_strings, rag_blanket, indexes_masked, None, False, True,word_to_index,word_2_words,model_i)

        str1 = "epoch NONE test_training_data "+str(len_X) + " " + str(test_loss)+" "+str(correct)+" "+str(incorrect)
        print(str1)
        fo = open(mode+"_ms.log","a")
        fo.write(str1+"\n")
        fo.close()
        #words = []
        #for t in training_data[1]:
        #    u = list(t.keys())
        #    #print(u,index_to_word[u[0]])
        #    if len(u) == 0:
        #        words.append("UNDEF")
        #    elif output_mode == "surface":
        #        words.append(index_to_word[u[0]][:-4])
        #    else:
        #        words.append(index_to_word[u[0]])

        #str1 = "\n".join(words)
        #fo = open(mode+"_targets.txt","w")
        #fo.write(str1+"\n")
        #fo.close()

        if pred_output_active:
            if len(previous_predictions) == 0:
                for i in range(len(output_list)):
                    output_list[i] = training_data[5][i]+","+output_list[i]
            else:
                for i in range(len(output_list)):
                    if i in prediction_map:
                        output_list[i] = previous_predictions[prediction_map[i]].strip()+","+output_list[i]
                    else:
                        output_list[i] = training_data[5][i]+","+output_list[i]

            pred_fields = previous_predictions_file_out.split("#")
            previous_predictions_file_out = pred_fields[0]+"#"+str(model_i)+"#"+pred_fields[2]

            print("test",folding_full,len(previous_predictions),len(output_list))
            if "testset" not in mode and "W" in folding_full and len(previous_predictions) > 0:
                output_list2 = output_list[:len(previous_predictions)]
            else:
                output_list2 = output_list
            with open(previous_predictions_file_out, "w") as fi:
                json.dump(output_list2, fi, indent=4)

            print(len(output_list2),"written to \n"+previous_predictions_file_out)
        model_i += 4
    if enable_pickle:
        subprocess.run(["rm", mode+"_start_trgt_word_v.pickle"])
        subprocess.run(["rm", mode+"_start_ctxt_word_v.pickle"])
        subprocess.run(["rm", mode+"_rag_word_v.pickle"])
        if enable_memmap:
            subprocess.run(["rm", mode+"_start_ctxt_word_v.dat"])
    return None,None,None


def train(training_dataR,word_embedding_dimension,window_size,epochs,training_data,training_data2,out_dim,learning_rate,vocab_size, word_to_index, label_to_index, target_label_unions, measurements_objects, measurements_freqs, corpus_grouped, top_n_words, words_subset,feature_strings,pos_map,skip_pos_list,definition_maps, words_freq, words_freq2, training_indexes, measurement_indexes, def_dimension, word_total, max_position_embeddings, X_dimension, y_dimension,rag_position_embeddings,neg_multiplier, definition_positives, ppn_set, targeted_tests, config, freqs, freqs_h, mod_freq, bases, model_file_in, model_file_out, output_strings, rag_blanket,batch_size, loader, mode):
    first = 0
    measure_list = {}

    measurements_obj = definition_map = None
    if "tuning" not in mode and 99 not in training_indexes:
        measurements_obj = measurements_objects[0]
        definition_map = definition_maps[0]
    assigned_map = {}
    training_i = -1
    measurement_i = 0
    all_indexes = [0,1,2,3,4,5,6,7,8]
    for i in all_indexes:
        assigned_map[i] = {}

    len_X_epoch = 0
    first = True
    neg_definitions = None
    neg_tensorsX = None
    neg_tensorsy = None
    neg_tensorsN = None
    neg_slices = None
    neg_y = None
    neg_defined_words = None
    neg_targets = None
    mod_training_cache = {}
    len_new_testing_data2 = []
    definition_picks = []

    if "transformer" in mode:
        len_X_epoch = 0
        len_X = 0
        num_negative_samples = 15  # Number of negative samples per positive sample
        learning_rate = 0.001

        last_change = 0
        saved_tensors = {}
        definition_picks1 = None

        context_cache=None
        if "tuning" in mode:
            context_cache = {}

        if "loadmodel" in mode:
            in_pickle_model = open(model_file_in, "rb")
            word_to_index = pickle.load(in_pickle_model)
            output_strings = pickle.load(in_pickle_model)
            label_to_index1 = pickle.load(in_pickle_model)
            definition_picks1 = pickle.load(in_pickle_model)
            freqs = pickle.load(in_pickle_model)
            freqs_h = pickle.load(in_pickle_model)
            words_freq2_override = pickle.load(in_pickle_model)
            mod_freq = pickle.load(in_pickle_model)
            model = pickle.load(in_pickle_model)
            model_tune = pickle.load(in_pickle_model)
            in_pickle_model.close()
            model.to(device)
            if not zero_shot_forced and ("tuning" in mode or "tune_only" in mode or "fine_tune" in mode):
                model.reset_classifier(config)
            del label_to_index1
            del definition_picks1
        else:
            options = {}
            if rag_blanket2_active:             options["rag_blanket2_active"] = 1
            if rag_blanket2_active:             options["rag_blanket_active"] = 1
            if inattention_active:              options["inattention_active"] = 1
            if inattention_all_active:          options["inattention_all_active"] = 1
            if inattention_cls_active:          options["inattention_cls_active"] = 1
            if inattention_cls1_active:         options["inattention_cls1_active"] = 1
            if inattention_cls2_active:         options["inattention_cls2_active"] = 1
            if inattention_cls3_active:         options["inattention_cls3_active"] = 1
            if inattention_ffn_active:         options["inattention_ffn_active"] = 1
            if inattention_sum_active:          options["inattention_sum_active"] = 1
            if radial_basis_all:                options["radial_basis_all"] = 1
            if radial_basis_active:             options["radial_basis_active"] = 1
            if radial_basis2_active:            options["radial_basis2_active"] = 1
            if radial_basis3_active:            options["radial_basis3_active"] = 1
            if radial_basis4_active:            options["radial_basis4_active"] = 1
            if headX_active:                    options["headX_active"] = 1
            if early_fusionH_active:            options["early_fusionH_active"] = 1
            if ragF_active:                     options["ragF_active"] = 1
            if ragG_active:                     options["ragG_active"] = 1
            if cosine_similarity_active:        options["cosine_similarity_active"] = 1
            if cosine1_similarity_active:       options["cosine1_similarity_active"] = 1
            if tanh_active:                     options["tanh_active"] = 1
            if sigmoid_active:             options["sigmoid_active"] = 1
            options["radial_basis_key"] = radial_basis_key
            options["window_size"] = window_size
            options["prune_size"] = prune_size
            options["rag_window_size"] = rag_window_size

            model = TransformerForPreTraining(config, options, mode)
            model_tune = None

        print(sorted(list(word_to_index))[600:640]," transformer sorted word to index")

        if "loadword" in mode:
            in_pickle_model = open(model_file_in, "rb")
            word_to_index = pickle.load(in_pickle_model)
            output_strings = pickle.load(in_pickle_model)
            label_to_index1 = pickle.load(in_pickle_model)
            definition_picks1 = pickle.load(in_pickle_model)
            word_model = pickle.load(in_pickle_model)
            in_pickle_model.close()
            word_model.to(device)
            del label_to_index1
            del definition_picks1

            embeddingsT,embeddingsTP = model.get_embeddings()
            mean_embeddingT = torch.mean(embeddingsT.weight) # Shape: [embedding_dim]
            stddevT = embeddingsT.weight.std()
            mean_embeddingTP = torch.mean(embeddingsTP.weight) # Shape: [embedding_dim]
            stddevTP = embeddingsTP.weight.std()

            embeddingsW = word_model.get_embeddings()
            mean_embeddingW = torch.mean(embeddingsW.weight) # Shape: [embedding_dim]
            stddevW = embeddingsW.weight.std()
            if abs(mean_embeddingT - mean_embeddingW) > 0.25 or abs(stddevT - stddevW) > 0.25:
                print(mean_embeddingT,stddevT,"Xavier init")
                print(mean_embeddingTP,stddevTP,"Xavier posi init")
                print(mean_embeddingW,stddevW,"Word2Vec init")
                print("mean or stddev mismatch")
                with torch.no_grad(): # Use torch.no_grad() to avoid tracking this operation for gradient calculations
                    embeddingsW.weight.data.mul_(stddevT) # Use in-place division for efficiency
                    embeddingsW.weight.data.div_(stddevW) # Use in-place division for efficiency
                mean_embeddingW = torch.mean(embeddingsW.weight) # Shape: [embedding_dim]
                stddevW = embeddingsW.weight.std()
                print(mean_embeddingW,stddevW,"Standardized")

            model.set_embeddings(embeddingsW)

        new_training_data2 = [None] * len(all_indexes)
        new_testing_data2 = [None] * len(all_indexes)
        len_new_testing_data2 = [None] * len(all_indexes)
        len_new_testing_data = [None] * len(all_indexes)
        for i in range(len(all_indexes)):
            new_training_data2[i] = [None] * 3
            new_testing_data2[i] = [None] * 3
            len_new_testing_data2[i] = 0
            len_new_testing_data[i] = 0
            definition_picks.append({})

        training_i = -1

        def_epochs = {}

        ctxt_label = "start_"

        for epoch in range(epochs):
            loss = 0
            loss2 = 0
            i = 0;
            count_i = 0

            if True:
                if "def" not in mode or epoch not in def_epochs:
                    total_loss = 0
                    count = 0
                    print("pre start traintest epoch"+str(epoch))

                    #if epoch != 0:
                    #training_data = reshuffle_training_data(original_training_data,word_to_index,words_freq,last_training_len,mode)
                    training_len = len(training_data[5])
                    #last_training_len = training_len
                    len_X_epoch = training_len   # len(training_data2[0])

                    total_loss = transformer_traintest(training_data[1],training_data[2],"start_",ctxt_label, training_len, batch_size, model,y_dimension, word_total, max_position_embeddings, X_dimension, model_tune,[], target_label_unions, mode, loader, training_data[5], words_freq, {}, None, output_strings, rag_blanket, None, context_cache, True, False,word_to_index,{})

                    #del training_data

                    str1 = "epoch "+str(epoch)+" training_data "+str(training_len) + " " + str(total_loss)
                    print(str1)
                    fo = open(mode+"_ms.log","a")
                    fo.write(str1+"\n")
                    fo.close()

            ########TRAINING
            training_loss = total_loss
            ########TEST
            test_lossA = 0
            count = 0

            if epoch % 4 == 3 or epoch == epochs-1:
                #fo.close()
                if "savemodel" in mode:
                    model_fields = model_file_out.split("#")
                    #model_max = int(model_fields[1])
                    #model_i = 4
                    #while model_i <= model_max:
                    model_file_out_i = model_fields[0]+"#"+str(epoch+1)+"#"+model_fields[2]
                    print("saving model to",model_file_out_i)
                    out_pickle_model = open(model_file_out_i, "wb")
                    out_pickle_model.write(pickle.dumps(word_to_index))
                    out_pickle_model.write(pickle.dumps(output_strings))
                    out_pickle_model.write(pickle.dumps(label_to_index))
                    out_pickle_model.write(pickle.dumps(definition_picks))
                    out_pickle_model.write(pickle.dumps(freqs))
                    out_pickle_model.write(pickle.dumps(freqs_h))
                    out_pickle_model.write(pickle.dumps(words_freq2))
                    out_pickle_model.write(pickle.dumps(mod_freq))
                    out_pickle_model.write(pickle.dumps(model))
                    out_pickle_model.write(pickle.dumps(model_tune))
                    out_pickle_model.close()

            if epoch > 0 and epoch % 8 != 3 and epoch != epochs-1: continue
            #test_losses_str = " ".join(str(x) for x in test_losses)
            #embeddings = model.embeddings.weight.detach().cpu().numpy()

            word_vectors = {}
            similar_words = {}
            theta_sums = {}
            word_matrix = {}
            pid = 999

        if enable_pickle:
            subprocess.run(["rm", mode+"_start_trgt_word_v.pickle"])
            subprocess.run(["rm", mode+"_start_ctxt_word_v.pickle"])
            subprocess.run(["rm", mode+"_start1_ctxt_word_v.pickle"])
            subprocess.run(["rm", mode+"_rag_ctxt_word_v.pickle"])
            subprocess.run(["rm", mode+"_rag_word_v.pickle"])
            if enable_memmap:
                subprocess.run(["rm", mode+"_start_ctxt_word_v.dat"])


        return None,None,None

    return epoch_loss, epoch_loss2, weights_input_hidden



def cosine_similarity1(word_vector_1,word_vector_2):
    theta_sum = np.dot(word_vector_1, word_vector_2)
    theta_den = np.linalg.norm(word_vector_1) * np.linalg.norm(word_vector_2)
    theta = theta_sum / theta_den
    return theta


def get_token_map():
    token_map = {}
    fi = open("../Scripts/token_string2long_map3.csv")
    lines = fi.readlines()
    for line in lines:
        fields = line.strip().split(",")
        token_map[fields[0]] = fields[1]
    fi.close()
    fi = open("../Scripts/resource_string2long_map.csv")
    lines = fi.readlines()
    for line in lines:
        fields = line.strip().split(",")
        token_map[fields[0]] = fields[1]
    fi.close()

    short_map = {}
    fi = open("../Scripts/parse_short_forms.csv")
    lines = fi.readlines()
    for line in lines:
        fields = line.strip().split(",")
        short_map[fields[0]] = fields[1]
    fi.close()

    return token_map, short_map


def get_primary_keys(word,mode,level=3,force=False):
    #if word["ID"] == "54138":
    #print(word,mode,"----------")
    if not force and "primary_keys" in word:
        return word["primary_keys"]
    idiom_mode = "idiom" in word
    #if primary_keys_dict is not None and "ID" in word and word["ID"] in primary_keys_dict[level]:
    #    return primary_keys_dict[level][word["ID"]]

    for f in word:
        if word[f] == "pos_word":
            #if primary_keys_dict is not None:
            #    primary_keys_dict[level][word["ID"]] = ["","",""]
            word["primary_keys"] = ["","",""]
            return "","",""
    primary_key = primary_key2 = primary_key3 = ""
    if level == 1:
        for f in word:
            if f == "primary_keys": continue
            if ("string" in mode or "parse" in mode) and f == "STR":
                primary_key = word[f]+"_STR"
                break
            elif f != "ID" and f != "STR" and "string" not in mode and "parse" not in mode:
                if word[f] == "word" and "zzzzzz_" not in f and "PPN_" not in f:
                    primary_key = f
                    break
        #if primary_keys_dict is not None:
        #    primary_keys_dict[level][word["ID"]] = [primary_key,"",""]
        word["primary_keys"] = [primary_key,"",""]
        return primary_key,"",""
    elif level == 2:
        primary_keys2 = {}
        for f in word:
            if f == "primary_keys": continue
            if ("string" in mode or "parse" in mode) and f == "STR" and not idiom_mode:
                primary_key = word[f]+"_STR"
                primary_keys2[word[f]+"_STR"] = 1
                break
            elif f != "ID" and f != "STR" and ("string" not in mode or "parse" not in mode or idiom_mode):
                if word[f] in ["word","sfx","base"]:
                    primary_keys2[f] = 1
                if word[f] == "word":
                    if "zzzzzz_" not in f and "PPN_" not in f:
                        primary_key = f
        primary_key2 = "|".join(sorted(primary_keys2.keys()))
        #if primary_keys_dict is not None:
        #    primary_keys_dict[level][word["ID"]] = [primary_key,primary_key2,""]
        word["primary_keys"] = [primary_key,primary_key2,""]
        return primary_key,primary_key2,""
    else:
        primary_keys2 = {}
        primary_keys3 = {}
        for f in word:
            if f == "primary_keys": continue
            if ("string" in mode or "parse" in mode) and f == "STR":
                primary_key = word[f]+"_STR"
                primary_keys2[word[f]+"_STR"] = 1
                primary_keys3[word[f]+"_STR"] = 1
            elif ("string" in mode or "parse" in mode) and (word[f] == "word" or word[f] == "idiom" or word[f] == "propn_phrase") and ("_STR" in f and "LIT_" not in f):
                if "zzzzzz_" not in f and "PPN_" not in f:
                    primary_key = f
                primary_keys2[f] = 1
                primary_keys3[f] = 1
            elif f != "ID" and f != "STR":
                if "string" in mode:
                    #print(f,word,"get_primary_keys")
                    #if word[f] in ["pos"] or "_STR" in f:
                    if "_STR" in f:
                        #print(f,word,"get_primary_keys=============================")
                        primary_keys3[f] = 1
                else:
                    #if word[f] in ["word","pos","pos2","s/p","sfx"]:
                    if f != "END":
                        primary_keys3[f] = 1
                    if word[f] in ["word","sfx","base"] and "string" not in mode and "parse" not in mode:
                        primary_keys2[f] = 1
                    if word[f] == "word" and "string" not in mode and "parse" not in mode:
                        if "zzzzzz_" not in f and "PPN_" not in f:
                            primary_key = f
        primary_key2 = "|".join(sorted(primary_keys2.keys()))
        primary_key3 = "|".join(sorted(primary_keys3.keys()))
        #if primary_keys_dict is not None:
        #    primary_keys_dict[level][word["ID"]] = [primary_key,primary_key2,primary_key3]
        word["primary_keys"] = [primary_key,primary_key2,primary_key3]
        #print(word)
        return primary_key,primary_key2,primary_key3

def get_corpus_grouped(corpus,mode):
    corpus_grouped = {}
    for word in corpus:
        primary_key,primary_key2,primary_key3 = get_primary_keys(word,mode,3)
        if primary_key == "": continue
        if primary_key not in corpus_grouped:
            corpus_grouped[primary_key] = {}
        if primary_key2 not in corpus_grouped[primary_key]:
            corpus_grouped[primary_key][primary_key2] = {}
        if primary_key3 not in corpus_grouped[primary_key][primary_key2]:
            corpus_grouped[primary_key][primary_key2][primary_key3] = []
        corpus_grouped[primary_key][primary_key2][primary_key3].append(word)
        #print(word)
        #if "propn_phrase" in word.values():
        #    corpus_grouped[primary_key][primary_key2][primary_key3].append(word)
    return corpus_grouped



# ------------------__MAIN__---------------------------

corpus_file = None
if len(sys.argv) > 1 and ".json" in sys.argv[1]:
    corpus_file = sys.argv[1]

mode = "new"
test_num = ""
num_heads = 4
freq_threshold = 5

if len(sys.argv) > 2:
    mode = "parse"
    mode = mode + "_word_only"
    mode = mode + "_word_src"
    mode = mode + "_lit_only"
    mode = mode+"_transformer10fqx"

    early_fusion = True
    mode = mode+"_early_fusion"

    #x0 xn10 xn xp11 xp01s xp10d xp10h xp10b xp1a xp2 xp103 xp3 xp104n xp4n xp4ppn xp4nppn xp4prn xp4

    if "x0p" in sys.argv[2].lower():
        mode = mode+"_x0p"
        dep_active = depd_active = deph_active = False
        suf_active = morph_active = pos_active = pos_active2 = False
        pos_active = False
        punct_active = True
        sent2_active = False
    elif "x0s" in sys.argv[2].lower():
        mode = mode+"_x0s"
        dep_active = depd_active = deph_active = False
        suf_active = morph_active = pos_active = pos_active2 = False
        pos_active = False
        sent2_active = True
    elif "x0" in sys.argv[2].lower():
        if "word_only" in mode or "word_src" in mode:
            x_dimension_1 = True
        mode = mode+"_x0"
        dep_active = depd_active = deph_active = False
        suf_active = morph_active = pos_active = pos_active2 = False
        pos_active = False
        sent2_active = False
    elif "x1" in sys.argv[2].lower():
        if "word_only" in mode or "word_src" in mode:
            x_dimension_1 = True
        mode = mode+"_x1"
        pos_active = False
    elif "xo" in sys.argv[2].lower():
        xo_active = True
        mode = mode+"_xo"
        pos_active = False
    elif "xp11" in sys.argv[2].lower():
        mode = mode+"_xp11"
        xp_active = True
        dep_active = False
        morph_active = True
        suf_active = False
        depd_active = False
        deph_active = False
        pos_active = False
        pos_active2 = False
    elif "xp01sdm" in sys.argv[2].lower():
        mode = mode+"_xp01sdm"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = False
        pos_active = False
        pos_active2 = False
        masked_init_active = True
    elif "xp01sd" in sys.argv[2].lower():
        mode = mode+"_xp01sd"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = False
        pos_active = False
        pos_active2 = False
    elif "xp01sm" in sys.argv[2].lower():
        mode = mode+"_xp01sm"
        xp_active = True
        dep_active = False
        morph_active = True
        suf_active = True
        depd_active = False
        deph_active = False
        pos_active = False
        pos_active2 = False
        masked_init_active = True
    elif "xp01s" in sys.argv[2].lower():
        mode = mode+"_xp01s"
        xp_active = True
        dep_active = False
        morph_active = True
        suf_active = True
        depd_active = False
        deph_active = False
        pos_active = False
        pos_active2 = False
    elif "xp10d" in sys.argv[2].lower():
        mode = mode+"_xp10d"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = False
        depd_active = True
        deph_active = False
        pos_active = False
        pos_active2 = False
    elif "xp10h" in sys.argv[2].lower():
        mode = mode+"_xp10h"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = False
        depd_active = False
        deph_active = True
        pos_active = False
        pos_active2 = False
    elif "xp10b" in sys.argv[2].lower():
        mode = mode+"_xp10b"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = False
        depd_active = True
        deph_active = True
        pos_active = False
        pos_active2 = False
    elif "xp1a" in sys.argv[2].lower():
        mode = mode+"_xp1a"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = True
        pos_active = False
        pos_active2 = False
    elif "xp103" in sys.argv[2].lower():
        mode = mode+"_xp103"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = False
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = True
    elif "xp4nppn" in sys.argv[2].lower():
        mode = mode+"_xp4nppn"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = True
        gender_active = True
        genderNPPN_active = True
        coref_active = coref2_active = gender_prop_active = True
    elif "xp4ppn" in sys.argv[2].lower():
        mode = mode+"_xp4ppn"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = True
        gender_active = True
        genderPPN_active = True
        coref_active = coref2_active = gender_prop_active = True
    elif "xp4prn" in sys.argv[2].lower():
        mode = mode+"_xp4prn"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = True
        gender_active = True
        genderPRN_active = True
        coref_active = coref2_active = gender_prop_active = True
    elif "xp4nprn" in sys.argv[2].lower():
        mode = mode+"_xp4nprn"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = True
        gender_active = True
        genderNPRN_active = True
        coref_active = coref2_active = gender_prop_active = True
    elif "xp104n" in sys.argv[2].lower():
        mode = mode+"_xp104n"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = False
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = True
        gender_active = True
        genderN_active = True
        coref_active = coref2_active = gender_prop_active = True
    elif "xp4nsc1d3" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1d3"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = True
        clause2_active = False
        dep2_active = True
        dep3_active = True
        dep2a_active = True
        dep3a_active = False
    elif "xp4nsc1d2" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1d2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = True
        clause2_active = False
        dep2_active = True
        dep3_active = True
        dep2a_active = True
        dep3a_active = True
    elif "xp4nsc1d1" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1d1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = True
        clause2_active = False
        dep2_active = True
        dep3_active = False
    elif "xp4nsc1d" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1d"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = True
        clause2_active = False
        dep2_active = True
        dep3_active = True
    elif "xp4nscd1" in sys.argv[2].lower():
        mode = mode+"_xp4nscd1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = True
        clause2_active = True
        dep2_active = True
        dep3_active = False
    elif "xp4nscd" in sys.argv[2].lower():
        mode = mode+"_xp4nscd"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = True
        clause2_active = True
        dep2_active = True
        dep3_active = True
    elif "xp4ns12c1" in sys.argv[2].lower():
        mode = mode+"_xp4ns12c1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = False
        clause_active = True
        clause2_active = False
    elif "xp4ns12c2" in sys.argv[2].lower():
        mode = mode+"_xp4ns12c2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = False
        clause_active = False
        clause2_active = True
    elif "xp4ns12c" in sys.argv[2].lower():
        mode = mode+"_xp4ns12c"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = False
        clause_active = True
        clause2_active = True
    elif "xp4nsc1h1" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1h"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = True

        dep4_active = True
        dep5_active = True
    elif "xp4nsc1h" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1h"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = True

        dep4_active = True
    elif "xp4nsc1p" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1p"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True

        coref_active = coref2_active = gender_prop_active = True
        punct_active = True
    elif "xp4nsc1ip" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1ip"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True

        coref_active = coref2_active = gender_prop_active = True
        missing_active = True
        punct_active = True
    elif "xp4nsc1igo4" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo4"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        pos_pointers_active = True
        rep2_pointers_active = True
    elif "xp4nsc1igo3" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo3"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        pos_pointers_active = True
    elif "xp4nsc1igo2r4" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2r4"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        coref_active = coref2_active = False
        gender_prop_active = gender_active = genderN_active = False
    elif "xp4nsc1igo2r3" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2r3"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        coref_active = False
        coref2_active = False
        gender_prop_active = False
    elif "xp4nsc1igo2r2" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2r2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        coref2_active = True
        gender_prop_active = True
    elif "xp4nsc1igo2r" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2r"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        coref2_active = False
    elif "xp4nsc1igo2n" in sys.argv[2].lower(): #deprecated
        mode = mode+"_xp4nsc1igo2n"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        gender_prop_active = True
    elif "xp4nsc1igo2d" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2d"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        dep_missing_active = True
    elif "xp4nsc1igo2l2" in sys.argv[2].lower(): #deprecated
        mode = mode+"_xp4nsc1igo2l2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        lemma_unk_active = True
        freq_threshold = 2
    elif "xp4nsc1igo2l" in sys.argv[2].lower(): #deprecated
        mode = mode+"_xp4nsc1igo2l"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        lemma_unk_active = False
    elif "xp4nsc1igo2sb7b" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2sb7b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis4_active = True
        radial_basis_key = 4
    elif "xp4sc1igo2sb7b" in sys.argv[2].lower():
        mode = mode+"_xp4sc1igo2sb7b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis4_active = True
        radial_basis_key = 4
    elif "xp4nsc1igo2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4nsc1igo2b6" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b6"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis3_active = True
        radial_basis_all = True
    elif "xp4nsc1igo2b5d" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b5d"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis3_active = True
        radial_basis_key = 12
    elif "xp4nsc1igo2b5c" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b5c"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis3_active = True
        radial_basis_key = 9
    elif "xp4nsc1igo2b5b" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4nsc1igo2b5" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b5"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis3_active = True
    elif "xp4nsc1igo2b4" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b4"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis2_active = True
        radial_basis_all = True
    elif "xp4nsc1igo2b2" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis2_active = True
    elif "xp4nsc1igo2b" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        radial_basis_active = True
    elif "xp4s2c1igo2sb7b" in sys.argv[2].lower():
        mode = mode+"_xp4s2c1igo2sb7b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis4_active = True
        radial_basis_key = 4
    elif "xp4s2c1igo2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4s2c1igo2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4s2cigo2s" in sys.argv[2].lower():
        mode = mode+"_xp4s2cigo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = clause_active = clause2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s2c2igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4s2c2igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = clause2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s2c1igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4s2c1igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s2igo2sb7b" in sys.argv[2].lower():
        mode = mode+"_xp4s2igo2sb7b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis4_active = True
        radial_basis_key = 4
    elif "xp4s2go2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4s2go2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4s2igo2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4s2igo2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4s2igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4s2igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s2igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4s2igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s2igo2" in sys.argv[2].lower():
        mode = mode+"_xp4s2igo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
    elif "xp4s2ig" in sys.argv[2].lower():
        mode = mode+"_xp4s2ig"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        missing_active = segment_lemma_active  = True
    elif "xp4s2i" in sys.argv[2].lower():
        mode = mode+"_xp4s2i"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
        missing_active  = True
    elif "xp4s2" in sys.argv[2].lower():
        mode = mode+"_xp4s2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent2_active = True
    elif "xp4ns2go2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4ns2go2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4ns2igo2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4ns2igo2sg2b" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2sg2b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionG2B_active = True
    elif "xp4ns2igo2sg2" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2sg2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionG2_active = True
    elif "xp4ns2igo2sg" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2sg"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionG_active = True
    elif "xp4ns2igo2sc" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2sc"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionC_active = True
    elif "xp4ns2igo2sw" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2sw"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionW_active = True
    elif "xp4nigo2sg2b" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sg2b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionG2B_active = True
    elif "xp4nigo2sg2" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sg2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionG2_active = True
    elif "xp4nigo2sg" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sg"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionG_active = True
    elif "xp4nigo2sc" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sc"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionC_active = True
    elif "xp4nigo2sw" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sw"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        early_fusionW_active = True
    elif "xp4ns2igo2sk4b" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2sk4b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
        cosine1_similarity_active = True
        radial_basis_key = 4
    elif "xp4ns2igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4ns2igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4rs2igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4rs2igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        coref_active = coref2_active = True
        sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4sc1igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4sc1igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s12igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4s12igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s1igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4s1igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4sigo2s" in sys.argv[2].lower():
        mode = mode+"_xp4sigo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

        segment_dep_active = True
    elif "xp4s12igo2" in sys.argv[2].lower():
        mode = mode+"_xp4s12igo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

    elif "xp4s1igo2" in sys.argv[2].lower():
        mode = mode+"_xp4s1igo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

    elif "xp4sigo2" in sys.argv[2].lower():
        mode = mode+"_xp4sigo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

    elif "xp4sc1igo2" in sys.argv[2].lower():
        mode = mode+"_xp4sc1igo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True

    elif "xp4nsc1igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True

        segment_dep_active = True
    elif "xp4nsc1igo2" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = True
        sent_active = sent2_active = sent3_active = clause_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True

    elif "xp4nsc1igo" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1igo"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True

        coref_active = coref2_active = gender_prop_active = True
        missing_active = True
        segment_lemma_active = True
        dep_pointers_active = True
    elif "xp4nsc1ig2" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1ig2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = True
        missing_active = True
        segment_lemma_active = True
        segment2_lemma_active = True
    elif "xp4nsc1ig" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1ig"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True

        coref_active = coref2_active = gender_prop_active = True
        missing_active = True
        segment_lemma_active = True
    elif "xp4nsc1i" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1i"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True
        coref_active = coref2_active = gender_prop_active = True
        missing_active = True
    elif "xp4nsc1g" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1g"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True

        coref_active = coref2_active = gender_prop_active = True
        segment_lemma_active = True
    elif "xp4nsc1" in sys.argv[2].lower():
        mode = mode+"_xp4nsc1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        sent_active = sent2_active = sent3_active = clause_active = True

        coref_active = coref2_active = gender_prop_active = True
    elif "xp4nsc2" in sys.argv[2].lower():
        mode = mode+"_xp4nsc2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = False
        clause2_active = True
    elif "xp4nsc" in sys.argv[2].lower():
        mode = mode+"_xp4nsc"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        clause_active = True
        clause2_active = True
    elif "xp4nsd1" in sys.argv[2].lower():
        mode = mode+"_xp4nsd1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        dep2_active = True
        dep3_active = False
    elif "xp4nsd" in sys.argv[2].lower():
        mode = mode+"_xp4nsd"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
        dep2_active = True
        dep3_active = True
    elif "xp4ns12" in sys.argv[2].lower():
        mode = mode+"_xp4ns12"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = False
    elif "xp4ns13" in sys.argv[2].lower():
        mode = mode+"_xp4ns13"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = False
        sent3_active = True
    elif "xp4ns23" in sys.argv[2].lower():
        mode = mode+"_xp4ns23"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = False
        sent2_active = True
        sent3_active = True
    elif "xp4ns1" in sys.argv[2].lower():
        mode = mode+"_xp4ns1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = False
        sent3_active = False
    elif "xp4ns2" in sys.argv[2].lower():
        mode = mode+"_xp4ns2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = False
        sent2_active = True
        sent3_active = False
    elif "xp4ns3" in sys.argv[2].lower():
        mode = mode+"_xp4ns3"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = False
        sent2_active = False
        sent3_active = True
    elif "xp4ns" in sys.argv[2].lower():
        mode = mode+"_xp4ns"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        sent_active = True
        sent2_active = True
        sent3_active = True
    elif "xp4nrw" in sys.argv[2].lower():
        mode = mode+"_xp4nrw"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        coref_active = coref2_active = gender_prop_active = True
        early_fusionW_active = True
    elif "xp4nr" in sys.argv[2].lower():
        mode = mode+"_xp4nr"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        coref_active = coref2_active = gender_prop_active = True
    elif "xp4nw" in sys.argv[2].lower():
        mode = mode+"_xp4nw"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        early_fusionW_active = True
    elif "xp4nc" in sys.argv[2].lower():
        mode = mode+"_xp4nc"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        early_fusionC_active = True
    elif "xp4ng2" in sys.argv[2].lower():
        mode = mode+"_xp4ng2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        early_fusionG_active = True
        early_fusionG2_active = True
    elif "xp4ng1" in sys.argv[2].lower():
        mode = mode+"_xp4ng1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        early_fusionG_active = True
    elif "xp4nz0" in sys.argv[2].lower():
        mode = mode+"_xp4nz0"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        early_fusionZ0_active = True
    elif "xp4nz1" in sys.argv[2].lower():
        mode = mode+"_xp4nz1"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        early_fusionZ1_active = True
    elif "xp4np" in sys.argv[2].lower():
        mode = mode+"_xp4np"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True

        punct_active = True
    elif "xp4o2" in sys.argv[2].lower():
        mode = mode+"_xp4o2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        dep_pointers_active = rep_pointers_active = True
    elif "xp4go2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4go2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4go2s" in sys.argv[2].lower():
        mode = mode+"_xp4go2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
    elif "xp4go2" in sys.argv[2].lower():
        mode = mode+"_xp4go2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        segment_lemma_active = dep_pointers_active = rep_pointers_active = True
    elif "xp4g" in sys.argv[2].lower():
        mode = mode+"_xp4g"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        segment_lemma_active = True
    elif "xp4igo2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4igo2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4igo2s" in sys.argv[2].lower():
        mode = mode+"_xp4igo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
    elif "xp4igo2" in sys.argv[2].lower():
        mode = mode+"_xp4igo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
    elif "xp4ig" in sys.argv[2].lower():
        mode = mode+"_xp4ig"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        missing_active = True
        segment_lemma_active = True
    elif "xp4i" in sys.argv[2].lower():
        mode = mode+"_xp4i"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        missing_active = True
    elif "xp4ngo2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4ngo2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4ngo2s" in sys.argv[2].lower():
        mode = mode+"_xp4ngo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
    elif "xp4nigo2sk5b" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        sigmoid_active = True
        radial_basis_key = 4
    elif "xp4nigo2sk4c" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk4c"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        cosine1_similarity_active = True
        radial_basis_key = 9
    elif "xp4nigo2sk4b" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk4b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        cosine1_similarity_active = True
        radial_basis_key = 4
    elif "xp4nigo2sk4" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk4"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        cosine1_similarity_active = True
        radial_basis_key = 0
    elif "xp4nigo2sk3c" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk3c"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        tanh_active = True
        radial_basis_key = 9
    elif "xp4nigo2sk1c" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk1c"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        cosine_similarity_active = True
        radial_basis_key = 9
    elif "xp4nigo2sk3b" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk3b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        tanh_active = True
        radial_basis_key = 4
    elif "xp4nigo2sk3" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sk3"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        tanh_active = True
        radial_basis_key = 0
    elif "xp4nigo2sb5b" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2sb5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4nigo2s" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
        segment_dep_active = True
    elif "xp4nigo2" in sys.argv[2].lower():
        mode = mode+"_xp4nigo2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = segment_lemma_active = dep_pointers_active = rep_pointers_active = True
    elif "xp4nig" in sys.argv[2].lower():
        mode = mode+"_xp4nig"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = True
        segment_lemma_active = True
    elif "xp4ni" in sys.argv[2].lower():
        mode = mode+"_xp4ni"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        coref_active = coref2_active = True
        gender_prop_active = gender_active = genderN_active = True
        missing_active = True
    elif "xp4n" in sys.argv[2].lower():
        mode = mode+"_xp4n"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2 = gender_active = genderN_active = True
        coref_active = coref2_active = gender_prop_active = True
    elif "xp4w" in sys.argv[2].lower():
        mode = mode+"_xp4w"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True

        early_fusionW_active = True
    elif "xp4c" in sys.argv[2].lower():
        mode = mode+"_xp4c"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True

        early_fusionC_active = True
    elif "xp4s" in sys.argv[2].lower():
        mode = mode+"_xp4s"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
        segment_dep_active = True
    elif "xp4b6" in sys.argv[2].lower():
        mode = mode+"_xp4b6"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
        radial_basis3_active = True
        radial_basis_all = True
    elif "xp4b5b" in sys.argv[2].lower():
        mode = mode+"_xp4b5b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        radial_basis3_active = True
        radial_basis_key = 4
    elif "xp4b5" in sys.argv[2].lower():
        mode = mode+"_xp4b5"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
        radial_basis3_active = True
    elif "xp4b4" in sys.argv[2].lower():
        mode = mode+"_xp4b4"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
        radial_basis2_active = True
        radial_basis_all = True
    elif "xp4b3" in sys.argv[2].lower():
        mode = mode+"_xp4b3"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
        radial_basis_active = True
        radial_basis_all = True
    elif "xp4b2" in sys.argv[2].lower():
        mode = mode+"_xp4b2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
        radial_basis2_active = True
    elif "xp4b" in sys.argv[2].lower():
        mode = mode+"_xp4b"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
        radial_basis_active = True
    elif "xp40r2" in sys.argv[2].lower():
        mode = mode+"_xp40r2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        ragF_active = True
        ragG_active = True
    elif "xp40r" in sys.argv[2].lower():
        mode = mode+"_xp40r"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        ragF_active = True
    elif "xp40s2" in sys.argv[2].lower():
        mode = mode+"_xp40s2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        tanh_active = True
    elif "xp4" in sys.argv[2].lower():
        mode = mode+"_xp4"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = pos_active2  = True
        #gender_active = True
        #coref_active = coref2_active = gender_prop_active = True
    elif "xp2n" in sys.argv[2].lower():
        mode = mode+"_xp2n"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active = True
        gender_active = genderN_active = True
        #coref_active = coref2_active = gender_prop_active = True

    elif "xp2g2" in sys.argv[2].lower():
        mode = mode+"_xp2g2"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active  = True

        early_fusionG_active = True
        early_fusionG2_active = True
    elif "xp2g" in sys.argv[2].lower():
        mode = mode+"_xp2g"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active  = True

        early_fusionG_active = True
    elif "xp2w" in sys.argv[2].lower():
        mode = mode+"_xp2w"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active  = True

        early_fusionW_active = True
    elif "xp2c" in sys.argv[2].lower():
        mode = mode+"_xp2c"
        xp_active = morph_active = suf_active = dep_active = depd_active = deph_active = pos_active  = True

        early_fusionC_active = True
    elif "xp2" in sys.argv[2].lower():
        mode = mode+"_xp2"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = False
    elif "xp3" in sys.argv[2].lower():
        mode = mode+"_xp3"
        xp_active = True
        dep_active = True
        morph_active = True
        suf_active = True
        depd_active = True
        deph_active = True
        pos_active = True
        pos_active2 = True
    elif "xq" in sys.argv[2].lower():
        xq_active = True
        xp_active = True
        mode = mode+"_xq"
    elif "xn10sm" in sys.argv[2].lower():
        mode = mode+"_xn10sm"
        xp_active = dep_active = depd_active = deph_active = False
        suf_active = morph_active = pos_active = pos_active2 = False
        suf_active = True
        morph_active = True
    elif "xn10s" in sys.argv[2].lower():
        mode = mode+"_xn10s"
        xp_active = dep_active = depd_active = deph_active = False
        suf_active = morph_active = pos_active = pos_active2 = False
        suf_active = True
    elif "xnm" in sys.argv[2].lower():
        mode = mode+"_xnm"
        pos_active = False
        masked_init_active = True
    elif "xn" in sys.argv[2].lower():
        mode = mode+"_xn"
        xp_active = dep_active = depd_active = deph_active = False  #xp includes lemma and additional suffix
        suf_active = morph_active = pos_active = pos_active2 = False

    if "y1" in sys.argv[2].lower():
        y_dimension_1 = True
        mode = mode+"_y1"
    else:
        mode = mode+"_yn"

    if early_fusionC_active:
        mode = mode.replace("early_fusion","early_fusionC")
    elif early_fusionG2B_active:
        mode = mode.replace("early_fusion","early_fusionG2B")
    elif early_fusionG2_active:
        mode = mode.replace("early_fusion","early_fusionG2")
    elif early_fusionG_active:
        mode = mode.replace("early_fusion","early_fusionG")
    elif early_fusionW_active:
        mode = mode.replace("early_fusion","early_fusionW")

    if depd_active or deph_active:
        hypoth_active = True

    io_modes = ["_irop_", "_irob_", "_irod_", "_irosf_", "_irpob_", "_irpod_", "_irposf_", "_irbod_", "_irbosf_", "_irpbod_", "_irpbosf_", "_irdosf_", "_irpbdosf_", "_irbdosf_", "_irpdosf_"]

    input_mode = "parser"
    output_mode = "surface"
    mode = mode +"_irosf"
    mode = mode + "_p0"
    mode = mode + "_id0"

    if "loadmodel" in sys.argv[2].lower() and ("tuning" in sys.argv[2].lower() or "testset" in sys.argv[2].lower()):
        mode = mode + "_loadmodel"
    elif "loadword" in sys.argv[2].lower() and not("tuning" in sys.argv[2].lower() or "testset" in sys.argv[2].lower()):
        mode = mode + "_loadword"
    else:
        mode = mode + "_noload"

    if "savemodel" in sys.argv[2].lower() and "testset" not in sys.argv[2].lower():
        mode = mode + "_savemodel"
    else:
        mode = mode + "_nosave"

    if "tuning" in sys.argv[2].lower():
        mode = mode + "_tuning"
    elif "testset1" in sys.argv[2].lower():
        mode = mode + "_testset1"
        seed = int(time.time())
    elif "testset" in sys.argv[2].lower():
        mode = mode + "_testset"
    elif "pretrain1" in sys.argv[2].lower():
        mode = mode + "_pretrain1"
        seed = int(time.time())
    elif "pretrain" in sys.argv[2].lower():
        mode = mode + "_pretrain"

    if "loadword" in sys.argv[2].lower() and  "negsmpl" in sys.argv[2].lower():
        print("loadword and negsmpl: exiting")
        exit()
    if "w2v" in sys.argv[2].lower() and  "negsmpl" in sys.argv[2].lower():
        print("loadword and negsmpl: exiting")
        exit()

    fields1 = sys.argv[1].lower().split(".")
    fields2 = fields1[0].split("vectors_")
    test_num = fields2[-1]

hidden_dimension = 128
def_dimension = 128
cluster_file = ""
clusters = {}
previous_predictions = []
previous_predictions_file_out = ""
epoch_label = 1

if len(sys.argv) > 4:
    dim1_dim2 = sys.argv[-6]
    if "-" in dim1_dim2:
        fields = dim1_dim2.split("-")
        hidden_dimension = int(fields[0])
        num_heads = int(fields[1])
    else:
        hidden_dimension = int(sys.argv[-6])


X_dimension = 70  #70 + 19 + some padding with artificial deps and renorm deps; now 50 + 19 pad
y_dimension = 70

if not xp_active:
    X_dimension = 8
    y_dimension = 8

if "fx" in mode or y_dimension_1:
    y_dimension = 1
if x_dimension_1 and "propnreplcnd" not in mode:
    if "sqs" in mode or "sqx" in mode or "sqf" in mode or "fqx" in mode or "mqx" in mode:
        X_dimension = 2
    else:
        X_dimension = 1

neg_multiplier = 5

batch_size = 100
if "testset" in mode:
    batch_size = 100

if early_fusionG_active or early_fusionGA_active:
    X_dimension = 60  #70 + 19 + some padding with artificial deps and renorm deps; now 50 + 19 pad
    y_dimension = 60
    batch_size = 60

print(X_dimension,y_dimension,y_dimension_1,"dimensions")

mode = mode+"_"+str(freq_threshold)+"_"+str(dim1_dim2)+"_"+test_num+"_e"+str(epoch_label)

pos_map = {"ADJ":"adjective","ADV":"adverb","NOUN":"noun","VERB":"verb","PROPN":"proper_noun","any":"any","DET":"adjective"}
pos_map_rev = {"adjective":"ADJ","adverb":"ADV","noun":"NOUN","verb":"VERB","proper_noun":"PROPN","any":"any"}
skip_pos_list = ["AUX","ADP","PRON","UNK","","SCONJ","CCONJ","INTJ", "NUM", "PART", "X", "PUNCT", "SYM"]

propn_file = "NONE"
propn_measurements = None

#folding_options = {"u":"punctuation","j":"adjective","d","adverb","t":"determiner","e":"proper noun","n":"noun","x":"auxillary","v":"verb","p":"preposition","r":"object of preposition","o":"direct object","i":"indirect object","s":"subject"}
#folding ex. _jdt_  _jdtpx_  _dtpxe_  _ujdtpxenv_   _ujdtpxenvior_   _ujdtpxens_
if "_" in sys.argv[-5]:
    folding0 = sys.argv[-5]
    if "-" in folding0:
        folding = folding0.split("-")[0]
    if "0" in folding or "1" in folding or "2" in folding or "3" in folding or "4" in folding or "5" in folding or "6" in folding or "7" in folding or "8" in folding or "9" in folding:
        rag_active = True
        if "4" in folding or "5" in folding or "6" in folding or "7" in folding or "8" in folding or "9" in folding:
            ragW_active = True      #ragU and ragV do not include the whole word vector, only the rag_key token
            #ragV_active = True    #ragV_active creates a separate set of rag_ or rag_pos tokens
        if "8" in folding or "4" in folding:
            ragF_active = True
        if "9" in folding:
            ragF_active = True
            ragG_active = True
    folding_full = folding
    rag_folding_full = folding_full
    if "F" in folding_full:
        folding_active = True
        folding_rag_active = True
    if "G" in folding_full:
        folding_active = True
        folding_rag_active = True
        folding_echo_active = True
        folding_rag_echo_active = True
        folding_once_active = True
        folding_rag_once_active = True
    if "H" in folding_full:
        folding_active = True
        folding_rag_active = True
        folding_echo_active = True
        folding_rag_echo_active = True
    if "I" in folding_full:
        folding_active = True
        folding_echo_active = True
    if "J" in folding_full:
        folding_rag_active = True
        folding_rag_echo_active = True
    if "K" in folding_full:
        folding_active = True
        folding_bidirectional_active = True
    if "L" in folding_full:
        folding_rag_active = True
        folding_rag_bidirectional_active = True
    if "M" in folding_full:
        folding_active = True
        folding_once_active = True
    if "N" in folding_full:
        folding_rag_active = True
        folding_rag_once_active = True
    if "O" in folding_full:
        folding_active = True
        folding_echo_active = True
        folding_bidirectional_active = True
        folding_once_active = True
        folding_rag_active = True
        folding_rag_echo_active = True
        folding_rag_bidirectional_active = True
        folding_rag_once_active = True
    if "P" in folding_full:
        folding_active = True
        folding_echo_active = True
        folding_once_active = True
        folding_rag_active = True
        folding_rag_echo_active = True
        folding_rag_bidirectional_active = True
        folding_rag_once_active = True
        rag_folding_full = all_folding_full
    if "R" in folding_full:
        ragR_active = True
    if "^" in folding_full:
        folding = folding_full.split("^")[0]
    if "@" in folding_full:
        pred_output_active = True
    if "$" in folding_full:
        pred_input_active = True
else:
    folding0 = folding = folding_full = "_w_"

training_indexes = []
train_str = measure_str = ""
window_size = 2
rag_window_size = 0
skip_size = 1
window_setting = "16"
total_window_size = 0

measurement_indexes = []

if len(sys.argv) > 4:
    if "-" in sys.argv[-2]:
        window_setting = sys.argv[-2]
        fields = window_setting.split("-")
        if fields[0][0] == "i":
            fields[0] = fields[0][1:]
            inattention_all_active = True
        elif fields[0][0] == "j":
            fields[0] = fields[0][1:]
            inattention_all_active = True
            inattention_cls_active = True
        elif fields[0][0] == "k":
            fields[0] = fields[0][1:]
            inattention_all_active = True
            inattention_cls1_active = True
        elif fields[0][0] == "l":
            fields[0] = fields[0][1:]
            inattention_all_active = True
            inattention_cls1_active = True
            inattention_cls3_active = True
        elif fields[0][0] == "c":
            fields[0] = fields[0][1:]
            inattention_cls_active = True
        elif fields[0][0] == "d":
            fields[0] = fields[0][1:]
            inattention_cls1_active = True
        elif fields[0][0] == "e":
            fields[0] = fields[0][1:]
            inattention_cls2_active = True
        elif fields[0][0] == "s":
            fields[0] = fields[0][1:]
        elif fields[0][0] == "t":
            fields[0] = fields[0][1:]
            inattention_cls_active = True
        elif fields[0][0] == "u":
            fields[0] = fields[0][1:]
            inattention_cls1_active = True
        elif fields[0][0] == "v":
            fields[0] = fields[0][1:]
            inattention_cls1_active = True
            inattention_cls3_active = True
        elif fields[0][0] == "w":
            fields[0] = fields[0][1:]
            inattention_cls1_active = True
            inattention_ffn_active = True
        if len(fields) == 3:
            total_window_size = int(fields[0])
            rag_window_size = total_window_size - int(fields[1])
            window_size = int(fields[1])
            prune_size = int(fields[2])
        elif len(fields) == 2:
            total_window_size = int(fields[0])
            rag_window_size = 0
            window_size = int(fields[0])
            prune_size = int(fields[1])
    else:
        rag_window_size = 0
        window_setting = sys.argv[-2]
        window_size= int(sys.argv[-2])
        prune_size = window_size
        total_window_size = window_size
    if prune_size != total_window_size:
        inattention_active = True
    max_position_embeddings = total_window_size
    word_max = (max_position_embeddings * 2)
    phrase_max = max_position_embeddings

    rag_position_embeddings = rag_window_size
    print(max_position_embeddings,total_window_size,window_size,rag_window_size,prune_size,"max_pos total_win win_size rag_win prune_size settings sizes")

window_sizeG = window_size
rag_window_sizeG = rag_window_size
prune_sizeG = prune_size

epoch_override = 1
if len(sys.argv) > 4:
    epoch_override = int(sys.argv[-1])

if len(sys.argv) > 4:
    model_file_in = sys.argv[3]
    model_file_out = sys.argv[4]
    print(model_file_in,"model_file_in")
    print(model_file_out,"model_file_out")

mode = mode + train_str + measure_str + "_"+window_setting+folding0+str(epoch_override)

if len(sys.argv) > 2:
    options_T = ["zero_shotcw","zero_shotcf","zero_shotc", \
    "zero_shotmw","zero_shotmf","zero_shotm", \
    "zero_shotw","zero_shotf","zero_shot", \
    "add_missing","tune_only", \
    "reg_test1cf","reg_test1c", \
    "reg_test1mf","reg_test1m", \
    "reg_test1f","reg_test1", \
    "reg_test0w","reg_test0cw","reg_test0mw","reg_test0f","reg_test0cf","reg_test0mf","reg_test0c","reg_test0m","reg_test0", \
    "reg_testf","reg_test"]

    matchT = False
    option_T = ""
    for o in options_T:
        if o in sys.argv[2].lower():
            matchT = True
            option_T = o
            break
    if matchT:
        mode = mode + "_" + option_T
    #else:
    #    mode = mode + "_reg_test"

    if "reg_test0" in sys.argv[2].lower() or "zero_shot" in sys.argv[2].lower():
        zero_shot_forced = True

    if not matchT and "pretrain" not in mode:
        options_T = ["tune_only", \
        "fine_tune1cf","fine_tune1c", \
        "fine_tune1mf","fine_tune1m", \
        "fine_tune1f","fine_tune1", \
        "fine_tune0w","fine_tune0cw","fine_tune0mw","fine_tune0f","fine_tune0cf","fine_tune0mf","fine_tune0c","fine_tune0m","fine_tune0", \
        "fine_tunef","fine_tune"]

        matchT = False
        option_T = ""
        for o in options_T:
            if o in sys.argv[2].lower():
                matchT = True
                option_T = o
                break
        if matchT:
            mode = mode + "_" + option_T
        #else:
        #    mode = mode + "_fine_tune"

        if "fine_tune0" in sys.argv[2].lower():
            zero_shot_forced = True

if len(sys.argv) > 2:
    if "_w2v" in sys.argv[2].lower():
        mode = mode + "_w2v"
    else:
        mode = mode + "_t2v"
    if "_xav" in sys.argv[2].lower():
        mode = mode + "_xav"
    else:
        mode = mode + "_uniform"

fo = open(mode+"_ms.txt","w")
fo.close()
print(mode+"_ms.txt", "<------------------ms.txt")
fo = open(mode+"_ms.log","w")

str1 = "mode "+str(mode)
print(str1)
fo.write(str1+"\n")

if "_loadmodel" in mode and "zero_shot" not in mode:
    #check that the file exists early
    in_pickle_model = open(model_file_in, "rb")
    in_pickle_model.close()

pages = []
feature_strings = {}
text_list= []
token_map, short_map = get_token_map()

learning_rate = 0.003
ppn_set = {}
word_to_index,index_to_word,corpus,vocab_size,length_of_corpus,feature_strings,ppn_set,mod_freq,output_strings,corpus_effective_len = generate_dictionary_data2(pages, token_map, short_map, corpus_file, ppn_set, mode, X_dimension, max_position_embeddings,False,False)

corpusR = {}

del token_map
del short_map
print(len(word_to_index),"len word_to_index post gen data2")

corpus_grouped = get_corpus_grouped(corpus,mode)
str1 = "vocab_size "+str(vocab_size)+" len_word_to_index "+str(len(word_to_index))+" length_of_corpus "+str(length_of_corpus)+" len_corpus_grouped "+str(len(corpus_grouped))
fo.write(str1+"\n")

str1 = "len feature_strings(non-word/def/prpn) "+str(len(feature_strings))
fo.write(str1+"\n")


words_freq = {}
for c in corpus_grouped:
    for c2 in corpus_grouped[c]:
        count = 0
        #if "-" in c2 or "-" in c: count = 10
        for c3 in corpus_grouped[c][c2]:
            #print(c,c2,c3,count)
            count += len(corpus_grouped[c][c2][c3])
        words_freq[c2] = count


del corpus_grouped
corpus_grouped = None

num1 = numN = num_multi = 0
num1_corpus = {}
if "tuning" not in mode and "testset" not in mode:
    for c2 in words_freq:
        multiword = 0
        if "|" in c2:
            c3 = c2.split("|")
            for c4 in c3:
                if "PPN_" not in c4: multiword+=1
        if words_freq[c2]<=freq_threshold and multiword < 2:
            num1+=1
            num1_corpus[c2] = 1
        else:
            numN+=1
            if "|" in c2 and multiword >= 2:
                num_multi+=1

targeted_tests = {"king":None,"queen":None,"man":None,"woman":None,"apple":None,"apples":None,"car":None,"cars":None,"son":None,"sons":None,"company":None,"companies":None,"story":None,"stories":None}

corpus2 = []
vocab = {}
for i in range(len(corpus)):
    word = corpus[i]
    a,p_key2,c = get_primary_keys(word,mode,3)
    if a == "" and i < len(corpus)-1 and corpus[i+1]["STR"] != "ascii": continue
    if (p_key2 not in num1_corpus or p_key2.replace("_STR","") in targeted_tests or (i < len(corpus)-1 and corpus[i+1]["STR"] == "ascii")):
        corpus2.append(word)
        vocab[p_key2] = 1
    else:
        word["STR"] = "UNK"
        keys_list = list(word.keys())
        for w in keys_list:
            if w in word and word[w] == "word":
                del word[w]
                if "_STR" in w:
                    word["UNK_STR"] = "word"
                else:
                    word["UNK"] = "word"
            if w in word and word[w] == "base":
                del word[w]
                word["UNK"] = "base"
            if lemma_unk_active and ("lemma_" in w or "fold_pos" in w) and "_" in w:
                del word[w]
                w1 = w.split("_")
                word[w1[0]+"_UNK"] = "metadata"
        a,b,c = get_primary_keys(word,mode,3,True)
        corpus2.append(word)
        pass

del corpus
corpus = None

str1 = "vocab_size1 "+str(len(vocab))+" len_word_to_index "+str(len(word_to_index))+" num_multi "+str(num_multi) #+"__"+str(list(vocab)[:50])
fo.write(str1+"\n")
del vocab
vocab = None

label_to_index_saved = {}
label_to_index = {}
definition_picks_saved = None
freqs = {}
freqs_h = {}

if ("_loadmodel" in mode or "_loadword" in mode):
    model_file_in_i = model_file_in
    if "testset" in mode:
        model_fields = model_file_in.split("#")
        model_file_in_i = model_fields[0]+"#4#"+model_fields[2]
    in_pickle_model = open(model_file_in_i, "rb")
    word_to_index = pickle.load(in_pickle_model)
    output_strings = pickle.load(in_pickle_model)
    label_to_index_saved = pickle.load(in_pickle_model)
    definition_picks_saved = pickle.load(in_pickle_model)
    freqs = pickle.load(in_pickle_model)
    freqs_h = pickle.load(in_pickle_model)
    words_freq2_override = pickle.load(in_pickle_model)
    mod_freq = pickle.load(in_pickle_model)
    in_pickle_model.close()

print(max_position_embeddings,"max_position_embeddings")
word_to_index,index_to_word,corpus,vocab_size,length_of_corpus,feature_strings,ppn_set,freqs,freqs_h,mod_freq,output_strings,corpus_effective_len, blanket_drop = get_corpus_metadata(corpus_file, corpus2, ppn_set, mode, X_dimension, max_position_embeddings, False, False, False, freqs, freqs_h)

rag_blanket = {}

str1 = "vocab_size2 "+str(vocab_size)+" len_word_to_index "+str(len(word_to_index))+" length_of_corpus2 "+str(length_of_corpus)+" ppn_len "+str(len(ppn_set))
print(str1)
fo.write(str1+"\n")

str2 = ",".join(sorted(list(ppn_set.keys())))
fo.write(str2+"\n")

foa = open("prop_noun_set.txt","a")
foa.write(str2+"\n")
foa.close()

if propn_measurements is not None:
    del ppn_set
    ppn_set = propn_measurements

if "negsmpl" not in mode:
    del ppn_set
    ppn_set = {}

del corpus2
corpus2 = []
corpus_grouped = get_corpus_grouped(corpus,mode)

words_freq = {}
for c in corpus_grouped:
    for c2 in corpus_grouped[c]:
        count = 0
        for c3 in corpus_grouped[c][c2]:
            count += len(corpus_grouped[c][c2][c3])
        words_freq[c2] = count

measurements_objects = []
measurements_freqs = []
definition_maps = []
separators1 = ["noun.","verb.","adjective.","adverb.","preposition."]
separators2 = ["idiom."]
idioms = {}
word_2_words = {}
prune = False

words_subset1 = [y[0] for y in sorted(words_freq.items(), key=lambda x: x[1], reverse=True)[:numN-10]]
words_subset2 = []
for w in words_subset1:
    if "propn_" not in w and "PPN_" not in w and "DEF_" not in w and "UNK" not in w and "Suffix" not in w:
        words_subset2.append(w)
del words_subset1
words_subset1 = None
#print(words_subset)
str1 = "top_n_words2 "+str(len(words_subset2))
#print(str1)
fo.write(str1+"\n")
words_subset = words_subset2[:int(len(words_subset2)/10)+10]
del words_subset2
words_subset2 = None
top_n_words = len(words_subset)
#print(words_subset[:52])
str1 = "top_n_words "+str(top_n_words)+" len_word_to_index "+str(len(word_to_index))
#print(str1)
fo.write(str1+"\n")

#print(len(words_subset),num1,numN,length_of_corpus)

label_to_index_saved = {}
label_to_index = {}
definition_picks_saved = None
clusters_loaded = False
words_freq2_override = None

if ("_loadmodel" in mode or "_loadword" in mode):
    freqs = {}
    freqs_h = {}
    mod_freq = {}
    clusters_loaded = True
    model_file_in_i = model_file_in
    if "testset" in mode:
        model_fields = model_file_in.split("#")
        model_file_in_i = model_fields[0]+"#4#"+model_fields[2]
    in_pickle_model = open(model_file_in_i, "rb")
    word_to_index = pickle.load(in_pickle_model)
    output_strings = pickle.load(in_pickle_model)
    label_to_index_saved = pickle.load(in_pickle_model)
    definition_picks_saved = pickle.load(in_pickle_model)
    freqs = pickle.load(in_pickle_model)
    freqs_h = pickle.load(in_pickle_model)
    words_freq2_override = pickle.load(in_pickle_model)
    mod_freq = pickle.load(in_pickle_model)
    model = pickle.load(in_pickle_model)
    model_tune = pickle.load(in_pickle_model)
    in_pickle_model.close()

bases = {}
for word in corpus:
    for w in word:
        if word[w] == "base":
            bases[w] = 1

torch.manual_seed(seed)
np.random.seed(seed)
gc.collect()
dataset = None
loader = None

training_data2 = None
target_label_unions = None
test_ids = []
valid_indexes = {}
prediction_map = {}

if "negsmpl" in mode or "transformer" in mode:

    training_data, out_dim, label_to_index, target_label_unions, prediction_map, output_strings, rag_blanket2, word_total, words_freq2 = generate_training_data3(corpus_file, corpus, word_to_index, label_to_index, words_freq, words_freq2_override, mod_freq, window_size,skip_size, X_dimension, y_dimension, mode, valid_indexes, max_position_embeddings, rag_position_embeddings, previous_predictions, output_strings, rag_blanket, batch_size, corpus_effective_len, loader, False)

    training_dataR = {}
    if rag_active:
        print(max_position_embeddings,"max_position_embeddings",rag_position_embeddings,"rag_position_embeddings")
        training_dataR, out_dimR, label_to_indexR, target_label_unionsR, prediction_map, output_strings, rag_blanket2, word_total2, words_freq2 = generate_training_data3(rag_file, corpusR, word_to_index, label_to_index, words_freq, words_freq2_override, mod_freq, window_size,skip_size, X_dimension, y_dimension, mode, valid_indexes, max_position_embeddings, rag_position_embeddings, previous_predictions, output_strings, rag_blanket, batch_size, corpus_effective_len, loader, True)
        del label_to_indexR
        del target_label_unionsR

    if "transformer" in mode and "testset" in mode:
        test_ids = training_data[4]

else:
    training_data, out_dim, label_to_index = generate_training_data2("start_", corpus, window_size, vocab_size, word_to_index, label_to_index, words_freq, mode)

torch.manual_seed(seed)
np.random.seed(seed)
gc.collect()


prune = True
definitions = []
if "word_only" in mode:
    if "_def" not in mode and "negsmpl" not in mode:
        training_indexes = [99]
    else:
        training_indexes = [0]

definition_positives = {}

if len(label_to_index_saved) > 0:
    label_to_index = label_to_index_saved
del label_to_index_saved
del definition_picks_saved

test_ids_uniq = {}
for t in test_ids:
    test_ids_uniq[t] = 1

vocab_size = len(word_to_index)
str1 = "training data prepared vocab_size "+str(vocab_size)
fo.write(str1+"\n")
fo.close()

del corpus2
del corpus
if "negsmpl" not in mode:
    del feature_strings
    feature_strings = {}

valid_indexes = {}
for w in word_to_index:
    if is_target_feature(w,output_strings):
        valid_indexes[word_to_index[w]] = 1
print("num valid_indexes",len(valid_indexes),"num output strings",len(output_strings))

max_valid = 0
for v in valid_indexes:
    if v > max_valid: max_valid = v
print("max valid index",max_valid)

config.hidden_size = hidden_dimension
config.hidden_act = X_dimension
config.vocab_size = vocab_size
config.max_position_embeddings = max_position_embeddings # upto 6 random additions + CLS + MASK
config.num_attention_heads = num_heads
config.intermediate_size = hidden_dimension * 4
config.hidden_dropout_prob = 0.1
config.num_hidden_layers = 3
config.num_labels = max_valid+1
config.model_type = "none"
if "xav" in mode:
    config.model_type = "xav"

gc.collect()

if "loadmodel" in mode and "testset" in mode:
    test(training_dataR,training_data,model_file_in,config,word_to_index,test_ids,target_label_unions,mode,valid_indexes,word_2_words, words_freq, previous_predictions, previous_predictions_file_out,rag_position_embeddings, prediction_map, output_strings, rag_blanket2, batch_size, word_total, max_position_embeddings, X_dimension, loader)

else:
    epochs = epoch_override

    epoch_loss,epoch_loss2,weights_1 = train(training_dataR,hidden_dimension,window_size,epochs,training_data,training_data2,out_dim,learning_rate,vocab_size, word_to_index, label_to_index, target_label_unions, measurements_objects, measurements_freqs, corpus_grouped, top_n_words, words_subset, feature_strings, pos_map,skip_pos_list, definition_maps, words_freq, words_freq2, training_indexes, measurement_indexes,def_dimension, word_total, max_position_embeddings, X_dimension, y_dimension, rag_position_embeddings, neg_multiplier,definition_positives, ppn_set,targeted_tests,config,freqs,freqs_h,mod_freq,bases,model_file_in, model_file_out, output_strings, rag_blanket2, batch_size, loader, mode)


end_time = time.time()
diff_time = str(int(end_time - start_time))
str1 = diff_time+","+mode
print(str1)
fo = open("performance_results.csv","a")
fo.write(str1+"\n")
fo.close()

#warnings.filterwarnings('error')



