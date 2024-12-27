#!/usr/bin/env python
# coding: utf-8

# # Relationship Extraction
# 
# Code prepared by Jonathan Heras, Universidad de La Rioja

# ### Model
# 
# * Unlike the `XXXForTokenClassification` model, which comes with a standard token classification head on top of some XXX transformer, there is no pre-built model available OOB at HuggingFace for Relation Extraction.
# * So we will build our own `XXXForRelationExtraction` model by composing a pretrained Transformer encoder with a classification head for classifying the relation type.

# ## Environment Setup

import collections
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import re
import shutil
import torch
import torch.nn as nn

from collections import Counter, defaultdict
from datasets import load_dataset, ClassLabel
from sklearn.metrics import (
    ConfusionMatrixDisplay, confusion_matrix, 
    classification_report, accuracy_score, precision_score, recall_score, f1_score
)
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim import AdamW

from transformers import (
    DebertaV2TokenizerFast,DebertaV2Model,DebertaV2Config,DebertaV2PreTrainedModel,PreTrainedModel,
    DataCollatorWithPadding,
    get_scheduler
)
from transformers.modeling_outputs import SequenceClassifierOutput
torch.cuda.set_device(0)


# ## Data Processing
# ### Mount Google Drive

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--iteration", required=True)
ap.add_argument("-m", "--model", required=True)
args = vars(ap.parse_args())

# Models: microsoft/mdeberta-v3-base
model = args["model"]
iteration = args["iteration"]

GS_INPUT_DIR = "datasetv2"
DATA_DIR = "./data"

BASE_MODEL_NAME = model 
MODEL_DIR = os.path.join(DATA_DIR, "{:s}-ct-ebm-sp".format(BASE_MODEL_NAME))


# ### Input Data Format
# 
# Our dataset is pre-partitioned into train, validation and test splits.
# 
# The `relationMentions` tag lists the relations and matched text of the entities, and the corresponding entity types can be found in the `entityMentions` tag.
# 
# The sentence text can be found in `sentText`.
# 

f = open(os.path.join(GS_INPUT_DIR, "test.json"))
rec = json.load(f)
    

print(json.dumps(rec[0], indent=2))


# ## Entity Types
# Here we analyze the distribution of entity tags across the different splits to verify that it is approximately uniform across the splits.

splits = ["train", "valid", "test"]
ent_counts = collections.defaultdict(collections.Counter)
for split in splits:
  with open(os.path.join(GS_INPUT_DIR, "{:s}.json".format(split)), "r") as fin:
    rec = json.load(fin)
    for json_dict in rec:
        for ent_mention in json_dict["entityMentions"]:
            label = ent_mention["label"]
            ent_counts[label][split] += 1
      

ent_df = pd.DataFrame.from_dict(ent_counts, orient="index").sort_values("train", ascending=False)


# ## Relation Types
# 
# We analyze the distribution of relation types across different splits, and we find that while the distribution is uniform across splits, the distribution is very uneven across different relation types.
# 
# Here we discard the relation types that are very heavily represented and those which are very lightly represented by filtering our dataset such that only relations with train split counts between 1000 and 10000 are retained.
# 
# In real life, we might want to structure our relation extraction pipeline into a hierarchy of models to reflect a more uniform split, or do over/undersampling to attempt to make the dataset less imbalanced, or reduce the model into a set of binary classifiers. But here we take the easy way out and discard relations that are outside the range.


rel_counts = collections.defaultdict(collections.Counter)
for split in splits:
  with open(os.path.join(GS_INPUT_DIR, "{:s}.json".format(split)), "r") as fin:
    rec = json.load(fin)
    for json_dict in rec:
        for ent_mention in json_dict["relationMentions"]:
            label = ent_mention["label"].split("/")[-1]
            rel_counts[label][split] += 1


rel_df = pd.DataFrame.from_dict(rel_counts, orient="index").sort_values("train", ascending=False)

# ## Raw Dataset
# 
# We introduce entity marker tags that wrap the entity mentions in the sentence. For example, the sentence:
# 
# ```Aspirina administrada cada 8 horas .```
# 
# is converted to:
# 
# ```<S:CHE> Aspirina </S:CHE> administrada <O:Fre> cada 8 horas </O:Fre> .```
# 
# We read the input data and write out the augmented sentence using the information from `sentText`, `entityMentions` and `relationMentions` and tokenize them (by space). Also, we extract the relation type from `relationMentions`. The temporary JSON files have records with the following keys: `{"tokens", "label"}`
# 
# This is then converted to a raw HuggingFace dataset using the `load_dataset` function.

types = set()

def reformat_json(infile, outfile):
  print("reformating {:s} -> {:s}".format(infile, outfile))
  fout = open(outfile, "w")
  with open(infile, "r") as fin:
    lines = json.load(fin)
    
    
    for rec in lines:
      
      text = rec["sentText"]
      entities = {}
      for entity_mention in rec["entityMentions"]:
        entity_text = entity_mention["text"]
        entity_type = entity_mention["label"][0:3]
        entities[entity_text] = entity_type
      for relation_mention in rec["relationMentions"]:
        label = relation_mention["label"].split("/")[-1]
        # if label not in valid_relations:
        #   continue
        try:
          sub_text = relation_mention["em1Text"]
          sub_type = entities[sub_text]
          obj_text = relation_mention["em2Text"]
          obj_type = entities[obj_text]
          
          # assumption: em1Text == SUBJECT and occurs before em2Text == OBJECT
          sub_start = text.find(sub_text)
          sub_end = sub_start + len(sub_text)
          text_pre = text[:sub_start]
          text_sub = "<S:{:s}> {:s} </S:{:s}>".format(sub_type, sub_text, sub_type)
          types.add(sub_type)  
          obj_start = text.find(obj_text, sub_end)
          obj_end = obj_start + len(obj_text)
          text_mid = text[sub_end:obj_start]
          text_obj = "<O:{:s}> {:s} </O:{:s}>".format(obj_type, obj_text, obj_type)
          types.add(obj_type)
          text_post = text[obj_end:]
          text = text_pre + text_sub + text_mid + text_obj + text_post

          tokens = text.split()
          output = {
              "tokens": tokens,
              "label": label
          }
          
          fout.write(json.dumps(output) + "\n")
        except:
          pass
  fout.close()


os.makedirs(DATA_DIR, exist_ok=True)

for i, split in enumerate(splits):
  reformat_json(os.path.join(GS_INPUT_DIR, "{:s}.json".format(split)),
                os.path.join(DATA_DIR, "{:s}.json".format(split)))

data_files = {split: os.path.join(DATA_DIR, "{:s}.json".format(split)) for split in splits}

dataset = load_dataset("json", data_files=data_files)

# ### Label Distribution
# 
# Even with the filtering, the label distribution is not uniform. However, it is not as bad as it would be if no filtering was done.

dataset.set_format(type="pandas")


# ### Sentence Length Distribution
# 
# We were encountering errors where we were running out of CUDA Memory. This was happening even when I reduced the batch size from 32 to 8. The other reason this could be happening is that some batches might contain very long outlier sentences and this causes the batch to be too big for the GPU memory.
# 
# Solution is to truncate the sentence size to a size that will not impact the accuracy too much. 
# 
# The box plots below shows that we won't lose too many samples if we ignore sentences that are over 100 tokens in length.

dataset.reset_format()


# ## Tokenizer

tokenizer = DebertaV2TokenizerFast.from_pretrained(BASE_MODEL_NAME)
vocab_size_orig = len(tokenizer.vocab)


# ### Add Entity Marker Tokens
# 
# Now that we added these entity marker tokens, we need to tell the tokenizer to treat them as unbreakable tokens, so we add these tokens to our tokenizer.

marker_tokens = []
entity_types = types
for entity_type in entity_types:
  marker_tokens.append("<S:{:s}>".format(entity_type))
  marker_tokens.append("</S:{:s}>".format(entity_type))
  marker_tokens.append("<O:{:s}>".format(entity_type))
  marker_tokens.append("</O:{:s}>".format(entity_type))

tokenizer.add_tokens(marker_tokens)
vocab_size_new = len(tokenizer.vocab)

print("original vocab size:", vocab_size_orig)
print("new vocab size:", vocab_size_new)


# ## Encoded Dataset
# 
# As before, we create an encoded dataset that contains the output of the tokenizer and numeric labels.
# 
# To compute numeric labels, we create a `ClassLabel` object with our relation tags, and use its built-in functions to create lookup tables from label to label id and vice versa.
# 
# We discard sentences that are larger than `MAX_TOKENS` which we have decided should be 100 tokens (these are space delimited tokens prior to subword tokenization).
# 
# In addition, we also compute and store the value of the entity marker tokens with respect to the positions _after subword tokenization_. This is written into a fixed size int vector consisting of 4 elements, and stored under the `span_idxs` key.

valid_relations = sorted(list(rel_counts.keys()))
rel_tags = ClassLabel(names=valid_relations)
label2id = {name: rel_tags.str2int(name) for name in valid_relations}
id2label = {id: rel_tags.int2str(id) for id in range(len(valid_relations))}

def encode_data(examples):
  tokenized_inputs = tokenizer(examples["tokens"],
                               is_split_into_words=True,
                               truncation=True)
  span_idxs = []
  for input_id in tokenized_inputs.input_ids:
    tokens = tokenizer.convert_ids_to_tokens(input_id)
    span_idxs.append([
      [idx for idx, token in enumerate(tokens) if token.startswith("<S:")][0],
      [idx for idx, token in enumerate(tokens) if token.startswith("</S:")][0],
      [idx for idx, token in enumerate(tokens) if token.startswith("<O:")][0],
      [idx for idx, token in enumerate(tokens) if token.startswith("</O:")][0]
    ])
  tokenized_inputs["span_idxs"] = span_idxs
  tokenized_inputs["labels"] = [label2id[label] for label in examples["label"]]
  return tokenized_inputs

encoded = encode_data(dataset["train"][0:5])
encoded.keys()

MAX_LENGTH = 100
encoded_dataset = (dataset
                       .filter(lambda example: len(example["tokens"]) < MAX_LENGTH)
                       .map(encode_data, batched=True, remove_columns=["tokens", "label"]))

rec = encoded_dataset["train"][0:5]
print("rec.labels:", rec["labels"])
print("rec.input_ids:", len(rec["input_ids"]), len(rec["input_ids"][0]))
print("rec.span_idxs:", rec["span_idxs"])


# ## DataLoader
# 
# As we have done before in our NER notebooks (the ones using PyTorch native idioms), we build `DataLoaders` for each of training, validation and test splits by wrapping the corresponding encoded `Dataset`.
# 
# We use a `DataColatorWithPadding` data collator to automatically pad our batch with the longest sentence in the batch.
# 
# Also as before, we have used the `sampler` trick for doing quick development iterations by keeping the data volume down.

BATCH_SIZE = 16

collate_fn = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

train_dl = DataLoader(encoded_dataset["train"], 
                      shuffle=True, 
                      # sampler=SubsetRandomSampler(np.random.randint(0, encoded_nyt_dataset["train"].num_rows, 1000).tolist()),
                      batch_size=BATCH_SIZE, 
                      collate_fn=collate_fn)
valid_dl = DataLoader(encoded_dataset["valid"], 
                      shuffle=False, 
                      # sampler=SubsetRandomSampler(np.random.randint(0, encoded_nyt_dataset["valid"].num_rows, 200).tolist()),
                      batch_size=BATCH_SIZE, 
                      collate_fn=collate_fn)
test_dl = DataLoader(encoded_dataset["test"], 
                     shuffle=False,
                    #  sampler=SubsetRandomSampler(np.random.randint(0, encoded_nyt_dataset["test"].num_rows, 100).tolist()),
                     batch_size=BATCH_SIZE, 
                     collate_fn=collate_fn)


# ## Model
# 
# Since HuggingFace does not provide us with a built-in `XXXForRelationExtraction` model, we need to build our own. Our model is based on the implementation described in [A frustratingly easy approach for Entity and Relation Extraction](https://arxiv.org/abs/2010.12812) (Zhong and Chen 2020). The authors also provide code in the github repository [princeton-nlp/PURE](https://github.com/princeton-nlp/PURE) to support their paper, which we have consulted as well when building our model.
# 
# The idea is to use a PreTrainedModel as the encoder and a small linear head consisting of `torch.nn.Dropout`, `torch.nn.LayerNorm` and `torch.nn.Linear` layers as the classifier head. The encoder should start with weights from the pretrained model and be fine-tuned from that point using our labeled data.
# 
# Since our base model is BERT, we will use a `BertModel` as our encoder and subclass the full model from `BertPreTrainedModel` so we can inherit its weights via the `self.init_weights()` call. When instantiating the model, we will first instantiate the `BertConfig` class with the appropriate `from_pretrained` model, then instantiate our custom `BertForRelationExtraction` with the weights of its superclass also using `from_pretrained`.
# 
# Finally, we will increase the size of the BERT encoder's embedding matrix to accomodate the extra entity marker tokens we added to the tokenizer vocabulary.
# 
# The `forward` method takes a batch of examples from the encoded dataset, and passes the `input_ids`, `token_type_ids` and `attention_mask` into the BERT encoder (initialized with BertForPreTraining weights). Then the maxpool value across tokens within the entity marker spans (including the entity markers) are computed for the subject and object spans, concatenated and passed into the classifier head.
# 
# The classifier head outputs a logits vector the size of the number of classes, and the argmax over the logits is the predicted relation class.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DebertaV2ForRelationExtraction(PreTrainedModel):
  def __init__(self, config, num_labels):
    super(DebertaV2ForRelationExtraction, self).__init__(config)
    self.num_labels = num_labels
    # body
    self.deberta = DebertaV2Model(config)
    # head
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.layer_norm = nn.LayerNorm(config.hidden_size * 2)
    self.linear = nn.Linear(config.hidden_size * 2, self.num_labels)
    self.init_weights()

  def forward(self, input_ids, token_type_ids, attention_mask,
              span_idxs, labels=None):
    outputs = (
        self.deberta(input_ids, token_type_ids=token_type_ids,
                  attention_mask=attention_mask,
                  output_hidden_states=False)
            .last_hidden_state)
            
    sub_maxpool, obj_maxpool = [], []
    for bid in range(outputs.size(0)):
      # span includes entity markers, maxpool across span
      sub_span = torch.max(outputs[bid, span_idxs[bid, 0]:span_idxs[bid, 1]+1, :], 
                           dim=0, keepdim=True).values
      obj_span = torch.max(outputs[bid, span_idxs[bid, 2]:span_idxs[bid, 3]+1, :],
                           dim=0, keepdim=True).values
      sub_maxpool.append(sub_span)
      obj_maxpool.append(obj_span)

    sub_emb = torch.cat(sub_maxpool, dim=0)
    obj_emb = torch.cat(obj_maxpool, dim=0)
    rel_input = torch.cat((sub_emb, obj_emb), dim=-1)

    rel_input = self.layer_norm(rel_input)
    rel_input = self.dropout(rel_input)
    logits = self.linear(rel_input)

    if labels is not None:
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
      return SequenceClassifierOutput(loss, logits)
    else:
      return SequenceClassifierOutput(None, logits)

# ## Training Loop

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 5


config = DebertaV2Config.from_pretrained(BASE_MODEL_NAME)
model = DebertaV2ForRelationExtraction.from_pretrained(BASE_MODEL_NAME, 
                                                  config=config,
                                                  num_labels=len(valid_relations))
model.deberta.resize_token_embeddings(len(tokenizer.vocab))
model = model.to(device)

optimizer = AdamW(model.parameters(),
                  lr=LEARNING_RATE,
                  weight_decay=WEIGHT_DECAY)

num_training_steps = NUM_EPOCHS * len(train_dl)
lr_scheduler = get_scheduler("linear",
                             optimizer=optimizer,
                             num_warmup_steps=0,
                             num_training_steps=num_training_steps)

def compute_accuracy(labels, logits):
  preds_cpu = torch.argmax(logits, dim=-1).cpu().numpy()
  labels_cpu = labels.cpu().numpy()
  return accuracy_score(labels_cpu, preds_cpu)


def do_train(model, train_dl):
  train_loss = 0
  model.train()
  for bid, batch in enumerate(train_dl):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    train_loss += loss.detach().cpu().numpy()
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

  return train_loss


def do_eval(model, eval_dl):
  model.eval()
  eval_loss, eval_score, num_batches = 0, 0, 0
  for bid, batch in enumerate(eval_dl):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      outputs = model(**batch)

    loss = outputs.loss

    eval_loss += loss.detach().cpu().numpy()
    eval_score += compute_accuracy(batch["labels"], outputs.logits)
    num_batches += 1

  eval_score /= num_batches
  return eval_loss, eval_score


def save_checkpoint(model, model_dir, epoch):
  model.save_pretrained(os.path.join(MODEL_DIR, "ckpt-{:d}".format(epoch)))


def save_training_history(history, model_dir, epoch):
  fhist = open(os.path.join(MODEL_DIR, "history.tsv"), "w")
  for epoch, train_loss, eval_loss, eval_score in history:
    fhist.write("{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
        epoch, train_loss, eval_loss, eval_score))
  fhist.close()


# ## Training / Fine-tuning

if os.path.exists(MODEL_DIR):
  shutil.rmtree(MODEL_DIR)
  os.makedirs(MODEL_DIR)

history = []

for epoch in range(NUM_EPOCHS):
  train_loss = do_train(model, train_dl)
  eval_loss, eval_score = do_eval(model, valid_dl)
  history.append((epoch + 1, train_loss, eval_loss, eval_score))
  print("EPOCH {:d}, train loss: {:.3f}, val loss: {:.3f}, val-acc: {:.5f}".format(
      epoch + 1, train_loss, eval_loss, eval_score))
  save_checkpoint(model, MODEL_DIR, epoch + 1)
  save_training_history(history, MODEL_DIR, epoch + 1)

plt.subplot(2, 1, 1)
plt.plot([train_loss for _, train_loss, _, _ in history], label="train")
plt.plot([eval_loss for _, _, eval_loss, _ in history], label="validation")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="best")

plt.subplot(2, 1, 2)
plt.plot([eval_score for _, _, _, eval_score in history], label="validation")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(loc="best")

plt.tight_layout()
_ = plt.show()


# ## Evaluation

ytrue, ypred = [], []
for batch in test_dl:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
      outputs = model(**batch)
      predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
      labels = batch["labels"].cpu().numpy()
      ytrue.extend(labels)
      ypred.extend(predictions)

with open(BASE_MODEL_NAME.replace('/','-')+"_results.txt",mode="a") as f:
    f.write("test accuracy: {:.3f}\n".format(accuracy_score(ytrue, ypred)))
    f.write("test precision: {:.3f}\n".format(precision_score(ytrue, ypred,average="macro")))
    f.write("test recall: {:.3f}\n".format(recall_score(ytrue, ypred,average="macro")))
    f.write("test f1-score: {:.3f}\n".format(f1_score(ytrue, ypred,average="macro")))
    f.write(classification_report(ytrue, ypred, target_names=valid_relations))

def plot_confusion_matrix(ytrue, ypreds, labels,iteration):
  cm = confusion_matrix(ytrue, ypreds, normalize="true")
  fig, ax = plt.subplots(figsize=(12, 12))
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
  disp.plot(cmap="Blues", values_format="0.2f", ax=ax, colorbar=False)
  plt.title("Normalized Confusion Matrix")
  plt.savefig(BASE_MODEL_NAME.replace('/','-')+"-"+str(iteration)+".png")
  _ = plt.show()


plot_confusion_matrix(ytrue, ypred, valid_relations,iteration)
