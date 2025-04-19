#!/usr/bin/env python
# coding: utf-8

# In[1]:


LANGUAGE = 'zul'
EXPERIMENT = 'final_ground_truth_simple_training'
PREFERRED_GPU = 'cuda:6' # 'cuda:0'


# In[2]:


# Prevent WandB from printing summary in cell output
get_ipython().run_line_magic('env', 'WANDB_SILENT=true')


# In[3]:


import os
import sys
import yaml
import yaml
import wandb
import torch
import warnings

import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
from datasets import load_dataset
from sklearn.exceptions import UndefinedMetricWarning
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from getpass import getpass

torch.cuda.empty_cache()
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# In[4]:


# Add src module to system path
src_module_path = os.path.abspath('../../../')
if src_module_path not in sys.path:
    sys.path.append(src_module_path)

# Import functions and classes from custom modules
from src.data.preprocess import (
    align_labels_for_many_records,
    TorchDataset
)

from src.query.query_gpt import add_annotation_examples_for_batch, add_annotation_examples
from src.query.prompts import MAIN_PROMPT_FOR_BATCH, MAIN_PROMPT

from src.utils.utils import (print_classification_report,
                             calculate_micro_f1_for_batches,
                             calculate_macro_f1_for_batches)

from src.models.xlmr_ner import XLMRobertaForNER


# In[5]:


# Specifying path to the necessary files and folders
PATH_TO_SRC = os.path.abspath('../../../')

CONFIG_PATH = os.path.join(PATH_TO_SRC, "settings/config.yml")


# In[6]:


# Reading config file
config = yaml.safe_load(open(CONFIG_PATH))

# Printing out name of the current language
language_name = config['languages_names'][LANGUAGE]
language_name


# In[7]:


wandb.login(key=getpass("Weights and Biases API key:"))


# In[8]:


label_mapping = config['label_mapping']

# Loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

# Initialize model
base_model = XLMRobertaForNER(model_name=config['model_name'],
                              num_labels=len(config['label_mapping'].keys()))

# Choose a GPU to use
default_device = config['gpu_settings']['default_device'] if \
    PREFERRED_GPU=='' else PREFERRED_GPU

# Send model to GPU if cuda is available otherwise use CPU
device = torch.device(default_device if torch.cuda.is_available() else "cpu")
print(device)

base_model.to(device)


# In[9]:


# Download dataset for the specific language 
data = load_dataset(config['dataset'], LANGUAGE)
print("Original dataset:\n", data)


# In[10]:


data = data.map(
    align_labels_for_many_records,
    batched=True,
    fn_kwargs={'tokenizer': tokenizer}
)


# In[11]:


# Settings
max_len = config['languages_max_tokens'][LANGUAGE]
print(f'Maximum token length for language {LANGUAGE} is {max_len}')
padding_val = config['tokenizer_settings']['padding_value']

# Convert the datasets.Dataset to a PyTorch Dataset
dataset_train = TorchDataset(data['train'], max_length=max_len, padding_value=padding_val)
dataset_test = TorchDataset(data['test'], max_length=max_len, padding_value=padding_val)
dataset_val = TorchDataset(data['validation'],max_length=max_len, padding_value=padding_val)


# In[12]:


# Settings
batch_size = config['train_settings']['batch_size']
shuffle = config['train_settings']['shuffle']

# Create PyTorch DataLoaders
dataloader_train = DataLoader(dataset_train,
                             batch_size=batch_size,
                             shuffle=shuffle) # Shuffle only training set
dataloader_val = DataLoader(dataset_val, batch_size=batch_size)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size)


# In[13]:


def train_ner(model, train_loader, val_loader, device, epochs, lr, num_warmup_steps=5):
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "val_micro_f1": [],
        "val_macro_f1": []
    }
    # Convert lr to float
    lr = float(config['train_settings']['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    # Loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)

            logits_reshaped = logits.view(-1, logits.size(-1))
            labels_reshaped = labels.view(-1)

            # Calculate loss
            loss = loss_fn(logits_reshaped, labels_reshaped)

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask=attention_mask)
                logits_reshaped = logits.view(-1, logits.size(-1))
                labels_reshaped = labels.view(-1)

                # Calculate loss
                loss = loss_fn(logits_reshaped, labels_reshaped)

                total_val_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)

                val_predictions.append(predictions.detach())
                val_labels.append(batch['labels'].detach())

        avg_val_loss = total_val_loss / len(val_loader)
        micro_f1 = calculate_micro_f1_for_batches(val_predictions, val_labels, ignore_class=0)
        macro_f1 = calculate_macro_f1_for_batches(val_predictions, val_labels, ignore_class=0)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {micro_f1:.4f}")

        # Update training history
        training_history["train_loss"].append(avg_train_loss)
        training_history["val_loss"].append(avg_val_loss)
        training_history["val_micro_f1"].append(micro_f1)
        training_history["val_macro_f1"].append(macro_f1)

        # WandB logger
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_micro_f1": micro_f1,
            "val_macro_f1": macro_f1
        })

    return model, training_history


# In[14]:


# Prepare WandB for training
wandb.init(
    project=EXPERIMENT,
    name=LANGUAGE,
    config={
        'epochs': config['train_settings']['epochs'],
        'learning_rate': config['train_settings']['lr']
    },
    settings=wandb.Settings(disable_job_creation=True)
)

# Initial model training
model, history = train_ner(
    base_model,
    dataloader_train,
    dataloader_val,
    device,
    epochs=config['train_settings']['epochs'],
    lr=config['train_settings']['lr'])

# Disable WandB logger
wandb.finish()


# In[15]:


# Save model
torch.save(model.state_dict(), f'model_weights/{LANGUAGE}/{EXPERIMENT}.pth')


# In[16]:


# Model evaluation
print_classification_report(config, model, dataloader_test, device, ignore_index=-100, ignore_class=0)


# In[17]:


# Clear memory

# Delete all models as they are no longer utilized.
del model

# Using garbage collector
import gc
gc.collect()

torch.cuda.empty_cache() 


# In[18]:


# try:
#     os.remove(f'{src_module_path}/../../.netrc')
#     print("Logged out of WandB.")
# except:
#     print("Unsuccessful WandB logging out.")

