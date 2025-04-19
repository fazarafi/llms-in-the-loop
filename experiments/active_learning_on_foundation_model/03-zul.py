#!/usr/bin/env python
# coding: utf-8

# In[1]:


LANGUAGE = 'zul'
EXPERIMENT = 'final_gpt_active_learning'
PREFERRED_GPU = 'cuda:6' # 'cuda:0'


# In[2]:


# Prevent WandB from printing summary in cell output
get_ipython().run_line_magic('env', 'WANDB_SILENT=true')


# In[3]:


import os
import sys
import yaml
import copy
import yaml
import json
import wandb
import torch
import datetime
import warnings
import itertools

import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm
from datasets import load_dataset
from sklearn.exceptions import UndefinedMetricWarning
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from openai import APIStatusError, RateLimitError, APIConnectionError
from sklearn.metrics import f1_score
import numpy as np

from openai import OpenAI
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
    TorchDataset,
    balanced_split_for_active_learning
)

from src.query.query_gpt import add_annotation_examples_for_batch, add_annotation_examples
from src.query.prompts import MAIN_PROMPT_FOR_BATCH, MAIN_PROMPT

from src.utils.utils import (predict_sequence_max_uncertainty,
                             batch_indices_to_global_indices,
                             print_classification_report,
                             calculate_micro_f1_for_batches,
                             calculate_macro_f1_for_batches)

from src.models.xlmr_ner import XLMRobertaForNER


# In[5]:


# Init openai client
openai_client = OpenAI(api_key=getpass("OPENAI API key:"))

# Weights and Biases login
wandb.login(key=getpass("Weights and Biases API key:"))


# In[6]:


# Specifying path to the necessary files and folders
PATH_TO_SRC = os.path.abspath('../../../')

CONFIG_PATH = os.path.join(PATH_TO_SRC, "settings/config.yml")

BATCH_ANNOTATION_EXAMPLES_PATH = os.path.join(
    PATH_TO_SRC, 'src/query/ner_examples_all_languages_for_batch.json'
)

SINGLE_ANNOTATION_EXAMPLES_PATH = os.path.join(
    PATH_TO_SRC, 'src/query/ner_examples_all_languages.json'
)


# In[7]:


# Reading config file
config = yaml.safe_load(open(CONFIG_PATH))

# Printing out name of the current language
language_name = config['languages_names'][LANGUAGE]
language_name


# In[8]:


batch_examples = add_annotation_examples_for_batch(BATCH_ANNOTATION_EXAMPLES_PATH, language_name)
single_examples = add_annotation_examples(SINGLE_ANNOTATION_EXAMPLES_PATH, language_name)


# In[9]:


label_mapping = config['label_mapping']

label_to_index = {value: key for key, value in label_mapping.items()}

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


# In[10]:


ask_gpt_params = {
    'language': language_name,
    #'examples': add_annotation_examples_for_batch(ANNOTATION_EXAMPLES_PATH, language_name),
    'openai_client': openai_client,
    #'user_prompt': MAIN_PROMPT_FOR_BATCH,
    'model': 'gpt-4-0125-preview',
    'temperature': config['foundation_model']['temperature']
}


# In[11]:


def ask_gpt_short(user_prompt, language, openai_client, temperature, model,
                  system_prompt=None, max_tokens=1000):

    if system_prompt is None:
        system_prompt = f"You are a named entity labelling expert in {language} language."

    # Save query params
    query_params = {
        'model': model,
        'temperature': temperature,
        'messages': [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        'max_tokens': max_tokens,
    }

    if model == 'gpt-4-1106-preview' or model == 'gpt-4-0125-preview':
        query_params['response_format'] = {"type": "json_object"}

    try:
        # Query the model
        response = openai_client.chat.completions.create(**query_params)
        # Extract model answer
        answer = response.choices[0].message.content
        return answer

    except APIConnectionError as e:
        raise Exception(f"The server could not be reached: {e.__cause__}")
    except RateLimitError as e:
        raise Exception(f"A 429 status code was received: {e}")
    except APIStatusError as e:
        raise Exception(f"Non-200-range status code received: {e.status_code}, {e.response}")


# In[12]:


def update_ner_annotations_with_gpt(dataset, batch_examples, single_examples):
    """Updates a dataset with Named Entity Recognition (NER) annotations 
    using predictions from a GPT model."""
    print("Asking GPT to annotate data...")
    indices_to_keep = []
    predicted_ner_tags = []
    true_ner_tags = []

    original_dataset_len = len(dataset)

    for i in range(0, original_dataset_len, 2):
        if i+1 < original_dataset_len:  # Batch of 2
            print(f'\nSample {i+1} and {i+2}:')

            record_0 = dataset[i]
            record_1 = dataset[i+1]

            # 1) Query model
            try:
                # Extract tokens from current record
                tokens = {'record_0': record_0['tokens'], 'record_1': record_1['tokens']}

                batch_user_prompt = MAIN_PROMPT_FOR_BATCH.format(
                    language=language_name,
                    inputs=str(tokens),
                    examples=batch_examples
                )

                # Query the model
                new_labels_gpt = ask_gpt_short(user_prompt=batch_user_prompt, **ask_gpt_params)
                outputs = json.loads(new_labels_gpt)['output']

            except Exception as e:  # If output from the model cannot be extracted
                print(f'Skipping indexes {i} and {i+1}, cannot extract output from the model. Error:')
                print(e, '\n')
                continue

            # Try to extract labels for the first record
            try:
                ner_tags_0 = [t[1] if len(t) > 1 and t[1] is not None else 'O' for t in outputs['record_0']]
                # If number of tokens == number of labels --> save result
                if len(record_0['true_ner_tags']) == len(ner_tags_0):
                    # Replace out-of-scope labels (e.g, 'B-EVENT')
                    ner_tags_0 = [label_to_index.get(t, 0) for t in ner_tags_0]

                    predicted_ner_tags.append(ner_tags_0)
                    true_ner_tags.append(record_0['true_ner_tags'])
                    indices_to_keep.append(i)
                else:
                    print(f'{i} --> different number of tokens and labels.')
            except Exception as e:
                print(f'{i} --> cannot extract NER labels. Error:')
                print(e, '\n')
                continue

            # Try to extract labels for the second record
            try:
                ner_tags_1 = [t[1] if len(t) > 1 and t[1] is not None else 'O' for t in outputs['record_1']]
                # If number of tokens == number of labels --> save result
                if len(record_1['true_ner_tags']) == len(ner_tags_1):
                    # Replace out-of-scope labels (e.g, 'B-EVENT')
                    ner_tags_1 = [label_to_index.get(t, 0) for t in ner_tags_1]

                    predicted_ner_tags.append(ner_tags_1)
                    true_ner_tags.append(record_1['true_ner_tags'])
                    indices_to_keep.append(i+1)
                else:
                    print(f'{i+1} --> different number of tokens and labels.')
            except Exception as e:
                print(f'{i+1} --> cannot extract NER labels. Error:')
                print(e, '\n')
                continue

        else:
            print(f'\nLast sample {i+1}:')

            record = dataset[i]

            # 1) Query model
            try:
                # Extract tokens from current record
                tokens = record['tokens']

                single_user_prompt = MAIN_PROMPT.format(
                    language=language_name,
                    sentence=str(tokens),
                    examples=single_examples
                )

                new_labels_gpt = ask_gpt_short(user_prompt=single_user_prompt, **ask_gpt_params)
                outputs = json.loads(new_labels_gpt)['output']

            except Exception as e:  # If output from the model cannot be extracted
                print(f'{i} --> cannot extract output from the model. Error:')
                print(e, '\n')
                continue

            # Try to extract labels for the first record
            try:
                ner_tags = [t[1] if len(t) > 1 and t[1] is not None else 'O' for t in outputs]
                # If number of tokens == number of labels --> save result
                if len(record['true_ner_tags']) == len(ner_tags):
                    # Replace out-of-scope labels (e.g, 'B-EVENT')
                    ner_tags = [label_to_index.get(t, 0) for t in ner_tags]
                    predicted_ner_tags.append(ner_tags)
                    true_ner_tags.append(record['true_ner_tags'])
                    indices_to_keep.append(i)
                else:
                    print(f'{i} --> different number of tokens and labels.')
            except Exception as e:
                print(f'{i} --> cannot extract NER labels. Error:')
                print(e, '\n')
                continue

    filtered_dataset = dataset.select(indices_to_keep)

    def update_dataset_with_gpt_annotations(example, index):
        example['ner_tags'] = predicted_ner_tags[index]
        return example

    updated_dataset = filtered_dataset.map(update_dataset_with_gpt_annotations, with_indices=True)

    updated_dataset = updated_dataset.map(
        align_labels_for_many_records,
        batched=True,
        fn_kwargs={'tokenizer': tokenizer}
    )

    # Calculate number of annotated and skipped records
    num_annotated_records = len(updated_dataset)
    print(f'{num_annotated_records} records were successfully annotated.')
    if num_annotated_records < original_dataset_len:
        print(f'{original_dataset_len - num_annotated_records} records were skipped.')

    true_ner_tags_flat = list(itertools.chain.from_iterable(true_ner_tags))
    predicted_ner_tags_flat = list(itertools.chain.from_iterable(predicted_ner_tags))
    
    # Calculate F1 score
    f1_score_micro = f1_score(true_ner_tags_flat, predicted_ner_tags_flat, average='micro')
    print(f'F1 micro: {round(f1_score_micro, 2)}')

    return updated_dataset


# In[13]:


# Download dataset for the specific language 
data = load_dataset(config['dataset'], LANGUAGE)
print("Original dataset:\n", data)

# splitting the data for active learning integration
data = balanced_split_for_active_learning(
    data,
    label_mapping,
    train_key='train',
    split_ratio=config['train_settings']['initial_train_size'],
    verbose=True
)


# In[14]:


data = data.map(
    align_labels_for_many_records,
    batched=True,
    fn_kwargs={'tokenizer': tokenizer}
)


# In[15]:


# Settings
max_len = config['languages_max_tokens'][LANGUAGE]
print(f'Maximum token length for language {LANGUAGE} is {max_len}')
padding_val = config['tokenizer_settings']['padding_value']

# Convert the datasets.Dataset to a PyTorch Dataset
dataset_init = TorchDataset(data['initial_training'], max_length=max_len, padding_value=padding_val)
dataset_unlabeled = TorchDataset(data['active_learning'], max_length=max_len, padding_value=padding_val)
dataset_test = TorchDataset(data['test'], max_length=max_len, padding_value=padding_val)
dataset_val = TorchDataset(data['validation'],max_length=max_len, padding_value=padding_val)


# In[16]:


# Settings
batch_size = config['train_settings']['batch_size']
shuffle = config['train_settings']['shuffle']

# Create PyTorch DataLoaders
dataloader_init = DataLoader(dataset_init,
                             batch_size=batch_size,
                             shuffle=shuffle) # Shuffle only training set
dataloader_unlabeled = DataLoader(dataset_unlabeled, batch_size=batch_size)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size)


# In[17]:


def train_ner(model, train_loader, val_loader, device, epochs, lr, updated_dataset_size, num_warmup_steps=5):
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
    best_val_f1 = 0.0

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
            "val_macro_f1": macro_f1,
            "updated_dataset_size": updated_dataset_size
        })

    return model, training_history


# In[18]:


# Prepare WandB for initial training
wandb.init(
    project=f'{EXPERIMENT}_{LANGUAGE}',
    name=f'{LANGUAGE}_initial',
    config={
        'epochs': config['train_settings']['epochs'],
        'learning_rate': config['train_settings']['lr'],
        'num_active_learning_rounds': config['active_learning_settings']['num_rounds']
    },
    settings=wandb.Settings(disable_job_creation=True)
)

# create a new model out of base model for re-training and prevent fine-tuning
base_model_for_train = copy.deepcopy(base_model)

# Initial model training
initial_model, history = train_ner(
    base_model_for_train,
    dataloader_init,
    dataloader_val,
    device,
    epochs=config['train_settings']['epochs'],
    lr=config['train_settings']['lr'],
    updated_dataset_size=len(dataset_init)
)

# Delete the model as it is no longer utilized.
del base_model_for_train

# Disable WandB logger
wandb.finish()


# In[19]:


# Model evaluation before re-training
print_classification_report(config, initial_model, dataloader_test, device, ignore_index=-100, ignore_class=0)


# In[20]:


# Select initial model as best model for the first round of active learning
best_model = copy.deepcopy(initial_model)

# Delete the model as it is no longer utilized.
del initial_model

# Use initial dataset to be combined with re-annotated uncertain samples for first round of active learning 
combined_dataset = copy.deepcopy(dataset_init)


# In[21]:


# Active learning framework
for active_learning_round in range(config['active_learning_settings']['num_rounds']):
    print(10*"=" + f" Active Learning - Round {active_learning_round+1} " + 10*"=")
    # Prepare WandB for active learning
    wandb.init(
        project=f'{EXPERIMENT}_{LANGUAGE}',
        name=f'{LANGUAGE}_round_{active_learning_round+1}',
        config={
            'epochs': config['train_settings']['epochs'],
            'learning_rate': config['train_settings']['lr'],
            'num_active_learning_rounds': config['active_learning_settings']['num_rounds']
        },
        settings=wandb.Settings(disable_job_creation=True)
    )

    uncertain_samples = predict_sequence_max_uncertainty(
        best_model,
        dataloader_unlabeled,
        device,
        fraction=config['train_settings']['label_fraction']
    )

    # Delete the model as it is no longer utilized.
    del best_model

    global_indices = batch_indices_to_global_indices(uncertain_samples,
                                                    batch_size)

    # Create a subset from the pseudo-unlabeled data
    dataset_tune = data['active_learning'].select(global_indices)

    # 1) Rename columns to save old labels
    dataset_tune = dataset_tune.rename_column('ner_tags', 'true_ner_tags')
    # 2) Drop aligned labels -> need to realign after LLM querying
    dataset_tune = dataset_tune.remove_columns(['labels', 'attention_mask', 'input_ids'])
    # 3) Add empty column for predicted labels
    dataset_tune = dataset_tune.map(lambda example: {**example, 'ner_tags': None})

    # Get GPT 
    dataset_tune = update_ner_annotations_with_gpt(dataset_tune, batch_examples, single_examples)

    # Convert datasets.dataset to PyTorch dataset
    converted_dataset_tune = TorchDataset(dataset_tune,
                                        max_length=max_len,
                                        padding_value=padding_val)

    # Combine previous training data with newly labeled data
    combined_dataset = torch.utils.data.ConcatDataset([combined_dataset,
                                                    converted_dataset_tune])
    print("Size of updated re-training dataset: ", len(combined_dataset))

    # Create dataloader
    dataloader_tune = DataLoader(combined_dataset, batch_size=batch_size)

    # create a new model out of base model for re-training and prevent fine-tuning
    base_model_for_train = copy.deepcopy(base_model)

    # Re-train the model
    best_model, history = train_ner(
        base_model_for_train,
        dataloader_tune,
        dataloader_val,
        device,
        epochs=config['train_settings']['epochs'],
        lr=config['train_settings']['lr'],
        updated_dataset_size=len(combined_dataset)
    )

    # Delete the model as it is no longer utilized.
    del base_model_for_train

    # Model evaluation after each round of active learning
    print_classification_report(config, best_model, dataloader_test, device, ignore_index=-100, ignore_class=0)

    # Disable WandB logger
    wandb.finish()


# In[22]:


# Save model
torch.save(best_model.state_dict(), f'model_weights/{LANGUAGE}/{EXPERIMENT}.pth')


# ### Final model evaluation

# In[23]:


print_classification_report(config, best_model, dataloader_test, device, ignore_class=0)


# ### Cleaning  up GPU memory

# In[24]:


# Clear memory

# Delete all models as they are no longer utilized.
del base_model
del best_model

# Using garbage collector
import gc
gc.collect()

torch.cuda.empty_cache() 


# ### WandB logging out

# In[25]:


# try:
#     import os
#     PATH_TO_SRC = os.path.abspath('../../../')
#     os.remove(f'{PATH_TO_SRC}/../../.netrc')
#     print("Logged out of WandB.")
# except Exception as e:
#     print(e)
#     print("Unsuccessful WandB logging out.")

