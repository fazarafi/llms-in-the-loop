#!/usr/bin/env python
# coding: utf-8

# # Inter-annotator agreement (IAA)

# To access IAA, masakhaner2 authors calculated Fleiss' Kappa on entity level. We follow this approach and compare IAA for model-annotated vs human-annotated data. Instead of having multiple annotators, we ask the model to reannotate the same samples N=10 times. 
# 
# We follow the logic that a good model should assign the labels "without any doubts" and therefore it should have high IAA score.

# In[95]:


models = ['gpt-4-turbo-0.1-temp', 
          'gemini-1.0_temp_0.1', 'gemini-1.0_batch_size_2',
          'mistral-0.1-temp', 
          'starling-lm-0.1-temp', 
          'llama_70b',
          'claude3-sonet',
          'gpt_4_turbo_batch_size_2', 'gpt-4-turbo', 'claude3-opus', 'claude3-sonet']
'''
['claude3-opus', 'gpt-4', 'gpt-4-turbo', 'gpt_4_turbo_batch_size_2',
 'gemini-1.0', 'gemini-1.0_temp_0.1', 'gemini-1.0_batch_size_2',
 'gemini-1.0_shorter_prompt', 'llama_70b', 'mistral', 'starling-lm',
 'starling-lm-0.1-temp', 'starling-lm-zero-temp']
'''
languages = ['zul', 'bam', 'tsn', 'fon', 'bbj', 'swa']


# 
# ## Fleiss` Kappa

# ### Main steps & calculation example

# In[96]:


import json
import numpy as np
from IPython.display import display
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
import os
import yaml
from collections import Counter
from collections import defaultdict
from sklearn.metrics import f1_score
import sys
from contextlib import contextmanager

@contextmanager
def extend_sys_path(path):
    if path not in sys.path:
        # Append the path to sys.path
        sys.path.append(path)
    try:
        # Execute code inside the 'with' statement
        yield
    finally:
        # Remove the path from sys.path
        if path in sys.path:
            sys.path.remove(path)


# In[97]:


SAMPLE_SIZE = 50
REPEAT_ANNOTATION = 10


# In[98]:


# Specifying path to the necessary files and folders
PATH_TO_SRC = os.path.abspath('../../../')

CONFIG_PATH = os.path.join(PATH_TO_SRC, "settings/config.yml")
RESULTS_PATH = os.path.join(PATH_TO_SRC, 'data/foundation_model_selection')


# In[99]:


with extend_sys_path(PATH_TO_SRC):
    from src.utils.utils import calculate_consistency_score


# In[100]:


# Reading config file
config = yaml.safe_load(open(os.path.join(PATH_TO_SRC, "settings/config.yml")))

# Load unique label categories
unique_labels = list(config['label_mapping'].values())
unique_labels


# Creating matrix in the following format:
# 
# 
# 
# ```
# [
#  [3, 0, 0],  # Sentence 1, Token 1
#  [0, 2, 1],  # Sentence 1, Token 2
#  [1, 1, 1],  # Sentence 1, Token 3
#  [0, 0, 3],  # Sentence 1, Token 4
#  [2, 1, 0],  # Sentence 2, Token 1
#  [1, 2, 0],  # Sentence 2, Token 2
#  [3, 0, 0],  # Sentence 2, Token 3
#  [0, 1, 2],  # Sentence 2, Token 4
# ]
# ```
# , where columns correspond to unique label categories, and values correspond to the number of annotators assigned the category.
# 
# 
# 

# In[101]:


example_bbj = json.load(open(os.path.join(RESULTS_PATH, 'gemini-1.0/bbj.json')))


# Annotations for the first sentence:

# In[102]:


sent_ann = []

# For each reannnotation for record_0
for prediction in example_bbj['record_0']['pred']:
    # Extract labels
    pred_label = [t[1] for t in prediction]

    sent_ann.append(pred_label)
    print(pred_label)


# Creating matrix:

# In[103]:


def annotations_to_matrix(sentence_annotations, unique_labels):

    # Shape -> number of tokens in sentence * number of unique categories
    label_counts = np.zeros((len(sentence_annotations[0]), len(unique_labels)))

    # For each annotation attempt
    for annotation in sentence_annotations:
        # For each token in sentence
        for i, label in enumerate(annotation):
            # Label position (column)
            label_index = unique_labels.index(label)
            # Append annotator counts
            label_counts[i][label_index] += 1
    return label_counts


pd.DataFrame(annotations_to_matrix(sent_ann, unique_labels), columns=unique_labels)\
    .style.applymap(lambda x: 'background-color: lightblue' if x > 0 else 'background-color: white')


# Creating matrix for all records:

# In[104]:


def get_aggregate_matrix(data, sample_size, unique_labels):
    records = [f'record_{i}' for i in range(sample_size)]

    all_counts = []
    
    # For each record (record_0, record_1, record_2...)
    for record in records:
        # Get annotations for this record
        record_ann = []
        if record in data:
            # Extract predicted labels
            for prediction in data[record]['pred']:
                if len(prediction) > 0:
                    try:
                        pred_label = [t[1] for t in prediction]
                        record_ann.append(pred_label)
                    except Exception as e:
                        print(prediction)
        else:
            print(f'This record is omitted: {record}')
    
        if len(record_ann) > 0:
            try:
                sentence_matrix = annotations_to_matrix(record_ann, unique_labels)
                all_counts.append(sentence_matrix)
            except Exception as e:
                print(e)
                print(record)
                print(record_ann)
                continue
    
    # Concatenate all sentence matrices vertically
    aggregate_matrix = np.vstack(all_counts)
    
    return pd.DataFrame(aggregate_matrix, columns=unique_labels)    


# In[105]:


df = get_aggregate_matrix(example_bbj, SAMPLE_SIZE, unique_labels)
df.head()


# Per-row sum should be equal to the number of annotators (reannotation attempts):

# In[106]:


df['skipped_annotations'] = REPEAT_ANNOTATION - df[unique_labels].sum(axis=1)

df[df['skipped_annotations'] != 0]


# In[107]:


df_no_skipped_ann = df[df['skipped_annotations'] == 0].copy()
df_no_skipped_ann.shape


# In[108]:


df[df['skipped_annotations'] != 0].shape


# Replacing all skipped annotations with non-entity tokens to have consistent number of annotations per sentence:

# In[109]:


df['O'] += df['skipped_annotations']

df['skipped_annotations'] = REPEAT_ANNOTATION - df[unique_labels].sum(axis=1)

df[df['skipped_annotations'] != 0]


# In[110]:


# Calculate Fleiss' Kappa for the aggregated annotation data
kappa = fleiss_kappa(df[unique_labels], method='fleiss')
print(f"Fleiss' Kappa: {round(kappa, 4)}")

kappa = fleiss_kappa(df_no_skipped_ann[unique_labels], method='fleiss')
print(f"Fleiss' Kappa for records without missing annotations: {round(kappa, 4)}")


# ### Kappa calculation for foundation models' annotations

# In[111]:


class FleissKappaCalculator:
    def __init__(self, sample_size, repeat_annotation, unique_labels):
        self.sample_size = sample_size
        self.repeat_annotation = repeat_annotation
        self.unique_labels = unique_labels
        self.log = ''
        self.counter = {
            'num_skipped_records': 0,
            'skipped_records': [],
            'different_ann_length': 0,
            'contain_empty_predictions': 0, 
        }

    def annotations_to_matrix(self, sentence_annotations):
        # Shape -> number of tokens in sentence * number of unique categories
        label_counts = np.zeros((len(sentence_annotations[0]), len(self.unique_labels)))
    
        # For each annotation attempt
        for annotation in sentence_annotations:
            # For each token in sentence
            for i, label in enumerate(annotation):
                # Label position (column)
                label_index = unique_labels.index(label)
                # Append annotator counts
                label_counts[i][label_index] += 1
        return label_counts      
          
    def get_aggregate_matrix(self, data):
        records = [f'record_{i}' for i in range(self.sample_size)]
        all_counts = []
    
        # For each record (record_0, record_1, record_2...)
        for record in records:
            # Get annotations for this record
            record_ann = []
            skip_flag = False

            if record in data:
                
                # Counting number of skipped annotations per record
                if len(data[record]['pred']) < self.repeat_annotation:
                    self.log += (f'{record} --> less than 10 annotations for record '
                                 f'(num annotations: {len(data[record]["pred"])})\n')
                    
                # Counting length of each prediction (should be equal)
                lengths = [len(pred) for pred in data[record]['pred']]
                length_counts = dict(Counter(lengths))
                if 0 in length_counts:
                    self.log += f'{record} --> has {length_counts[0]} empty predictions ([]).\n'
                   
                # Combining two previous conditions
                if (len(data[record]['pred']) < self.repeat_annotation or
                        0 in length_counts):
                    self.counter['contain_empty_predictions'] += 1
                    skip_flag = True
                    
                # Check if there's more than one length
                if len(length_counts) > 1:
                    self.log += f'{record} --> different number of predicted labels ({dict(length_counts)}), true num labels = {len(data[record]["tokens"])}.\n'
                    self.counter['skipped_records'].append(record)
                    self.counter['different_ann_length'] += 1
                    continue
                
                elif skip_flag is True:
                    self.counter['skipped_records'].append(record)
                    continue
                    
                else:
                    # Extract predicted labels
                    for prediction in data[record]['pred']:
                        if len(prediction) > 0:
                            try:
                                pred_label = [t[1] if len(t) > 1 else "O" for t in prediction]
                                record_ann.append(pred_label)
                            except Exception as e:
                                self.log += f'{record} --> cannot extract tokens.\n'
                                self.counter['skipped_records'].append(record)
                            
                    
            else:
                self.log += f'{record} --> record is not in the data.\n'
                self.counter['skipped_records'].append(record)
    
            if len(record_ann) > 0:
                try:
                    sentence_matrix = self.annotations_to_matrix(record_ann)
                    all_counts.append(sentence_matrix) # Create sent matrix as shown above
                except ValueError as e:  # If additional labels are encountered (e.g., B-MISC)
                    self.log += f'{record} --> has additional tokens ({e}).\n'
                    self.counter['skipped_records'].append(record)
                    continue

        self.counter['num_skipped_records'] = len(self.counter['skipped_records'])
        self.log += f'Number of omitted records: {self.counter["num_skipped_records"]}\n'
        # Concatenate all sentence matrices vertically
        aggregate_matrix = np.vstack(all_counts)
        return pd.DataFrame(aggregate_matrix, columns=unique_labels)

    def get_kappa(self, data):
        """Calculate Fleiss' Kappa for foundation model annotations"""
        # 1. Get aggregated matrix
        df = self.get_aggregate_matrix(data)
        # 2. Calculate Fleiss' Kappa for the aggregated annotation data
        kappa = fleiss_kappa(df, method='fleiss')
        
        return kappa


# In[ ]:


metrics = {
    'kappa': defaultdict(dict),
    'num_skipped_records': defaultdict(dict),
    'different_ann_length': defaultdict(dict),
    'contain_empty_predictions': defaultdict(dict)
}

logs = {}

# Loop through each model and language to gather metrics
for model in models:
    for language in languages:
        try:
            data_path = os.path.join(RESULTS_PATH, model, language + '.json')
            with open(data_path) as f:
                data = json.load(f)

            # Calculate Fleiss' Kappa, extract logs
            calc = FleissKappaCalculator(SAMPLE_SIZE, REPEAT_ANNOTATION, unique_labels)
            kappa = calc.get_kappa(data)
            logs[f'{model} | {language}'] = calc.log

            # Assign values to each metric dictionary
            metrics['kappa'][model][language] = kappa
            metrics['num_skipped_records'][model][language] = calc.counter['num_skipped_records']
            metrics['different_ann_length'][model][language] = calc.counter['different_ann_length']
            metrics['contain_empty_predictions'][model][language] = calc.counter['contain_empty_predictions']

        except Exception as e:
            print(e)
            # Assign None for each metric in case of an exception
            for metric in metrics.keys():
                metrics[metric][model][language] = None

# Convert metrics dictionaries into pandas DataFrames
kappa_results = pd.DataFrame(metrics['kappa']).T
skipped_records = pd.DataFrame(metrics['num_skipped_records']).T
different_ann_length = pd.DataFrame(metrics['different_ann_length']).T
contain_empty_predictions = pd.DataFrame(metrics['contain_empty_predictions']).T

# Manually update entries for human annotations in the kappa_results DataFrame
kappa_results.loc['human annotation in masakhaner2', 'bbj'] = 1.000
kappa_results.loc['human annotation in masakhaner2', 'zul'] = 0.953 
kappa_results.loc['human annotation in masakhaner2', 'bam'] =  0.980
kappa_results.loc['human annotation in masakhaner2', 'fon'] = 0.941
kappa_results.loc['human annotation in masakhaner2', 'tsn'] =  0.962

print('\nFleiss` Kappa score')
kappa_results.round(3)


# In[113]:


print(f'Number of records with different number of predicted tokens')
different_ann_length.round()


# In[114]:


print(f'Number of records containing empty predictions or records where number of reannotations is less than 10')
contain_empty_predictions


# In[115]:


print(f'Total skipped records')
skipped_records


# In[116]:


print(f'% of skipped records')
skipped_records / SAMPLE_SIZE


# In[117]:


for k, v in logs.items():
    if len(v) > 0:
        print(k)
        print(v)


# # F1-Score

# In[118]:


skipped_records_df = pd.DataFrame(index=models, columns=languages)
f1_df = pd.DataFrame(index=models, columns=languages)

records = [f'record_{i}' for i in range(SAMPLE_SIZE)]

for model in models:
    for language in languages:
        print(f'{model} | {language}')
        try:
            # Initialize variables for each language iteration
            pred, true = [], []
            skipped_records = 0

            # Load data
            data_path = os.path.join(RESULTS_PATH, model, language + '.json')
            with open(data_path, 'r') as file:
                data = json.load(file)

            # Process each record
            for record in records:
                if record in data:  # Check if the record exists in the data
                    non_empty_pred = [pred for pred in data[record]['pred'] if len(pred) > 0]

                    if non_empty_pred:
                        first_pred = non_empty_pred[0]  # Selecting first non-empty record
                        pred_labels = [t[1] if len(t) > 1 else "O" for t in first_pred]

                        if len(data[record]['true']) == len(pred_labels):
                            pred.extend(pred_labels)
                            true.extend(data[record]['true'])
                        else:
                            skipped_records += 1
                            print(f'{record} --> different number of labels (pred={len(pred_labels)}, true={len(data[record]["true"])}).')
                    else:
                        skipped_records += 1
                        print(f'{record} --> all predictions are empty.')
                else:
                    skipped_records += 1  # Increment skipped records if not found
                    print(f'{record} --> not in data.')

            # Calculate F1 score if applicable
            f1_score_value = f1_score(true, pred, average='micro') if true and pred else None

            # Assign calculated values to the respective DataFrame cells
            skipped_records_df.at[model, language] = skipped_records
            f1_df.at[model, language] = f1_score_value

        except Exception as e:
            print(f"Error processing {model} in {language}: {e}")
            skipped_records_df.at[model, language] = None
            f1_df.at[model, language] = None
        print()


# In[119]:


f1_df


# In[120]:


skipped_records_df


# # Consistency 

# In[121]:


@contextmanager
def extend_sys_path(path):
    if path not in sys.path:
        # Append the path to sys.path
        sys.path.append(path)
    try:
        # Execute code inside the 'with' statement
        yield
    finally:
        # Remove the path from sys.path
        if path in sys.path:
            sys.path.remove(path)
            


# In[ ]:


records = [f'record_{i}' for i in range(SAMPLE_SIZE)]
consistency_df = pd.DataFrame(index=models, columns=languages)

for model in models:
    for language in languages:
        consistency = []
        try:
            # Load data
            data_path = os.path.join(RESULTS_PATH, model, language + '.json')
            with open(data_path, 'r') as file:
                data = json.load(file)

            # Process each record
            for record in records:
                if record in data:  # Check if the record exists in the data
                    while len(data[record]['pred']) < REPEAT_ANNOTATION:
                        data[record]['pred'].append([])  # Append an empty array

                    consistency.append(calculate_consistency_score(data[record]['pred'], data[record]['true']))
                else:
                   consistency.append(0)

            consistency_df.at[model, language] = np.mean(consistency)
        except Exception as e:
            print(e)
            consistency_df.at[model, language] = None


# In[123]:


consistency_df


# Keep code below to easy paste the results from different dataframes to tables in latex

# In[ ]:


for model in consistency_df.T.columns:
    
    # Round each value in the column to 1 decimal place, convert to string, and then join with ' & '
    rounded_values = consistency_df.T[model].apply(lambda x: 'None' if pd.isnull(x) else str(round(x, 1))).values

    print("{:<30}".format(model), ' & '.join(filter(None, rounded_values)))


# In[ ]:


' & '.join(contain_empty_predictions.T.index)


# ### Claude VS GPT

# In[138]:


for model in models:
    for language in languages:
        consistency = []
  
        data_path = os.path.join(RESULTS_PATH, model, language + '.json')
        try:
            with (open(data_path, 'r') as file):
                data = json.load(file)
                # For gpt use first record out of 10 reannotations
                if 'gpt' in model:
                    consistency_score = []
                    for record in records:
                        if record in data:
                            consistency_score.append(calculate_consistency_score(
                                [data[record]['pred'][0]], data[record]['true']))
                        else:
                            consistency_score.append(0)
                    print(model, language, round(np.mean(consistency_score), 2))
                elif 'claude' in model:
                    print(model, language, round(data['overall_consistency'], 2))
        except FileNotFoundError as e:
            continue
        except Exception as e:
            print(e)

