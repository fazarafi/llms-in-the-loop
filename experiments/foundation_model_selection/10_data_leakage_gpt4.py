#!/usr/bin/env python
# coding: utf-8

# ## Setup & Initialization

# ### Imports and variables

# In[1]:


import os
import sys
import yaml
from contextlib import contextmanager
from datasets import load_dataset, DatasetDict, concatenate_datasets
from openai import OpenAI
from tqdm.notebook import tqdm
from openai import APIStatusError, RateLimitError, APIConnectionError
import numpy as np
import time
from getpass import getpass
import datetime
from collections import Counter
import json
import requests
import regex as re
from datasets.utils import disable_progress_bar
import random


# In[2]:


# gpt-4 -> GPT-4 cutoff 2021
MODEL = 'gpt-4'
# Testing leakage with 0 temperature for more deterministic results
TEMPERATURE = 0


# In[3]:


# Specifying path to the necessary files and folders
PATH_TO_SRC = os.path.abspath('../../../')
CONFIG_PATH = os.path.join(PATH_TO_SRC, "settings/config.yml")
RESULTS_PATH = os.path.join(PATH_TO_SRC, 'data/data_leakage', MODEL)

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)


# In[4]:


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


# In[5]:


# Temporarily add module_path and import functions
with extend_sys_path(PATH_TO_SRC):
    from src.utils.utils import get_api_usage


# In[6]:


# Init openai client
openai_client = OpenAI(api_key=getpass("OPENAI API key:"))


# In[7]:


ASK_GPT_PARAMS = {
    'openai_client': openai_client,
    'model': MODEL,
    'temperature': TEMPERATURE
}


# In[8]:


# Reading config file
config = yaml.safe_load(open(os.path.join(PATH_TO_SRC, "settings/config.yml")))


# ### Utils

# In[9]:


def sample_from_dataset(dataset_dict, num_samples=10, seed=42):
    """
    Randomly samples a specified number of examples from each split 
    in a DatasetDict in a reproducible manner.
    """
    # Set the numpy random seed for reproducibility
    np.random.seed(seed)

    sampled_dataset_dict = DatasetDict()

    # Iterate over each split in the original DatasetDict
    for split, dataset in dataset_dict.items():
        # Generate a list of random indices with no replacement
        indices = np.random.choice(len(dataset), num_samples, replace=False)

        # Select the samples corresponding to the generated indices
        sampled_dataset = dataset.select(indices)

        # Add the sampled dataset to the new DatasetDict
        sampled_dataset_dict[split] = sampled_dataset

    return sampled_dataset_dict


# In[10]:


def ask_gpt(
        user_prompt, 
        openai_client,
        max_tokens=1000,
        temperature=0.7,
        model='gpt-4-0125-preview',
        system_prompt=None):

    if system_prompt is None:
        system_prompt = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."

    # Save query params
    query_params = {
        'model': model,
        'temperature': temperature,
        'messages': [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        'max_tokens': max_tokens,
    }

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


# In[11]:


def compute_guess_stats(answers, pattern):
    """
    Computes the correct guess counts and percentages for each dictionary in answers based on a regex pattern.
    """
    correct_guess_counts = []
    correct_guess_percentages = []

    for answer_dict in answers:
        match_count = sum(answer_dict[key] for key in answer_dict if re.search(pattern, key))
        num_records = sum(answer_dict.values())

        correct_guess_counts.append(match_count)
        correct_guess_percentages.append(match_count / num_records if num_records else 0)

    mean_percentage = round(np.mean(correct_guess_percentages), 2)
    std_deviation = round(np.std(correct_guess_percentages), 2)

    return mean_percentage, std_deviation


# ### Prompts

# In[12]:


# Guided instruction for sentence continuation 
GUI_SENT = """INSTRUCTION:
You are provided with the FIRST PIECE of data sample tokens from the {split_name} split of the {dataset_name} dataset. Finish the SECOND PIECE of the sample tokens as EXACTLY appeared in the dataset. ONLY rely on the original form of the sample tokens in the dataset to finish the SECOND PIECE. Do not return labels, provide only tokens exactly as they appear in the dataset.

FIRST PIECE:
{sentence}

SECOND PIECE:"""


# In[13]:


# GPT-4 evaluation prompt
ICL_EVAL = """INSTRUCTION:
You are provided with a piece of reference text and a piece of candidate text.
The candidate text is a generated replica of the reference text.
Your task is to determine if the candidate text is an exact or near-exact match of the reference text.
Near-exact match candidates must match the reference candidates in terms of sentence structure, overlap, and contextual similarity.
Respond only with 'Yes' or 'No'.
---
Example 1:
REFERENCE TEXT:
The cat waited at the top.
CANDIDATE TEXT:
The cat waited at the top.

ANSWER: Yes (exact match)
---
Example 2:
REFERENCE TEXT:
'نيرو 2'
CANDIDATE TEXT:
'نيرو '

ANSWER: Yes (near-exact match)
---
Example 3:
REFERENCE TEXT:
Japan) Suzuki 1:27.501
CANDIDATE TEXT:
JPN) Kawasaki 1:27.808

ANSWER: Yes (near-exact match)
---
Example 4:
REFERENCE TEXT:
1996-08-22
CANDIDATE TEXT:
1996-08-26

ANSWER: Yes (near-exact match)
---
Example 5:
REFERENCE TEXT:
{reference_text}
CANDIDATE TEXT:
{candidate_text}

ANSWER:
"""


# # Asking LLM about data sample source

# A few previous experiments have shown that a foundation model (currently we focus on GPT-4-Turbo) answers correctly name of the dataset for randomly chosen samples from old datasets. At the same time, it is not able to answer the source of the records from our NER dataset masakhaner2.
# 
# We evaluate how well the model knows the source for the random data samples from a few famous datasets and our focus dataset masakhaner2. 
# 

# ## GPT-4-Turbo

# ### CoNLL-2003

# In[14]:


user_prompt = """Identify the source NER dataset for this sample. Respond with the dataset name alone. {sentence}
"""


# In[15]:


conll = load_dataset("conll2003")

conll


# In[16]:


repeat_experiment = 3
conll_answers = []

for experiment_i in tqdm(range(repeat_experiment), desc="Experiment #"):
    # Dictionary to save current experiments results
    experiment_answers = {}

    # Random sampling of N records from each split in the dataset
    conll_samples = sample_from_dataset(conll, num_samples=10, seed=experiment_i)
    
    # Iterate over each data split and its corresponding samples
    for data_split, samples in conll_samples.items():
        # Store answers for each split
        experiment_answers[data_split] = {}
    
        # For each sample in the current data split -> ask source
        for i, (id, tokens) in enumerate(zip(samples['id'], samples['tokens'])):
            # Format prompt with current data sample
            query = user_prompt.format(sentence=str(tokens))
            experiment_answers[data_split][id] = ask_gpt(query, **ASK_GPT_PARAMS).lower()

    conll_answers.append(experiment_answers)

conll_answers


# In[17]:


with open(os.path.join(RESULTS_PATH, f'CoNLL-2003_sample_source.json'), 'w') as file:
    json.dump(conll_answers, file, indent=4)


# In[18]:


# Calculate number of times the data sample source was guessed correctly
conll_answers = json.load(open(os.path.join(RESULTS_PATH, f'CoNLL-2003_sample_source.json'), 'r'))


# In[19]:


# Count number of times each dataset name is predicted
conll_counts = []

for experiment_i in tqdm(range(repeat_experiment), desc="Experiment #"):
    conll_answers_list = []

    experiment_answers = conll_answers[experiment_i]
    for split, samples in experiment_answers.items():
        conll_answers_list += list(samples.values())

    # Append current experiment answers
    conll_counts.append(dict(Counter(conll_answers_list)))

conll_counts


# In[20]:


mean, std = compute_guess_stats(answers=conll_counts, pattern=re.compile(r'conll.*03'))
print(f'Mean: {mean}')
print(f'Std: {std}')


# ### WikiAnn

# WikiAnn is a dataset for cross-lingual name tagging and linking based on Wikipedia articles in 295 languages.

# In[21]:


# Get languages from the dataset
url = "https://datasets-server.huggingface.co/splits?dataset=wikiann"

# Send a GET request
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
else:
    print(f"Failed to fetch data: {response.status_code}")


# In[22]:


wikiann_languages = sorted(list({item['config'] for item in data['splits']}))
wikiann_languages[:5]


# In[23]:


# Generate a list of random languages with no replacement
np.random.seed(42)
num_languages = 3

sampled_languages = np.random.choice(wikiann_languages, size=num_languages, replace=False)
sampled_languages


# bh: Bihari languages. It's a group of languages spoken in the Bihar region of India, but "bh" is often used specifically to refer to Bhojpuri.
# et: Estonian. A Uralic language spoken primarily in Estonia.
# sk: Slovak. A West Slavic language spoken in Slovakia.
# 

# In[24]:


def add_id(example, idx):
    example['id'] = idx
    return example


# In[25]:


user_prompt = """Identify the source multilingual NER dataset for this sample. Respond with the dataset name alone. {sentence}
"""


# In[26]:


repeat_experiment = 3
wikiann_answers = []

# Disable datasets load_dataset progress bar
disable_progress_bar()

for experiment_i in tqdm(range(repeat_experiment), desc="Experiment #"):
    experiment_answers = {}

    for i, lan in enumerate(sampled_languages):
        # Load wikiann split for each of the sampled languages
        wikiann = load_dataset("wikiann", data_dir=lan, 
                               download_mode="force_redownload",
                               verification_mode="no_checks")
    
        # Concatenate train, test, and validation splits 
        wikiann = DatasetDict({
            "merged": concatenate_datasets([wikiann['train'], wikiann['test'], wikiann['validation']])
        })
        # Add ids per records based on the index of the record
        wikiann = wikiann.map(add_id, with_indices=True)
        # Sample
        wikiann_samples = sample_from_dataset(wikiann, 10, seed=experiment_i)['merged']

        experiment_answers[lan] = {}
        # Ask source for the sampled records
        for id_, tokens in zip(wikiann_samples['id'], wikiann_samples['tokens']):
            # Format prompt with current data sample
            query = user_prompt.format(sentence=str(tokens))
            experiment_answers[lan][id_] = ask_gpt(query, **ASK_GPT_PARAMS).lower()

    wikiann_answers.append(experiment_answers)

wikiann_answers


# In[27]:


with open(os.path.join(RESULTS_PATH, f'WikiAnn_sample_source.json'), 'w') as file:
    json.dump(wikiann_answers, file, indent=4)


# In[28]:


# Calculate number of times the data sample source was guessed correctly
wikiann_answers = json.load(open(os.path.join(RESULTS_PATH, f'WikiAnn_sample_source.json'), 'r'))

# Count number of times each dataset name is predicted
wikiann_counts = []

for experiment_i in tqdm(range(repeat_experiment), desc="Experiment #"):
    wikiann_answers_list = []

    experiment_answers = wikiann_answers[experiment_i]
    for split, samples in experiment_answers.items():
        wikiann_answers_list += list(samples.values())

    # Append current experiment answers
    wikiann_counts.append(dict(Counter(wikiann_answers_list)))

wikiann_counts


# In[29]:


mean, std = compute_guess_stats(answers=wikiann_counts, pattern=re.compile(r'(wikiann.*|xtreme)'))
print(f'Mean: {mean}')
print(f'Std: {std}')


# ### masakhaner2

# In[30]:


# Languages that were added to the second version of the masakhaner2
masakhaner2_languages = ['bam', 'ewe', 'fon', 'twi', 'bbj', 'nya', 'tsn', 'sna', 'xho', 'zul']

# Generate a list of random languages 
np.random.seed(42)
num_languages = 3

sampled_languages = np.random.choice(masakhaner2_languages, size=num_languages, replace=False)
sampled_languages


# In[31]:


user_prompt = """Identify the source multilingual NER dataset for this sample. Respond with the dataset name alone. {sentence}
"""


# In[32]:


repeat_experiment = 3
masakhaner2_answers = []

# Disable datasets load_dataset progress bar
disable_progress_bar()

for experiment_i in tqdm(range(repeat_experiment), desc="Experiment #"):
    experiment_answers = {}

    for i, lan in enumerate(sampled_languages):
        # Load wikiann split for each of the sampled languages
        masaknaner2 = load_dataset(config['dataset'], lan)

        # Concatenate train, test, and validation splits 
        masaknaner2 = DatasetDict({
            "merged": concatenate_datasets(
                [masaknaner2['train'], masaknaner2['test'], masaknaner2['validation']])
        })
        # Sample
        masaknaner2_samples = sample_from_dataset(masaknaner2, 10)['merged']

        experiment_answers[lan] = {}
        # Ask source for the sampled records
        for id_, tokens in zip(masaknaner2_samples['id'], masaknaner2_samples['tokens']):
            # Format prompt with current data sample
            query = user_prompt.format(sentence=str(tokens))
            experiment_answers[lan][id_] = ask_gpt(query, **ASK_GPT_PARAMS).lower()

    masakhaner2_answers.append(experiment_answers)

masakhaner2_answers


# In[33]:


with open(os.path.join(RESULTS_PATH, f'masakhaner2_sample_source.json'), 'w') as file:
    json.dump(masakhaner2_answers, file, indent=4)


# In[34]:


# Calculate number of times the data sample source was guessed correctly
masakhaner2_answers = json.load(open(os.path.join(RESULTS_PATH, f'masakhaner2_sample_source.json'), 'r'))

# Count number of times each dataset name is predicted
masakhaner2_counts = []

for experiment_i in tqdm(range(repeat_experiment), desc="Experiment #"):
    masakhaner2_answers_list = []

    experiment_answers = masakhaner2_answers[experiment_i]
    for split, samples in experiment_answers.items():
        masakhaner2_answers_list += list(samples.values())

    # Append current experiment answers
    masakhaner2_counts.append(dict(Counter(masakhaner2_answers_list)))

masakhaner2_counts


# In[63]:


mean, std = compute_guess_stats(answers=masakhaner2_counts, pattern=re.compile(r'masakhaner 2.0'))
print(f'Mean: {mean}')
print(f'Std: {std}')


# In[61]:


# Calculate number of times the data sample source was guessed incorrectly
mean, std = compute_guess_stats(answers=masakhaner2_counts, 
                                pattern=re.compile(r'(masakhanener ?$|masakhaner ?$|masakhane ner)'))
print(f'Mean: {mean}')
print(f'Std: {std}')


# # Asking the model to continue sentences

# Following the approach suggested here: https://arxiv.org/abs/2308.08493

# ### CoNLL-2003

# In[37]:


conll = load_dataset("conll2003")
conll_samples = sample_from_dataset(conll, num_samples=10)['train']
conll_samples


# In[38]:


np.random.seed(42)
results = {}

for id_, tokens in tqdm(zip(conll_samples['id'], conll_samples['tokens']), total=10):
    results[id_] = {}
    
    # Cut-off - at least 2 tokens at the beginning
    cut_off = int(np.ceil(len(tokens)/2))

    query_text = ' '.join(tokens[:cut_off])
    reference_text = ' '.join(tokens[cut_off:])
    
    user_prompt = GUI_SENT.format(
        split_name='train', 
        dataset_name='CoNLL-2003',
        sentence=query_text
    )
     
    answer = ask_gpt(user_prompt, **ASK_GPT_PARAMS)

    results[id_]['query_text'] = query_text
    results[id_]['reference_text'] = reference_text
    results[id_]['answer'] = answer   


# In[39]:


with open(os.path.join(RESULTS_PATH, f'CoNLL-2003_sentence_continuation.json'), 'w') as file:
    json.dump(results, file, indent=4)


# In[40]:


results = json.load(open(os.path.join(RESULTS_PATH, f'CoNLL-2003_sentence_continuation.json'), 'r'))


# In[41]:


for id_, results_dict in tqdm(results.items()):
    print('--> Start of the sentence:\n', results_dict['query_text'])
    print("--> Reference (ground truth):")
    print(results_dict['reference_text'])
    print("--> Model's prediction:")
    print(results_dict['answer'])
    print()
    


# In[42]:


match_arr = []

for id_, results_dict in tqdm(results.items(), total=10):
    
    user_prompt = ICL_EVAL.format(
        reference_text=results_dict['reference_text'],
        candidate_text=results_dict['answer']
    )
    answer = ask_gpt(user_prompt, openai_client, temperature=TEMPERATURE, model='gpt-4')
    results[id_]['match'] = answer

    match_arr.append(answer)
    
Counter(match_arr)


# ### WikiANN (Multilingual)

# In[43]:


# Get languages from the dataset
url = "https://datasets-server.huggingface.co/splits?dataset=wikiann"

# Send a GET request
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
else:
    print(f"Failed to fetch data: {response.status_code}")

wikiann_languages = sorted(list({item['config'] for item in data['splits']}))


# In[44]:


random.seed(42)
results = {}

# Disable datasets load_dataset progress bar
disable_progress_bar()

for i in tqdm(range(10)):
    seed = 0
    random_language = random.choice(wikiann_languages)

    wikiann = load_dataset("wikiann", data_dir=random_language,
                           download_mode="force_redownload",
                           verification_mode="no_checks")

    wikiann_samples = sample_from_dataset(wikiann, num_samples=1, seed=seed)['train']
    
    while len(wikiann_samples['tokens'][0]) <= 2:
        seed += 1
        wikiann_samples = sample_from_dataset(wikiann, num_samples=1, seed=seed)['train']
        
    tokens = wikiann_samples['tokens'][0]

    results[random_language] = {}

    # Cut-off - at least 2 tokens at the beginning
    cut_off = int(np.ceil(len(tokens)/2))

    query_text = ' '.join(tokens[:cut_off])
    reference_text = ' '.join(tokens[cut_off:])
    print(query_text, reference_text)

    user_prompt = GUI_SENT.format(
        split_name='train',
        dataset_name='WikiAnn',
        sentence=query_text
    )

    answer = ask_gpt(user_prompt, **ASK_GPT_PARAMS)

    results[random_language]['query_text'] = query_text
    results[random_language]['reference_text'] = reference_text
    results[random_language]['answer'] = answer

results


# In[45]:


with open(os.path.join(RESULTS_PATH, f'WikiAnn_sentence_continuation.json'), 'w') as file:
    json.dump(results, file, indent=4)


# In[46]:


results = json.load(open(os.path.join(RESULTS_PATH, f'WikiAnn_sentence_continuation.json'), 'r'))


# In[47]:


match_arr = []

for id_, results_dict in tqdm(results.items(), total=10):

    user_prompt = ICL_EVAL.format(
        reference_text=results_dict['reference_text'],
        candidate_text=results_dict['answer']
    )
    answer = ask_gpt(user_prompt, openai_client, temperature=TEMPERATURE, model='gpt-4')
    results[id_]['match'] = answer

    match_arr.append(answer)

Counter(match_arr)


# ### WikiANN (English)

# In[48]:


wikiann = load_dataset("wikiann", data_dir='en',
                       download_mode="force_redownload",
                       verification_mode="no_checks")
wikiann_samples = sample_from_dataset(wikiann, num_samples=10)['train']
wikiann_samples


# In[49]:


np.random.seed(42)
results = {}

for id_, tokens in tqdm(enumerate(wikiann_samples['tokens']), total=10):
    results[id_] = {}

    # Cut-off - at least 2 tokens at the beginning
    cut_off = int(np.ceil(len(tokens)/2))

    query_text = ' '.join(tokens[:cut_off])
    reference_text = ' '.join(tokens[cut_off:])

    user_prompt = GUI_SENT.format(
        split_name='train',
        dataset_name='WikiAnn',
        sentence=query_text
    )

    answer = ask_gpt(user_prompt, **ASK_GPT_PARAMS)

    results[id_]['query_text'] = query_text
    results[id_]['reference_text'] = reference_text
    results[id_]['answer'] = answer   


# In[50]:


with open(os.path.join(RESULTS_PATH, f'WikiAnn_eng_sentence_continuation.json'), 'w') as file:
    json.dump(results, file, indent=4)


# In[51]:


results = json.load(open(os.path.join(RESULTS_PATH, f'WikiAnn_eng_sentence_continuation.json'), 'r'))


# In[52]:


for id_, results_dict in tqdm(results.items()):
    print('--> Start of the sentence:\n', results_dict['query_text'])
    print("--> Reference (ground truth):")
    print(results_dict['reference_text'])
    print("--> Model's prediction:")
    print(results_dict['answer'])
    print()


# In[53]:


match_arr = []

for id_, results_dict in tqdm(results.items(), total=10):

    user_prompt = ICL_EVAL.format(
        reference_text=results_dict['reference_text'],
        candidate_text=results_dict['answer']
    )
    answer = ask_gpt(user_prompt, openai_client, temperature=TEMPERATURE, model='gpt-4')
    results[id_]['match'] = answer

    match_arr.append(answer)

Counter(match_arr)


# ### masakhaner2

# In[54]:


# Languages that were added to the second version of the masakhaner2
masakhaner2_languages = ['bam', 'ewe', 'fon', 'twi', 'bbj', 'nya', 'tsn', 'sna', 'xho', 'zul']


# In[55]:


GUI_SENT = """INSTRUCTION:
You are provided with the FIRST PIECE of data sample tokens in {language} from the {split_name} split of the {dataset_name} dataset. Finish the SECOND PIECE of the sample tokens as EXACTLY appeared in the dataset. ONLY rely on the original form of the sample tokens in the dataset to finish the SECOND PIECE. Do not return labels, provide only tokens exactly as they appear in the dataset.

FIRST PIECE:
{sentence}

SECOND PIECE:"""


# In[56]:


random.seed(42)
results = {}


# Disable datasets load_dataset progress bar
disable_progress_bar()

for i in tqdm(range(10)):
    seed = 0
    random_language = random.choice(masakhaner2_languages)

    masaknaner2 = load_dataset(config['dataset'], random_language)

    masaknaner2_samples = sample_from_dataset(masaknaner2, num_samples=1, seed=i)['train']

    while len(masaknaner2_samples['tokens'][0]) <= 2:
        seed *= 100
        masaknaner2_samples = sample_from_dataset(masaknaner2, num_samples=1, seed=seed)['train']

    tokens = masaknaner2_samples['tokens'][0]

    results[i] = {}

    # Cut-off - at least 2 tokens at the beginning
    cut_off = int(np.ceil(len(tokens)/2))

    query_text = ' '.join(tokens[:cut_off])
    reference_text = ' '.join(tokens[cut_off:])
    print(query_text, reference_text)

    user_prompt = GUI_SENT.format(
        split_name='train',
        dataset_name='MasakhaNER 2.0',
        sentence=query_text,
        language=config['languages_names'][random_language]
    )

    answer = ask_gpt(user_prompt, **ASK_GPT_PARAMS)

    results[i]['query_text'] = query_text
    results[i]['reference_text'] = reference_text
    results[i]['answer'] = answer

results


# In[57]:


with open(os.path.join(RESULTS_PATH, f'masakhaner2_sentence_continuation.json'), 'w') as file:
    json.dump(results, file, indent=4)


# In[58]:


results = json.load(open(os.path.join(RESULTS_PATH, f'masakhaner2_sentence_continuation.json'), 'r'))


# In[59]:


for id_, results_dict in tqdm(results.items()):
    print('--> Start of the sentence:\n', results_dict['query_text'])
    print("--> Reference (ground truth):")
    print(results_dict['reference_text'])
    print("--> Model's prediction:")
    print(results_dict['answer'])
    print()


# In[60]:


match_arr = []

for id_, results_dict in tqdm(results.items(), total=10):

    user_prompt = ICL_EVAL.format(
        reference_text=results_dict['reference_text'],
        candidate_text=results_dict['answer']
    )
    answer = ask_gpt(user_prompt, openai_client, temperature=TEMPERATURE, model='gpt-4')
    results[id_]['match'] = answer

    match_arr.append(answer)

Counter(match_arr)

