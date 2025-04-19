#!/usr/bin/env python
# coding: utf-8

# ### Setup & Initialization

# In[1]:


import os
import sys
import yaml
from contextlib import contextmanager
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.notebook import tqdm
from openai import APIStatusError, RateLimitError, APIConnectionError
import json
import ast
import re
import numpy as np
import time


# In[2]:


SAMPLE_SIZE = 50
REPEAT_ANNOTATION = 10


# In[3]:


# Specifying path to the necessary files and folders
PATH_TO_SRC = os.path.abspath('../../../')

# Where to get annotation examples for the prompt
ANNOTATION_EXAMPLES_PATH = os.path.join(PATH_TO_SRC, 'src/query/ner_examples_all_languages.json')
CONFIG_PATH = os.path.join(PATH_TO_SRC, "settings/config.yml")
ENV_FILE_PATH = os.path.join(PATH_TO_SRC, '.env')
# Folder to save annotations
RESULTS_PATH = os.path.join(PATH_TO_SRC, 'data/foundation_model_selection/starling-lm-0.1-temp')


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
    from src.data.sample import sample_for_model_selection
    from src.query.query_gpt import ask_gpt, add_annotation_examples
    from src.query.prompts import MAIN_PROMPT
    from src.utils.utils import calculate_consistency_score


# In[6]:


# Load environment variables
print(load_dotenv(dotenv_path=ENV_FILE_PATH))

openai_client = OpenAI(
    base_url = 'http://localhost:11434/v1', # run with ollama api
    api_key="ollama" #os.environ['OPENAI_API_KEY'],
)


# In[7]:


# Reading config file
config = yaml.safe_load(open(os.path.join(PATH_TO_SRC, "settings/config.yml")))

# Load indx-to-label_name mapping
label_mapping = config['label_mapping']
label_mapping


# ### Utils

# In[8]:


def ask_gpt_short(
        tokens, language, examples, openai_client, user_prompt,
        max_tokens=1000,
        temperature=0.1,
        model='starling-lm',
        system_prompt=None):

    sentence = str(tokens)
    ner_tags = None

    if system_prompt is None:
        system_prompt = f"You are a named entity labelling expert in {language} language."

    # Format user prompt
    user_prompt = user_prompt.format(language=language, sentence=sentence, examples=examples)

    # Save query params
    query_params = {
        'model': model,
        'temperature': temperature,
        'messages': [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
        'max_tokens': max_tokens,
    }

    if model in ['gpt-4-1106-preview' , 'gpt-4-0125-preview',"llama2:70b","starling-lm"]:  
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


# In[9]:


def repeat_annotation(n_repeat=10, **ask_gpt_kwargs):
    # Counters
    no_json_counter = 0  # No json was provided by the model
    incorrect_format_counter = 0  # Number of records parsed in the incorrect format 

    # Results
    ner_tokens_arr = []

    for i in tqdm(range(n_repeat)):
        # Send request to a model
        model_response = ask_gpt_short(**ask_gpt_kwargs)

        if ask_gpt_kwargs['model'] in ['gpt-4-1106-preview', 'gpt-4-0125-preview',"llama2:70b","starling-lm"]:
            # Newer models provide json
            try:
                ner_tags = json.loads(model_response)['output']
                ner_tokens_arr.append(ner_tags)
            except Exception as e:
                print('#'*80)
                print(e)
                print(model_response)
                incorrect_format_counter += 1
                continue
        else:
            # Extract json only
            match = re.search(r'\{(.*?)\}', model_response)
            if match:
                content = match.group(0)
                # Format output string to parse it as JSON
                try:
                    ner_tags = json.loads(json.dumps(ast.literal_eval(content)))['output']
                    ner_tokens_arr.append(ner_tags)
                except Exception as e:
                    print('#'*80)
                    print(e)
                    print(model_response)
                    incorrect_format_counter += 1
                    continue
            else:
                print('#'*80)
                print('No json found in model\'s response:', model_response)
                no_json_counter += 1
                continue
    print(f'Number of model responses without json: {no_json_counter}')
    print(f'Number of model responses with incorrect formatting: {incorrect_format_counter}')
    return ner_tokens_arr


# In[10]:


target_languages= ['zul','tsn','bam']


# ### Querying all

# In[11]:


for LANGUAGE in target_languages:
    language_name = config['languages_names'][LANGUAGE]
    print("Language:",language_name)
    # Loading dataset from HuggingFace
    data = load_dataset(config['dataset'], LANGUAGE)
    sampled_subset = sample_for_model_selection(data, label_mapping, n_samples=SAMPLE_SIZE, verbose=True)
    sampled_subset
    ask_gpt_params = {
        'language': language_name,
        'examples': add_annotation_examples(ANNOTATION_EXAMPLES_PATH, language_name),
        'openai_client': openai_client,
        'user_prompt': MAIN_PROMPT,
        'model': 'starling-lm', # 'llama2',
        'temperature': 0.1 #config['foundation_model']['temperature']
    }
    gpt_annotations = {}
    consistency_scores = []

    # Measure how much time it takes to get all inferences
    start = time.time()


    for i, record in enumerate(sampled_subset):
        print(f'\nIteration: {i}')
        
        try: 
            # Extract ground truth
            ground_truth_labels = [label_mapping[t] for t in record['ner_tags']]
            
            # Extract tokens from current record
            ask_gpt_params['tokens'] = record['tokens']
            # Query the model
            new_labels_gpt4 = repeat_annotation(n_repeat=REPEAT_ANNOTATION, **ask_gpt_params)
            # Save annotations
            gpt_annotations[f'record_{i}'] = {}
            gpt_annotations[f'record_{i}']['pred'] = new_labels_gpt4
            gpt_annotations[f'record_{i}']['true'] = ground_truth_labels
            gpt_annotations[f'record_{i}']['tokens'] = record['tokens']
            
            # Calculate consistency score
            consistency = calculate_consistency_score(new_labels_gpt4, ground_truth_labels)
            gpt_annotations[f'record_{i}']['consistency'] = consistency
            consistency_scores.append(consistency)
            
        except Exception as e:
            print(e)
            continue

    # Overall consistency is calculated by averaging individual scores
    gpt_annotations['overall_consistency'] = np.mean(consistency_scores)

    end = time.time()
    print('Execution time: ', end - start, 's')
    with open(os.path.join(RESULTS_PATH, f'{LANGUAGE}.json'), 'w') as file:
        json.dump(gpt_annotations, file, indent=4)

