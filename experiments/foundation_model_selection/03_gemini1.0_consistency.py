#!/usr/bin/env python
# coding: utf-8

# ### Setup & Initialization

# In[ ]:


import re
import os
import sys
import ast
import yaml
import json
import time

import numpy as np
from tqdm.notebook import tqdm
from datasets import load_dataset
from contextlib import contextmanager

import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel, 
    HarmCategory, 
    HarmBlockThreshold
)


# In[ ]:


SAMPLE_SIZE = 50
REPEAT_ANNOTATION = 10


# In[ ]:


# Specifying path to the necessary files and folders
PATH_TO_SRC = os.path.abspath('../../../')

# Where to get annotation examples for the prompt
ANNOTATION_EXAMPLES_PATH = os.path.join(PATH_TO_SRC, 'src/query/ner_examples_all_languages.json')
CONFIG_PATH = os.path.join(PATH_TO_SRC, "settings/config.yml")
# Folder to save annotations
RESULTS_PATH = os.path.join(PATH_TO_SRC, 'data/foundation_model_selection/gemini-1.0')


# In[ ]:


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


# Temporarily add module_path and import functions
with extend_sys_path(PATH_TO_SRC):
    from src.data.sample import sample_for_model_selection
    from src.query.query_gpt import add_annotation_examples
    from src.query.prompts import MAIN_PROMPT
    from src.utils.utils import calculate_consistency_score


# In[ ]:


# Reading config file
config = yaml.safe_load(open(os.path.join(PATH_TO_SRC, "settings/config.yml")))

# Load indx-to-label_name mapping
label_mapping = config['label_mapping']


# ### Utils

# In[ ]:


def ask_gemini_short(tokens, language, examples, user_prompt,
                    temperature, model, system_prompt=None):

    sentence = str(tokens)

    if system_prompt is None:
        system_prompt = f"You are a named entity labelling expert in {language} language."

    # Format user prompt
    user_prompt = user_prompt.format(language=language, sentence=sentence, examples=examples)

    # Save query params
    json_query_params = {
        'messages': [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": user_prompt}],
    }

    str_query_params = json.dumps(json_query_params)

    # Initialization of the model
    vertexai.init()
    gemini = GenerativeModel(model)

    # Query the model
    response = gemini.generate_content(
        str_query_params,
        generation_config={
            "temperature": temperature
        },
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )

    # Model answer
    try:
        return response.text
    except:
        print('#'*80)
        print('Model didn\'t generate any output.')
        return "{'output':[]}"


# In[ ]:


def repeat_annotation(n_repeat=10, **ask_gemini_kwargs):
    # Counters
    no_json_counter = 0  # No json was provided by the model
    incorrect_format_counter = 0  # Number of records parsed in the incorrect format 

    # Results
    ner_tokens_arr = []

    for i in tqdm(range(n_repeat)):
        # Send request to a model
        model_response = ask_gemini_short(**ask_gemini_kwargs)

        # Handle rate limitation on Gemini API
        # This is done by waiting for a second after each query
        #time.sleep(1)

        # # Extract json only
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


# ### Querying (fon)

# In[ ]:


LANGUAGE = 'fon'

language_name = config['languages_names'][LANGUAGE]
language_name


# In[ ]:


# Loading dataset from HuggingFace
data = load_dataset(config['dataset'], LANGUAGE)


# In[ ]:


sampled_subset = sample_for_model_selection(data, label_mapping, n_samples=SAMPLE_SIZE, verbose=True)
sampled_subset


# In[ ]:


ask_gemini_params = {
    'language': language_name,
    'examples': add_annotation_examples(ANNOTATION_EXAMPLES_PATH, language_name),
    'user_prompt': MAIN_PROMPT,
    'model': 'gemini-1.0-pro-vision-001',
    'temperature': config['foundation_model']['temperature']
}


# In[ ]:


gemini_annotations = {}
consistency_scores = []

# Measure how much time it takes to get all inferences
start = time.time()


for i, record in enumerate(sampled_subset):
    print(f'\nSample {i+1}:')
    
    try: 
        # Extract ground truth
        ground_truth_labels = [label_mapping[t] for t in record['ner_tags']]
        
        # Extract tokens from current record
        ask_gemini_params['tokens'] = record['tokens']

        # Query the model
        new_labels_gemini = repeat_annotation(n_repeat=REPEAT_ANNOTATION, **ask_gemini_params)
        
        # Save annotations
        gemini_annotations[f'record_{i}'] = {}
        gemini_annotations[f'record_{i}']['pred'] = new_labels_gemini
        gemini_annotations[f'record_{i}']['true'] = ground_truth_labels
        gemini_annotations[f'record_{i}']['tokens'] = record['tokens']
        
        # Calculate consistency score
        consistency = calculate_consistency_score(new_labels_gemini, ground_truth_labels)
        gemini_annotations[f'record_{i}']['consistency'] = consistency
        consistency_scores.append(consistency)
        
    except Exception as e:
        print(e)
        continue

end = time.time()
print('Execution time: ', end - start, 's')  


# In[ ]:


# Overall consistency is calculated by averaging individual scores
gemini_annotations['overall_consistency'] = np.mean(consistency_scores)


# In[ ]:


print("Number of samples that model didn't generate any output for them: ", consistency_scores.count(0))

print("Overall consistency on all samples: ", gemini_annotations['overall_consistency'])


# In[ ]:


with open(os.path.join(RESULTS_PATH, f'{LANGUAGE}.json'), 'w') as file:
    json.dump(gemini_annotations, file, indent=4)

