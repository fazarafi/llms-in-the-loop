#!/usr/bin/env python
# coding: utf-8

# This notebook tests and showcases usage of the custom `ask_gpt` function that is used to query foundation model for NER tags.

# In[2]:


import ast
from openai import OpenAI, APIStatusError, RateLimitError, APIConnectionError
from dotenv import load_dotenv
from datasets import load_dataset
import json
import os
import sys
import yaml
import regex as re

queries_module_path = os.path.abspath('../src/query')
if queries_module_path not in sys.path:
    sys.path.append(queries_module_path)

from prompts import MAIN_PROMPT


# In[3]:


print(MAIN_PROMPT)


# In[4]:


# Reading config file
config = yaml.safe_load(open("../settings/config.yml"))


# In[5]:


# Load environment variables
# ! You should create .env file 
load_dotenv(dotenv_path=os.path.join('..', '.env'))


# In[6]:


# Initialising openai client
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
)


# In[7]:


# Loading data and selecting test sentence
test_language = 'bam'
data = load_dataset(config['dataset'], test_language)

language_full_name = config['languages_names'][test_language]
language_full_name


# In[8]:


test_sentence = data['train'][5]['tokens']
print(test_sentence)


# In[13]:


def ask_gpt(tokens, language, examples, openai_client, user_prompt, max_tokens=1000,
            temperature=0.7, model='gpt-4-1106-preview', system_prompt=None):
    """
    Generate named entity tags for a given sentence using the specified GPT model.

    Parameters:
    - tokens (str or list): list of tokens for which named entity recognition is desired.
    - language (str): The language of the input sentence.
    - openai_client (OpenAI API client object): The OpenAI client.
    - user_prompt (str, optional): Custom user prompt for the GPT model.
    - temperature (float, optional): A value between 0 and 1 that controls the randomness of the response.
      Lower values make the model more deterministic. Default is 0.3.
    - model (str, optional): The identifier of the GPT model to be used. Default is 'gpt-4-1106-preview'.
    - system_prompt (str, optional): Custom system prompt for the GPT model. If None, a default prompt is used.

    Returns:
    - ner_tags (list): A list of named entity tags generated for each token in the input sentence.
    - content (str): Text of the model response
    """
    # Convert token list to string
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

    if model == 'gpt-4-1106-preview' or model == 'gpt-3.5-turbo-1106':  # Add additional params for new model
        query_params['response_format'] = {"type": "json_object"}

    try:
        # Query the model
        response = openai_client.chat.completions.create(**query_params)
    except APIConnectionError as e:
        raise Exception(f"The server could not be reached: {e.__cause__}")
    except RateLimitError as e:
        raise Exception(f"A 429 status code was received: {e}")
    except APIStatusError as e:
        raise Exception(f"Non-200-range status code received: {e.status_code}, {e.response}")

    try:
        # Extract NER tags from the response
        content = response.choices[0].message.content

        if model == 'gpt-4-1106-preview' or model == 'gpt-3.5-turbo-1106':
            # Newer models provide json
            ner_tags = json.loads(content)['output']
        else:
            # Extract json only 
            match = re.search(r'\{(.*?)\}', content)
            
            if match:
                content = match.group(0) 
                # Format output string to parse it as JSON
                ner_tags = json.loads(json.dumps(ast.literal_eval(content)))['output']
            else:
                raise ValueError("No json found in model's response.")

    except Exception as e:
        print(response.choices[0].message.content)
        raise Exception(f"Cannot extract output from model's response: {e}")
        
    return ner_tags, content


# In[10]:


def add_annotation_examples(json_filepath, language):
    """
    Create formatted string containing annotation examples in specified language.
    """
    with open(json_filepath, 'r') as json_file:
        examples = json.load(json_file)[language]
        
    example_str = f"""Example 1:
Input: {examples['example1']['input']}
Output: {{ 'output': {examples['example1']['output']} }}
Example 2:
Input: {examples['example1']['input']}
Output: {{ 'output': {examples['example2']['output']} }}"""
    
    return example_str
    
    
bam_examples = add_annotation_examples('../src/query/annotation_examples.json', 'Bambara')
print(bam_examples)


# ### GPT-4 Turbo

# In[17]:


result, response = ask_gpt(
    tokens=test_sentence, 
    language=language_full_name, 
    examples=bam_examples,
    openai_client=client, 
    user_prompt=MAIN_PROMPT
)

print('Tokens provided by LLM:')
print([t[1] for t in result])


# In[18]:


true_tokens = data['train'][5]['ner_tags']
print('True tokens:')
print([config['label_mapping'][l] for l in true_tokens])


# In[19]:


print(len(test_sentence), len(result))


# ### GPT-4

# In[20]:


result, response = ask_gpt(
    tokens=test_sentence, 
    language=language_full_name, 
    openai_client=client, 
    model='gpt-4', 
    user_prompt=MAIN_PROMPT,
    examples=bam_examples
)

print('Tokens provided by LLM:')
print([t[1] for t in result])


# Approximate translation of the sentence:
# "Abudarahamani Sisoko / Maliweb.net Kasɔrɔ presidential candidate, for the first time, has announced his intention to run, as of February 2022, with significant support and a strong campaign plan."
# 
# "Abudarahamani Sisoko" is marked as a person's name (B-PER, I-PER).
# "Maliweb.net" is annotated as an organization (B-ORG, I-ORG, I-ORG).
# "feburuyekalo san 2022" is identified as a date (B-DATE, I-DATE, I-DATE).
# All other tokens are labeled as "O" since they do not represent named entities.

# **In the masakhaner2 “Maliweb.net” is not recognized as entity,
# however, the foundation model identified it as entity** due to the following reasons:

# The annotation of "Maliweb.net" as an organization in the named entity recognition (NER) task is based on the context and the structure of the term. Here's the rationale:
# 
# 1. Domain Name Suggesting an Organization: "Maliweb.net" appears to be a domain name, typically associated with a website. Websites are often representative of organizations, companies, or entities rather than individuals. The ".net" suffix is commonly used by organizations, especially those involved in technology, internet-based services, or networks.
# 
# 2. Common NER Practices: In NER tasks, entities like websites, companies, or other groups are usually classified as organizations.
# 
# 3. Lack of Contextual Clues for Other Entity Types: Without specific contextual clues that "Maliweb.net" refers to something other than an organization (like a person, location, or date), the default assumption based on its structure as a web domain is to classify it as an organization.

# Upon detailed investigation, we discovered that MUC-6 (Message Understanding Conference-6) Named Entity Recognition (NER) annotation guidelines that were used by the masakhaner2 annotators,  do not contain any guidelines on annotating websites. 
