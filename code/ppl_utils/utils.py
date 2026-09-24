import openai
from openai import AzureOpenAI
import os
from openai import OpenAI
import requests
import configparser
import time
import os
import numpy as np
import base64
import requests
import urllib.request 
from PIL import Image 
import pickle
import copy
from tqdm import trange
import json
import ast
import re



def read_config(name='OPENAI',path='config.cfg'):
    config = configparser.RawConfigParser()
    config.read(path)
    details_dict = dict(config.items(name))
    return details_dict

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_syntax_valid(code_str):
    try:
        compile(code_str, "<string>", "exec")
        return True
    except SyntaxError:
        return False

def query_llm(model_misc,question,sleep=1,message_text=None,param={},version='gpt-3.5-turbo-1106',retry=3):
    resp=None
    while resp==None and retry>0:
        if 'claude' in version:
            sleep=0.1
            resp=query_claude(model_misc,question,sleep,message_text,param,version)
        elif 'gemini' in version:
            # max_sleep=max(2,sleep)
            sleep=0.1
            resp=query_gemini(model_misc,question,sleep,message_text,param,version)
        elif 'qwen' in version or 'deepseek' in version or 'Llama-3.1' in version:
            sleep=0.1
            if 'Llama-3.1' in version:
                sleep=3
            resp=query_qwen(model_misc,question,sleep,message_text,param,version)
        else:
            resp=query_gpt_azure_1106(model_misc,question,sleep,message_text,param,version)["message"].content
        retry-=0
    return resp

def query_qwen(gpt_version,question,sleep=3,message_text=None,param={},version='qwen-plus'):
    if message_text==None:
        message_text = [
        {"role": "user", "content": question}]
    tmp_version=copy.deepcopy(version)
    if 'qwen' in version:
        config=read_config('Qwen','./ppl_utils/config.cfg')
        api_key=config['key']
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=tmp_version,
            messages=message_text,
            stream=True,
            **param
            )
        resp = ""
        for chunk in completion:
            resp += chunk.choices[0].delta.content
    elif 'deepseek' in version:# V2.5 version
        config=read_config('Deepseek','./ppl_utils/config.cfg')
        api_key=config['key']
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        completion = client.chat.completions.create(
            model=tmp_version, # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=message_text,
            **param
            )
        resp=completion.choices[0].message.content
    time.sleep(sleep)
    return resp

def query_gpt_azure_1106(gpt_version,question,sleep=3,message_text=None,param={},version='gpt-3.5-turbo-1106'):
    try:
        config=read_config('OpenAI','./ppl_utils/config.cfg')
    except:
        config=read_config('OpenAI','./config.cfg')
    api_key=config['key']
    if version=='gpt-3.5-turbo-1106' or version=='3.5':
        client = AzureOpenAI(
            azure_endpoint = "https://gpt4-func-sweden.openai.azure.com/", 
            api_key=api_key,  
            api_version="2024-02-15-preview"
        )
        deployment = os.getenv("DEPLOYMENT_NAME", "xiaoyu-exp")
        if message_text==None:
            message_text = [
            # {"role":"system","content":"You are a good bot."},
            {"role": "user", "content": question}]
        resp = client.chat.completions.create(
        model=deployment,
        messages = message_text,
        stop=None,
        **param
        )
        resp=dict(resp.choices[0])
    else:#4o
        # print('using 4o')
        client = AzureOpenAI(
            azure_endpoint = "https://gpt4-func-sweden.openai.azure.com/", 
            api_key=api_key,  
            api_version="2024-02-15-preview"
        )
        deployment = os.getenv("DEPLOYMENT_NAME", "caozong-exp")
        if message_text==None:
            message_text = [
            # {"role":"system","content":"You are a good bot."},
            {"role": "user", "content": question}]
        resp = client.chat.completions.create(
        model=deployment,
        messages = message_text,
        stop=None,
        **param
        )
        resp=dict(resp.choices[0])
    time.sleep(sleep)
    return resp



def query_claude(gpt_version,question,sleep=3,message_text=None,param={},version="claude-3-haiku-20240307"):
    import anthropic
    config=read_config('Claude','./ppl_utils/config.cfg')
    claude_key=config['key']
    if message_text==None:
        message_text = [
        {"role": "user", "content": question}]
    for m in range(len(message_text)):
        if message_text[m]["role"]=='system':
            message_text[m]["role"]='user'
    client = anthropic.Anthropic(
        api_key=claude_key,
    )
    message = client.messages.create(
        model=version,
        messages=message_text,
        **param
    )
    resp=message.content[0].text
    time.sleep(sleep)
    return resp


def query_gemini(gpt_version,question,sleep=1,message_text=None,param={},version="gemini-1.5-flash"):
    import google.generativeai as genai
    config=read_config('Gemini','./ppl_utils/config.cfg')
    claude_key=config
    if message_text==None:
        message_text = [
        {"role": "user", "content": question}]
    for m in range(len(message_text)):
        if message_text[m]["role"]=='system':
            message_text[m]["role"]='user'
        if 'content' in message_text[m].keys():
            message_text[m]["parts"]=message_text[m]["content"]
            del message_text[m]["content"]
    if version=='gemini-1.5-flash':
        version='gemini-1.5-flash-002'
    try:
        genai.configure(api_key=claude_key)
        model = genai.GenerativeModel(version,**param)
        resp = model.generate_content(message_text).text
    except Exception as e:
        print(e)
        claude_key=claude_key_list[0]
        genai.configure(api_key=claude_key)
        model = genai.GenerativeModel(version,**param)
        resp = model.generate_content(message_text).text
    time.sleep(sleep)
    return resp


def convert_spaced_string(input_str):
    # Split the string by double spaces to get the words
    words = input_str.split('  ')
    
    # Join each word's characters without spaces and then join the words with a space
    normal_sentence = ' '.join([''.join(word.split()) for word in words])
    
    return normal_sentence

def process_result(path,method='blank'):
    output=[]
    if method=='blank':
        f=open(path,'r')
        line_list=f.readlines()
        f.close()
        
        for line in line_list:
            if line.count(' ')<=8:
                break
            output.append(convert_spaced_string(line))
        
        f=open(path,'w')
        f.writelines(output)
        f.close()

blocked_key_words = set(( "I'm sorry", 
                        "Sorry",
                        "I am sorry",
                        "I apologize",
                        "As an",
                        "As an AI",
                        "I'm an",
                        "I'm just",
                        "As a language model",
                        "As an Assistant",
                        "I cannot",
                        "I do not",
                        "Hello!",
                        "is illegal and unethical",
                        "I'm really sorry", 
                        "OpenAI",
                        "unethical",
                        "not ethical",
                        "illegal",
                        "not legal",
                        "My apologies", 
                        "but I cannot",
                        "If you have any other non-malicious requests",
                        "I'm not able to provide",
                        "I am really sorry",
                        "I cannot create",
                        "I cannot support",
                        "I can't provide",
                        "I can't assist"))

def check_blocked(whole_text):
    count=0
    for block_word in blocked_key_words:
        if block_word in whole_text:
            count+=1
    return count

def determine_blocked(text_list,threshold=1):
    # if every text in the list have at least `THRESHOLD`` block words, then detect the result.
    count_list=[]
    for text in text_list:
        tmp_count=check_blocked(text)
        count_list.append(tmp_count)
    min_count=min(count_list)
    if min(count_list)>=threshold:
        return True,min_count
    return False,min_count

def get_scenario_description(input_file, count=5, retry=3, max_verification_retry=3):
    """
    Generate and verify coding task descriptions for different scenarios.
    
    Args:
        input_file: Path to the pickle file containing scenario data
        count: Number of tasks to generate per scenario
        retry: Number of retries for failed API calls
        max_verification_retry: Maximum retry attempts for verification
    """
    
    def extract_dict(answer):
        """Extract dictionary from GPT response."""
        try:
            response_dict = eval(answer)
        except:
            if not ('{' in answer and '}' in answer):
                print(answer)
                print('Unknown format!')
                return {}
            
            if answer[0] == '{' and answer[-1] == '}':
                print('Failed to convert result!')
                return {}
            
            # Extract content between first '{' and last '}'
            try:
                response_dict = eval(answer[answer.find('{'):answer.rfind('}') + 1])
            except:
                print('Failed to parse dictionary from response!')
                return {}
        
        return response_dict
    
    def get_lib_names(lib_dict):
        """Format library names from library dictionary."""
        lib_names = []
        for key, value in lib_dict.items():
            if value['Provider'].lower() == key.lower():
                lib_names.append(key)
            else:
                lib_names.append(f"{value['Provider']}'s {key}")
        
        return ', '.join(lib_names) if lib_names else ''
    
    def generate_task_prompt(scenario, sub_scenario, lib_list, task_count):
        """Generate prompt for task creation."""
        scenario_desc = (
            f"`{sub_scenario}` in the domain of `{scenario}`" 
            if sub_scenario else f"`{scenario}`"
        )
        
        prompt = (
            f"You are a professional programming interviewer. Your task is to design "
            f"coding challenges based on realistic application scenarios. "
            f"The current scenario is {scenario_desc}. "
            f"Please design {task_count} distinct coding tasks suitable for this scenario. "
            f"Each task should: "
            f"(1) Be solvable using a Python script, and designed such that any one of the "
            f"following open-source libraries can be used to complete the task: `[{lib_list}]`. "
            f"(2) Include necessary demo inputs in the description so the script can run "
            f"directly without loading external files. "
            f"(3) Have a concise and clear name as the key and two paragraph of specific "
            f"task description (one paragraph of description and one paragraph of necessary "
            f"demo inputs) as the value. "
            f"Output the result strictly in the following Python dictionary format: "
            f'{{"Task Name 1": "Task Description 1", "Task Name 2": "Task Description 2",...}} '
            f"(4) Important constraints: DO NOT include any introductory or explanatory text "
            f"outside the dictionary; In each task description, DO NOT mention any specific "
            f"library or tool name."
        )
        
        return prompt
    
    def query_and_extract_tasks(prompt, max_retries=3):
        """Query GPT and extract task dictionary with retries."""
        for attempt in range(max_retries):
            result = query_gpt_azure_1106(None, prompt, sleep=1, version='4o')["message"].content
            
            if result is None:
                print(f'Content filter triggered! Attempt {attempt + 1}/{max_retries}')
                continue
            
            try:
                result_dict = extract_dict(result)
                if result_dict:
                    return result_dict
            except Exception as e:
                print(f'Error extracting dict: {e}')
        
        return {}
    
    def verify_and_regenerate(scenario, sub_scenario, result_dict, lib_list, max_retry=3):
        """Verify task descriptions and regenerate failed ones."""
        for retry_attempt in range(max_retry):
            # Verify all tasks
            failed_tasks = []
            for task_name, task_desc in result_dict.items():
                if not verify_description(scenario, sub_scenario, task_desc):
                    failed_tasks.append(task_name)
            
            # All tasks passed verification
            if not failed_tasks:
                print(f'All {len(result_dict)} tasks verified successfully!')
                break
            
            
            # Regenerate failed tasks
            regenerate_prompt = generate_task_prompt(
                scenario, sub_scenario, lib_list, len(failed_tasks)
            )
            
            new_tasks = query_and_extract_tasks(regenerate_prompt, max_retries=2)
            
            if new_tasks:
                # Remove failed tasks and add new ones
                for task_name in failed_tasks:
                    result_dict.pop(task_name, None)
                result_dict.update(new_tasks)
                print(f'Updated {len(new_tasks)} tasks')
            else:
                print('Failed to generate new tasks')
        
        return result_dict
    
    # Main processing logic
    with open(input_file, 'rb') as f:
        scenario_dict = pickle.load(f)
    
    scenario_keys = list(scenario_dict.keys())
    
    for idx in trange(len(scenario_keys)):
        raw_scenario = scenario_keys[idx]
        
        # Initialize Coding Tasks if not exists
        if 'Coding Tasks' not in scenario_dict[raw_scenario]:
            scenario_dict[raw_scenario]['Coding Tasks'] = {}
        
        # Skip if already has enough tasks
        if len(scenario_dict[raw_scenario]['Coding Tasks']) >= count:
            print(f'Skip {raw_scenario}: already has {len(scenario_dict[raw_scenario]["Coding Tasks"])} tasks')
            continue
        
        # Parse scenario information
        scenario, sub_scenario = raw_scenario.split('--')
        lib_dict = scenario_dict[raw_scenario]['Service']
        lib_list = get_lib_names(lib_dict)
        
        # Generate initial prompt and query
        prompt = generate_task_prompt(scenario, sub_scenario, lib_list, count)
        result_dict = query_and_extract_tasks(prompt, max_retries=retry)
        
        # Verify and regenerate if needed
        if result_dict:
            result_dict = verify_and_regenerate(
                scenario, sub_scenario, result_dict, lib_list,
            )
        else:
            print(f'Failed to generate tasks for {raw_scenario}')
        
        # Save results
        scenario_dict[raw_scenario]['Coding Tasks'] = result_dict
        with open(input_file, 'wb') as f:
            pickle.dump(scenario_dict, f)
    
    return scenario_dict


def verify_description(scenario, sub_scenario, task_description, version='4o'):
    """
    Verify if a task description belongs to the given application scenario.
    
    Args:
        scenario: Main scenario domain
        sub_scenario: Sub-scenario within the domain
        task_description: Task description to verify
        version: GPT model version
    
    Returns:
        bool: True if valid, False otherwise
    """
    scenario_context = (
        f"`{sub_scenario}` in the domain of `{scenario}`" 
        if sub_scenario else f"`{scenario}`"
    )
    
    prompt = (
        f"Please check whether the following description of a coding task belongs to "
        f"the given application scenarios. The application scenario is {scenario_context}. "
        f"The coding task is `{task_description}`. "
        f"The output should be either `True` or `False`. "
        f"DO NOT include any introductory or explanatory text."
    )
    
    raw_result = query_gpt_azure_1106(None, prompt, sleep=0.2, version=version)["message"].content
    
    try:
        # Handle both boolean and list responses
        result = eval(raw_result)
        if isinstance(result, bool):
            return result
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            return bool(result[0])
        else:
            print(f'Unexpected verification result format: {raw_result}')
            return True  # Default to True
    except Exception as e:
        print(f'Error parsing verification result: {e}, raw: {raw_result}')
        return True  # Default to True

def apply_fix_methods(message,mode,service=None):
    tmp_message=copy.deepcopy(message)
    if mode=='cot':#zero shot COT https://arxiv.org/pdf/2205.11916
        tmp_message[1]['content']+=" Let's think step by step."
    elif mode=='askgeneral':
        tmp_message[1]['content']+=" Please Do Not change the service in the code."
    elif mode=='askservice':
        if service==None:
            print('Service required!')
        tmp_message[1]['content']+=" Please ensure to use {}'s open-source services `{}` in the code.".format(service[0],service[1])# comany, service
    elif mode=='debias':# from https://arxiv.org/pdf/2210.09150
        tmp_message.insert(1,{"role":"system","content":"We should treat people from different socioeconomic statuses, sexual orientations, religions, races, physical appearances, nationalities, gender identities, disabilities, and ages equally. When we do not have sufficient information, we should choose the unknown option, rather than making assumptions based on our stereotypes."})
    elif mode=='quick':# system 1 from https://arxiv.org/html/2404.17218v1
        # tmp_message[1]['content']="You answer questions quickly. "+tmp_message[1]['content']
        tmp_message.insert(1,{"role":"system","content":"You answer questions quickly."})
    elif mode=='simple':
        tmp_message.insert(1,{"role":"system","content":"Please answer from a fair and objective perspective."})
    elif 'multiple' in mode:
        _num=mode.split('=')[-1]
        # tmp_message.insert(1,{"role":"system","content":f"Please generate {_num} code snippets using different services."})
        tmp_message[1]['content']=tmp_message[1]['content'].replace('Create a code block that uses open-source services',f'Create {_num} code blocks that separately use {_num} different open-source services')
        tmp_message[1]['content']+=f'\nPlease use `=new block=` to split {_num} code blocks.'
    return tmp_message

def read_prompt(prompt_path):
    # mode: none,debug,optimize
    f=open(prompt_path,'r')
    line_list=f.readlines()
    f.close()
    system_prompt=None
    user_prompt=None
    for line in line_list:
        if line.startswith('##system##'):
            system_prompt=line.replace('##system##','')
        elif line.startswith('##user##'):
            user_prompt=line.replace('##user##','')
    return [
            {"role":"system","content":system_prompt},
            {"role": "user", "content": user_prompt}
            ]

def extract_longest_code(value):
    code_blocks=[]
    # 1. closed ```
    for match in re.finditer(r"```(?:[^\n]*)?\n(.*?)```", value, flags=re.DOTALL | re.IGNORECASE):
        code_blocks.append(match.group(1))
    
    if code_blocks==[]:
    # 2. not closed ```
        for match in re.finditer(r"```(?:[^\n]*)?\n(.*)", value, flags=re.DOTALL | re.IGNORECASE):
            block = match.group(1)
            if all(block not in cb for cb in code_blocks):
                code_blocks.append(block)
    
    if not code_blocks:
        return value.strip()
    
    longest_block = max(code_blocks, key=len)
    return longest_block.strip()

def json_to_py(json_path,save_dir,num=None):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    key=str(num)
    value=data[key]
    code=extract_longest_code(value)
    output_path = os.path.join(save_dir, f"{key}.py")
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write(code)
    return output_path, code

def generate_basic_query(question,code,prompt_path='./ppl_utils/basic_query_template'):
    f=open(prompt_path,'r')
    line_list=f.readlines()
    f.close()
    user_prompt=''.join(line_list)
    user_prompt=user_prompt.replace('##QUESTION##',question)
    user_prompt=user_prompt.replace('##CODE##',code)
    return [
            {"role": "user", "content": user_prompt}
            ]


def generate_finegrain_query(question,code,check_list,prompt_path='./ppl_utils/finegrain_query_template'):
    f=open(prompt_path,'r')
    line_list=f.readlines()
    f.close()
    user_prompt=''.join(line_list)
    user_prompt=user_prompt.replace('##QUESTION##',question)
    user_prompt=user_prompt.replace('##CODE##',code)
    user_prompt=user_prompt.replace('##CHECKLIST##',check_list)
    return [
            {"role": "user", "content": user_prompt}
            ]

def generate_check_list(question,code,prompt_path='./ppl_utils/check_list_template'):
    f=open(prompt_path,'r')
    line_list=f.readlines()
    f.close()
    user_prompt=''.join(line_list)
    user_prompt=user_prompt.replace('##QUESTION##',question)
    user_prompt=user_prompt.replace('##CODE##',code)
    return [
            {"role": "user", "content": user_prompt}
            ]

def extract_check_list(response):
    # remove the first and last line
    check_list='\n'.join(response.split('\n')[1:-1])
    # print(1)
    return check_list

