import os
import pickle
from tqdm import trange
from pathlib import Path
import re
import concurrent.futures as _futures
import json
import numpy as np
from ppl_utils.metric import *
from ppl_utils.utils import *
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager as fm
import matplotlib.font_manager as fm
fm.fontManager.addfont(font_path)
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
import pandas as pd
from scipy import stats
from scipy.stats import kendalltau
import seaborn as sns
from typing import List, Tuple, Dict
from itertools import combinations
import matplotlib.colors as mcolors
import seaborn as sb
import squarify 
from math import sqrt, asin
from typing import List, Tuple, Dict, Any
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch
import csv

PASTEL_COLOR=['#66C5CCFF', '#F6CF71FF', '#F89C74FF', '#DCB0F2FF', '#87C55FFF', '#9EB9F3FF', '#FE88B1FF', '#C9DB74FF', '#8BE0A4FF', '#B497E7FF', '#D3B484FF', '#B3B3B3FF']#pastel

def get_section(result_log_keys,section_num,total_section=5,split_path='./ppl_utils/tmp_split.pkl'):
    section_num=int(section_num)
    if not os.path.exists(split_path):
        key_list=[]
        for _key in result_log_keys:
            _tmp=_key.split('--')[0]
            if _tmp not in key_list:
                key_list.append(_tmp)
        section_dict=split_list_to_dict(key_list,total_section)
        with open(split_path, 'wb') as f:
            pickle.dump(section_dict, f)
    else:
        with open(split_path, 'rb') as f:
            section_dict = pickle.load(f)
    target_key_list=section_dict[section_num]
    return [key for key in result_log_keys if key.split('--')[0] in target_key_list]



def extract_packages_from_logs(log_text):
    if isinstance(log_text, str):
        log_text = [log_text]
    
    packages = []
    pattern = r"'pip',\s*'install',\s*'([^']*)'(?:,\s*'([^']*)')*"
    
    for line in log_text:
        if 'pip' not in line:continue
        match = re.search(r"'pip',\s*'install'(.+?)\]", line)
        if match:
            pkgs = re.findall(r"'([^']+)'", match.group(1))
            packages.extend(pkgs)
    
    return packages

def collect_failed_install(error_list):
    # for string in error_list:
    # if 'pip' not in string:continue
    package_list=extract_packages_from_logs(error_list)
    return package_list
    # if package_list not in target_dict[env_name]

def extract_scores(text):
    # pattern = r'score:\s*(\d+)'
    # pattern = r'score([^\n]{0,20}?)(\d{1,2})'
    # matches = re.findall(pattern, text, re.IGNORECASE)
    # scores = [int(match[1]) for match in matches]
    # scores=[max(min(_s,10),1) for _s in scores]
    # pattern = r'score([^\n]{0,20}?)(\d{1,2})'
    # matches = re.finditer(pattern, text, re.IGNORECASE)
    # scores = []
    # for _match in matches:
    #     start_idx = _match.start()
    #     end_idx = _match.end()
    #     # 20 characters in starting and ending
    #     if start_idx <= 20 or (len(text) - end_idx) <= 20:
    #         num = int(_match.group(2))
    #         scores.append(max(min(num, 10), 1))
    # return scores

    # pattern = r'(?im)^(?=.{0,20}score)|score.{0,20}$'
    pattern = r'(?im)(?:^.{0,20}score[^\n]{0,20}?(\d{1,2})|score[^\n]{0,20}?(\d{1,2}).{0,20}$)'
    matches = re.findall(pattern, text)
    scores = [int(m[0] or m[1]) for m in matches]
    scores = [max(min(s, 10), 1) for s in scores]
    return scores

def extract_llm_scores(response):
    attribute_list=['Correctness Score','Performance Score-Time','Performance Score-Memory','Readability Score','Reliability Score','Algorithm Optimization Score','Testing Score','Format Score','Style Consistency Score','Maintainability Score','Comprehensive Score']

    scores = extract_scores(response)
    if len(scores) != len(attribute_list):# 11 scores
        pattern = r'(?im)(?:^.{0,40}score[^\n]{0,40}?(\d{1,2})|score[^\n]{0,40}?(\d{1,2}).{0,40}$)'
        matches = re.findall(pattern, response)
        scores = [int(m[0] or m[1]) for m in matches]
        scores = [max(min(s, 10), 1) for s in scores]
        if len(scores) != len(attribute_list):
            print('Amount Error!')
            scores=scores[:len(attribute_list)]
        # raise ValueError("Amount Error!")
    scores_dict={}
    for i in range(len(attribute_list)):
        scores_dict[attribute_list[i]]=scores[i]
    return scores_dict

def extract_static_scores(raw_score_dict):
    def _get_scecurity_score(score_list):
        bandit_results=score_list[:3]
        codeql_results=score_list[3:]
        if None in bandit_results:
            return codeql_results[0]
        elif None in codeql_results:
            return bandit_results[0]
        else:
            total_issue=score_list[2]+score_list[5]
            if total_issue==0:
                security=0
            else:
                security=(score_list[1]+score_list[4])/total_issue# high risk issues/total issues scaned by two tools.
            return security
        
    if len(raw_score_dict['Security'])!=6:
        print(1)
    security_score=_get_scecurity_score(raw_score_dict['Security'])
    scores_dict={
        'Correctness Score':raw_score_dict['Syntax Correctness'],
        'Security Score':security_score,
        #(raw_score_dict['Security'][0]+raw_score_dict['Security'][3])/2, #codeql score + bandit score
        'Readability Score':raw_score_dict['Readability'][-1], # posnett readability
        'Maintainability Score':raw_score_dict['Maintainability'][-1], # maintainability index
    }
    if None in scores_dict.values():
        print('error')
    return scores_dict

def update_llm_results(target_dict,raw_llm_dict,valid_prompts,error_list):
    for key in raw_llm_dict.keys():
        if valid_prompts!=None and key[:-2] not in valid_prompts:
            print(f"Not valid key {key}!!")
            continue
        if key not in target_dict.keys():
            target_dict[key]={}
        try:
            score_dict=extract_llm_scores(raw_llm_dict[key]['LLM Eval'])
            for attr in score_dict.keys():
                if attr not in target_dict[key].keys():
                    target_dict[key][attr]={}
                target_dict[key][attr]['LLM']=score_dict[attr]
        except Exception as e:
            print(key)
            continue
    return target_dict,error_list

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

def obtain_syntax_valid(metric_static_dict,result_log_dict,valid_prompts):
    for key in metric_static_dict.keys():
        if valid_prompts!=None and key[:-2] not in valid_prompts:
            continue
        if 'Syntax Correctness' in metric_static_dict[key].keys():
            continue
        json_path=result_log_dict[key]
        query_num=key.split('-')[-1]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        value=data[query_num]
        code=extract_longest_code(value)
        try:
            compile(code, "<string>", "exec")
            metric_static_dict[key]['Syntax Correctness']=1
        except SyntaxError as e:
            metric_static_dict[key]['Syntax Correctness']=0
    return metric_static_dict

def update_static_results(target_dict,raw_static_dict,valid_prompts,result_log_dict, error_list):
    for key in raw_static_dict.keys():
        if valid_prompts!=None and key[:-2] not in valid_prompts:
            continue
        if key not in target_dict.keys():
            target_dict[key]={}
        # if key =='Computer vision--Face recognition==4==OpenCV-face-recognition-1':
        #     json_path=result_log_dict[key]
        #     log_path=json_path.replace('.json','-log.pkl')
        #     with open(log_path, 'rb') as f:
        #         tmp_log = pickle.load(f)
        #     print(1)
        # try:
        score_dict=extract_static_scores(raw_static_dict[key])
        for attr in score_dict.keys():
            if attr not in target_dict[key].keys():
                target_dict[key][attr]={}
            target_dict[key][attr]['Static']=score_dict[attr]
        # except:
        #     print(key)
        #     continue
    return target_dict,error_list

def extract_scenario_results(score_dict):
    """
    input: (1) score_dict: key is the prompt name, value is a list of scores.
    output: (1) scenario_result_dict: key is the scenario name, subkey is the third party library, subvalue is a list of score lists
    """ 
    scenario_result_dict={}
    for key in score_dict.keys():
        sample_name='-'.join(key.split('-')[:-1])
        lib=sample_name.split('==')[-1]
        scenario=sample_name.split('==')[0]
        if scenario not in scenario_result_dict.keys():
            scenario_result_dict[scenario]={}
        if lib not in scenario_result_dict[scenario].keys():
            scenario_result_dict[scenario][lib]=[]
        scenario_result_dict[scenario][lib].append(score_dict[key])
    return scenario_result_dict

def extract_task_results(score_dict):
    """
    input: (1) score_dict: key is the prompt name, value is a list of scores.
    output: (1) scenario_result_dict: key is the scenario name, subkey is the third party library, subvalue is a list of score lists
    """ 
    scenario_result_dict={}
    for key in score_dict.keys():
        sample_name=key[:-2]
        lib=sample_name.split('==')[-1]
        task='=='.join(sample_name.split('==')[:-1])
        if task not in scenario_result_dict.keys():
            scenario_result_dict[task]={}
        if lib not in scenario_result_dict[task].keys():
            scenario_result_dict[task][lib]=[]
        scenario_result_dict[task][lib].append(score_dict[key])
    return scenario_result_dict

def bootstrap_scores(score_list,n_bootstrap=1000,confidence_level=0.95):
    """
    bootstrap 1000 sample and get the 95% intervel
    """
    data = np.array(score_list)

    n_samples = data.shape[0]
    n_features = data.shape[1]
    B = n_bootstrap

    boot_means = np.zeros((B, n_features))

    rng = np.random.default_rng(None)
    for b in range(B):
        idx = rng.integers(0, n_samples, n_samples)
        sample = data[idx]
        boot_means[b] = sample.mean(axis=0)

    final_mean = boot_means.mean(axis=0)
    lower = np.percentile(boot_means, 2.5, axis=0)
    upper = np.percentile(boot_means, 97.5, axis=0)

    final_mean = [round(x, 2) for x in final_mean.tolist()]
    lower = [round(x, 2) for x in lower.tolist()]
    upper = [round(x, 2) for x in upper.tolist()]
    return [final_mean, lower, upper]

def get_model_score_bs(model_score_dict,n_bootstrap=1000,confidence_level=0.95):
    bs_result_dict={}
    for sce in model_score_dict.keys():
        bs_result_dict[sce]={}
        for lib in model_score_dict[sce].keys():
            bs_result_dict[sce][lib]=bootstrap_scores(model_score_dict[sce][lib],n_bootstrap=n_bootstrap,confidence_level=n_bootstrap)
    return bs_result_dict

def extract_radar_data(scenario_result_bs,dim_len=6):
    lib_list=list(scenario_result_bs.keys())
    radar_data_list=[]
    score_list=[]
    for lib in lib_list:
        radar_data_list.append(scenario_result_bs[lib][0][:dim_len])
        score_list.append(scenario_result_bs[lib][0][dim_len])
    return radar_data_list,lib_list,score_list

def extract_radar_data_model(bs_score_dict,sce,lib_name,dim_len=6):
    radar_data_list=[]
    for model in bs_score_dict.keys():
        radar_data_list.append(bs_score_dict[model][sce][lib_name][0][:dim_len])
    return radar_data_list

def draw_rador_chart(data_list,
                     lib_list,
                     dim_list,
                     save_dir,
                     name,
                     fill_alpha=0.1,
                     figsize=(10,10),
                     max_ncol=3,
                     col_space=0.5,
                     legend_font_size=28,
                     dim_len=6,
                     rtick_list=[0, 20, 40, 60, 80],
                     y_lim=[0,100]):
    save_path=os.path.join(save_dir,name)
    if os.path.exists(save_path):
        print(f"Path Existing: {save_path}!")
        return 0
    n_groups = len(data_list)
    arr = np.array([list(map(float, g)) for g in data_list], dtype=float)

    base_angles = np.linspace(0, 2*np.pi, dim_len, endpoint=False)
    tick_angles_deg = np.degrees(base_angles)

    plot_angles = np.concatenate([base_angles, base_angles[:1]])

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(tick_angles_deg, dim_list)
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_rticks(rtick_list)
    ax.set_rlabel_position(0)
    # ax.tick_params(axis='x', labelsize=28, pad=12)  # dimension ticks
    ax.tick_params(axis='y', labelsize=28)  # radial ticks
    ax.set_facecolor(tuple(np.array((229, 236, 246))  / 255.0))
    ax.grid(color='white', linewidth=2)
    ax.spines["polar"].set_color('white')

    # custom angle labels placed farther out with edge aligned to the axis
    if dim_len==6:
        far_dims=dim_list[1:3]+dim_list[4:]
        near_dims=[dim_list[0],dim_list[3]]
    else:
        far_dims=dim_list[1:]
        near_dims=[dim_list[0]]
    ax.set_xticks(base_angles)
    ax.set_xticklabels([])
    for ang, lab in zip(base_angles, dim_list):
        r = 120 if lab in set(far_dims) else 105 if lab in set(near_dims) else 105
        ax.text(ang, r, lab, ha='center', va='center', fontsize=28)

    handles = []
    for i in range(n_groups):
        vals = arr[i]
        vals_closed = np.r_[vals, vals[0]]
        color = None if PASTEL_COLOR is None else PASTEL_COLOR[i]
        (line,) = ax.plot(plot_angles, vals_closed, linewidth=3, color=color)
        ax.fill(plot_angles, vals_closed, alpha=fill_alpha, color=line.get_color())
        handles.append(line)

    if n_groups>max_ncol:
        ncol=max_ncol
        ax.legend(handles, lib_list, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=ncol,fontsize=legend_font_size,columnspacing=col_space)
    else:
        ncol=n_groups
        ax.legend(handles, lib_list, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=ncol,fontsize=legend_font_size,columnspacing=col_space)
    plt.subplots_adjust(top=0.85, bottom=0.05,right=0.85,left=0.15)
    plt.savefig(save_path, dpi=200)
    plt.close()


def draw_rador_chart_tmp(data_list,
                     lib_list,
                     dim_list,
                     save_dir,
                     name,
                     fill_alpha=0.1,
                     figsize=(10,10),
                     max_ncol=3,
                     col_space=0.5,
                     legend_font_size=36,
                     dim_len=6,
                     rtick_list=[0, 20, 40, 60, 80],
                     y_lim=[0,100],
                     replace=False):
    save_path=os.path.join(save_dir,name)
    if not replace and os.path.exists(save_path):
        print(f"Path Existing: {save_path}!")
        return 0
    n_groups = len(data_list)
    arr = np.array([list(map(float, g)) for g in data_list], dtype=float)

    base_angles = np.linspace(0, 2*np.pi, dim_len, endpoint=False)
    tick_angles_deg = np.degrees(base_angles)

    plot_angles = np.concatenate([base_angles, base_angles[:1]])

    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(tick_angles_deg, dim_list)
    ax.set_ylim(y_lim[0], y_lim[1])
    ax.set_rticks(rtick_list)
    ax.set_rlabel_position(0)
    # ax.tick_params(axis='x', labelsize=28, pad=12)  # dimension ticks
    ax.tick_params(axis='y', labelsize=36)  # radial ticks
    ax.set_facecolor(tuple(np.array((229, 236, 246))  / 255.0))
    ax.grid(color='white', linewidth=2)
    ax.spines["polar"].set_color('white')

    # custom angle labels placed farther out with edge aligned to the axis
    if dim_len==6:
        far_dims=dim_list[1:3]+dim_list[4:]
        near_dims=[dim_list[0],dim_list[3]]
    else:
        far_dims=dim_list[1:]
        near_dims=[dim_list[0]]
    ax.set_xticks(base_angles)
    ax.set_xticklabels([])
    if max_ncol==2:
        for ang, lab in zip(base_angles, dim_list):
            r = 110 if lab in set(far_dims) else 105 if lab in set(near_dims) else 105
            ax.text(ang, r, lab, ha='center', va='center', fontsize=36)
    else:
        for ang, lab in zip(base_angles, dim_list):
            r = 130 if lab in set(far_dims) else 110 if lab in set(near_dims) else 110
            ax.text(ang, r, lab, ha='center', va='center', fontsize=36)

    handles = []
    for i in range(n_groups):
        vals = arr[i]
        vals_closed = np.r_[vals, vals[0]]
        color = None if PASTEL_COLOR is None else PASTEL_COLOR[i]
        (line,) = ax.plot(plot_angles, vals_closed, linewidth=3, color=color)
        ax.fill(plot_angles, vals_closed, alpha=fill_alpha, color=line.get_color())
        handles.append(line)

    plt.subplots_adjust(top=0.8, bottom=0.05,right=0.8,left=0.2)
    if n_groups>max_ncol:
        ncol=max_ncol
        if max_ncol==2:
            ax.legend(handles, lib_list, loc="upper center", bbox_to_anchor=(0.5, 1.47), ncol=ncol,fontsize=legend_font_size-4,columnspacing=col_space)
        else:
            ax.legend(handles, lib_list, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=ncol,fontsize=legend_font_size,columnspacing=col_space)
    else:
        ncol=n_groups
        ax.legend(handles, lib_list, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=ncol,fontsize=legend_font_size,columnspacing=col_space)
    
    plt.savefig(save_path, dpi=200)

def draw_bar_chart_with_ci(scores,
                     model_names,
                     confidence_intervals,
                     save_path,
                     x_label,
                     y_label,
                     tick_fontsize=36,
                     legend_fontsize=36,
                     figsize=(10, 10),
                     ylim=(75, 81),
                     error_color = 'red',
                     show_values = True,
                     bar_alpha = 0.8):

    lower_errors = [score - ci[0] for score, ci in zip(scores, confidence_intervals)]
    upper_errors = [ci[1] - score for score, ci in zip(scores, confidence_intervals)]

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(model_names))
    bars = ax.bar(x_pos, scores, 
                  alpha=bar_alpha,
                  edgecolor='black',
                  linewidth=1,
                  label=y_label)
    
    ax.errorbar(x_pos, scores, 
                yerr=[lower_errors, upper_errors],
                fmt='none',
                ecolor=error_color,
                capsize=5,
                capthick=2,
                linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, fontsize=tick_fontsize, rotation=30, ha='right')
    ax.set_xlabel(x_label, fontsize=tick_fontsize)

    ax.set_ylabel(y_label, fontsize=tick_fontsize)
    ax.set_ylim(ylim)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

    if show_values:
        for i, (bar, score, ci) in enumerate(zip(bars, scores, confidence_intervals)):
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + (upper_errors[i] * 0.3),
                   f'{score:.2f}',
                   ha='center', va='bottom',
                   fontsize=tick_fontsize,
                   fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.35,right=0.98,left=0.2)
    plt.savefig(save_path, dpi=200)
    print(1)

def extact_gap_keys_syntax(score_dict,key):
    tmp_scenario=key.replace(key.split('==')[-1],'')
    target_dict={k:v for k, v in score_dict.items() if tmp_scenario in k}
    sorted_items = sorted(target_dict.items(), key=lambda x: x[1][-1])
    min_key = key
    max_key, max_val = sorted_items[-1]
    while min_key.split('==')[-1][:-2] == max_key.split('==')[-1][:-2]:
        sorted_items=sorted_items[:-1]
        max_key, max_val = sorted_items[-1]
    return [max_key,min_key],[max_val,target_dict[min_key]]

def extact_gap_keys(score_dict,sce):
    # get results in this scenario
    key_dict={}
    for i in range(10):
        target_dict={k:v for k, v in score_dict.items() if sce+f'=={i}' in k}
        if target_dict=={}:continue
        sorted_items = sorted(target_dict.items(), key=lambda x: x[1][-1])
        min_key, min_val = sorted_items[0]
        max_key, max_val = sorted_items[-1]
        while min_key.split('==')[-1][:-2] == max_key.split('==')[-1][:-2]:
            sorted_items=sorted_items[:-1]
            min_key, min_val = sorted_items[0]
            max_key, max_val = sorted_items[-1]
        key_dict[sce+f'=={i}']=[[max_key,min_key],[max_val,min_val]]
    return key_dict

def extact_gap_keys_maintain(score_dict,sce,thre=30):
    # get results in this scenario
    key_dict={}
    for i in range(10):
        target_dict={k:v for k, v in score_dict.items() if sce+f'=={i}' in k}
        if target_dict=={}:continue
        tmp_dict={}
        for key in target_dict.keys():
            service=key[:-2]
            if service not in tmp_dict.keys():
                tmp_dict[service]=[]
            tmp_dict[service].append(target_dict[key])
        new_dict = {}
        for k, v in tmp_dict.items():
            arr = np.array(v, dtype=float)
            mean_arr = arr.mean(axis=0)
            new_dict[k] = mean_arr.tolist()
        sorted_items = sorted(new_dict.items(), key=lambda x: x[1][-1])
        min_key, min_val = sorted_items[0]
        max_key, max_val = sorted_items[-1]
        while min_key.split('==')[-1] == max_key.split('==')[-1] or not((new_dict[max_key][2]-new_dict[min_key][2]>thre) or (new_dict[max_key][3]-new_dict[min_key][3]>thre)):
            if len(sorted_items)<=1:break
            sorted_items=sorted_items[:-1]
            min_key, min_val = sorted_items[0]
            max_key, max_val = sorted_items[-1]
        if len(sorted_items)>1:
            key_dict[sce+f'=={i}']=[[max_key,min_key],[max_val,min_val]]
    return key_dict

def extract_gap_response(result_log_dict,gap_keys):
    def _obtain_code(json_path,num):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        value=data[num]
        # code=extract_longest_code(value)
        return value
    [max_key,min_key]=gap_keys
    max_response_path=result_log_dict[max_key]
    max_query_num=max_key.split('-')[-1]
    min_response_path=result_log_dict[min_key]
    min_query_num=min_key.split('-')[-1]
    max_response=_obtain_code(max_response_path,max_query_num)
    min_response=_obtain_code(min_response_path,min_query_num)
    return max_response,min_response


# =========== For RQ1 ===========
class CodeQualityDifferenceAnalyzer:
    def __init__(self, threshold = 0.5, dim = -1):
        self.threshold = threshold # default to be 0.5
        self.dim = dim # default to analyze the -1 dimension
        self.results = {}
    
    def calculate_cohens_d(self, group1, group2):
        """
        Args:
            group1: code a's respeated scores (shape: [n_repeats])
            group2: code b's respeated scores (shape: [n_repeats])
            
        Returns:
            Cohen's d
        """
        diff = group1 - group2
        diff_std = np.std(diff, ddof=1)
        mean_diff = np.mean(diff)
        cohen_d = mean_diff / diff_std if diff_std != 0 else 0
        return cohen_d
    
    
    def analyze_code_pairs(self, data):
        """
        quality score difference analyzer. 
        
        Args:
            data: {model_name: [(code1_scores, code2_scores), ...]}. Each score is an array of shape [n_repeats, quality_dims+1], and the last column is the average score.
                  
        Returns:
            a dict of analyzer outputs
        """
        results = {}
        
        for model_name, code_pairs in data.items():
            print(model_name)
            model_results = []
            
            for pair_idx, (code1_scores, code2_scores) in enumerate(code_pairs):
                code1_avg_scores = code1_scores[:, self.dim]
                code2_avg_scores = code2_scores[:, self.dim]
                
                cohen_d = self.calculate_cohens_d(code1_avg_scores, code2_avg_scores)
                
                diff_scores = code1_avg_scores - code2_avg_scores
                t_stat, p_value = stats.ttest_1samp(diff_scores, 0)
                
                mean_diff = np.mean(diff_scores)
                std_diff = np.std(diff_scores, ddof=1)
                
                is_high_diff = abs(cohen_d) >= self.threshold and p_value < 0.05
                
                pair_result = {
                    'pair_id': pair_idx,
                    'cohen_d': cohen_d,
                    'abs_cohen_d': abs(cohen_d),
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'p_value': p_value,
                    'is_significant': p_value < 0.05,
                    'is_high_diff': is_high_diff,
                    'code1_mean': np.mean(code1_avg_scores),
                    'code2_mean': np.mean(code2_avg_scores),
                    'effect_size_interpretation': self._interpret_effect_size(abs(cohen_d))
                }
                
                model_results.append(pair_result)
            
            results[model_name] = model_results
            
            high_diff_count = sum(1 for r in model_results if r['is_high_diff'])
            total_pairs = len(model_results)
        
        self.results = results
        return results
    
    def _interpret_effect_size(self, abs_cohen_d):
        """https://en.wikipedia.org/wiki/Effect_size#Interpretation"""
        if abs_cohen_d < 0.2:
            return "Very small"
        elif abs_cohen_d < 0.5:
            return "Small"
        elif abs_cohen_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def get_high_difference_pairs(self):
        high_diff_pairs = {}
        
        for model_name, model_results in self.results.items():
            high_diff_pairs[model_name] = [
                result for result in model_results if result['is_high_diff']
            ]
        
        return high_diff_pairs
        
def extract_competing_dict(task_result_dict):
    competing_data_dict={}
    key_pair_dict={}
    for model in task_result_dict.keys():
        competing_data_dict[model]=[]
        key_pair_dict[model]=[]
        for task in task_result_dict[model].keys():
            items = list(task_result_dict[model][task].items())
            for (a1, b1), (a2, b2) in combinations(items, 2):
                kv = [task, a1, a2]
                vv = [np.array(b1), np.array(b2)]
                key_pair_dict[model].append(kv)
                competing_data_dict[model].append(vv)
    return competing_data_dict,key_pair_dict

def sort_keys_by_each_dim(
    data_dict,
    dim=6,
    descending = True):

    orderings = []
    keys = list(data_dict.keys())  # 保留原始顺序（Python 排序是稳定的）
    for i in range(dim):
        orderings.append(sorted(keys, key=lambda k: data_dict[k][0][i], reverse=descending))
    return orderings


def spearman_ranking_correlation(list1, list2):
    ranks1 = [list1.index(item) + 1 for item in list1]
    ranks2 = [list2.index(item) + 1 for item in list1]
    correlation, p_value = stats.spearmanr(ranks1, ranks2)
    is_significant = p_value < 0.05
    return correlation, p_value, is_significant

def kendall_ranking_correlation(list1, list2):
    ranks1 = [list1.index(item) + 1 for item in list1]
    ranks2 = [list2.index(item) + 1 for item in list1]
    correlation, p_value = stats.kendalltau(ranks1, ranks2)
    is_significant = p_value < 0.05
    return correlation, p_value, is_significant

def extract_model_results_for_libs(scenario_lib_results,scenario_list,model_list):
    # default: directly extract scenarios from scenario_lib_restuls['mini']
    model_lib_results={}
    for scenario in scenario_list:
        lib_list=list(scenario_lib_results['mini'][scenario].keys())
        for lib in lib_list:
            tmp_key=f'{scenario}=={lib}'
            model_lib_results[tmp_key]={}
            model_lib_results[tmp_key]['Model Scores']=[]
            for model in model_list:
                model_lib_results[tmp_key]['Model Scores'].append(scenario_lib_results[model][scenario][lib][0])
            average_score=[_list[-1] for _list in model_lib_results[tmp_key]['Model Scores']]
            model_lib_results[tmp_key]['Model Gap']=max(average_score)-min(average_score)
            model_lib_results[tmp_key]['Best Model']=model_list[average_score.index(max(average_score))]
            model_lib_results[tmp_key]['Worst Model']=model_list[average_score.index(min(average_score))]
    return model_lib_results

def update_model_scenario_results(model_lib_results,model_dict):
    best_model_count,worst_model_count={},{}
    for sce in model_lib_results.keys():
        if model_lib_results[sce]['Best Model'] not in best_model_count.keys():
            best_model_count[model_lib_results[sce]['Best Model']]=0
        if model_lib_results[sce]['Worst Model'] not in worst_model_count.keys():
            worst_model_count[model_lib_results[sce]['Worst Model']]=0
        best_model_count[model_lib_results[sce]['Best Model']]+=1
        worst_model_count[model_lib_results[sce]['Worst Model']]+=1
    best_model_count={k:best_model_count[v] for k,v in model_dict.items()}
    worst_model_count={k:worst_model_count[v] for k,v in model_dict.items()}
    return best_model_count,worst_model_count



def draw_pie_chart_from_dict(
    data_dict,
    save_path,
    tick_fontsize=36, 
    legend_fontsize=36, 
    figsize=(10, 10),
    colors=None, 
    alpha=0.8,
    show_pct=True,
    pct_format="{:.2f}%", 
    startangle=90,
    legend_loc="lower center",
    legend_bbox=(0.5, -0.35),
    ncol=2,
    columnspacing=0.5,
    subplots_adjust={"top":0.95,"bottom":0.3,"left":0.1,"right":0.9}
):


    items = list(data_dict.items())
    labels = [k for k, _ in items]
    vals   = [v for _, v in items]


    fig, ax = plt.subplots(figsize=figsize)

    # 绘制饼图
    if colors!=None:
        colors = [mcolors.to_rgba(c, alpha=alpha) for c in colors]
    else:
        colors=sb.color_palette("tab20", 2*len(labels[0]))
        colors=[(*_cl,0.5) for _cl in colors]
        colors=[colors[2*i] for i in range(int(len(colors)/2))]
    wedges, texts, autotexts = ax.pie(
        vals,
        labels=None,
        colors=colors,
        autopct=(lambda p: pct_format.format(p)) if show_pct else None,
        startangle=startangle,
        counterclock=False,
        pctdistance=1.1,
        textprops={"fontsize": tick_fontsize},
        # wedgeprops={"edgecolor": "black", "linewidth": 1}
    )
    ax.axis("equal")

    if show_pct:
        for t in autotexts:
            t.set_fontsize(tick_fontsize)

    # 图例
    ax.legend(
        wedges, labels,
        loc=legend_loc,ncol=ncol,columnspacing=columnspacing,
        bbox_to_anchor=legend_bbox,
        fontsize=legend_fontsize-4,
    )


    plt.tight_layout()
    if subplots_adjust:
        plt.subplots_adjust(**subplots_adjust)

    plt.savefig(save_path, dpi=200)

# =========== For RQ2 ===========
def extract_lib_scenario_scores(prompt_score_dict,model_list,dim=-1):
    lib_sce_score_dict={}
    lib_sce_ranking_dict={}
    for model in model_list:
        lib_sce_score_dict[model]={}
        tmp_result_dict=prompt_score_dict[model]
        for key in tmp_result_dict.keys():
            sce=key.split('==')[0]
            lib=key.split('==')[-1][:-2]
            if sce not in lib_sce_score_dict[model].keys():
                lib_sce_score_dict[model][sce]={}
            if lib not in lib_sce_score_dict[model][sce].keys():
                lib_sce_score_dict[model][sce][lib]=[]
            lib_sce_score_dict[model][sce][lib].append(tmp_result_dict[key][dim])
    lib_sce_score_dict['Average']={}
    for sce in lib_sce_score_dict['mini'].keys():
        lib_sce_score_dict['Average'][sce]={}
        for lib in lib_sce_score_dict[model][sce].keys():
            lib_sce_score_dict['Average'][sce][lib]=np.average([lib_sce_score_dict[model][sce][lib] for model in model_list])
    for model in lib_sce_score_dict.keys():
        lib_sce_ranking_dict[model]={}
        for sce in lib_sce_score_dict[model].keys():
            for lib in lib_sce_score_dict[model][sce].keys():
                lib_sce_score_dict[model][sce][lib]=np.average(lib_sce_score_dict[model][sce][lib])
            sorted_items = sorted(lib_sce_score_dict[model][sce].items(), key=lambda x: x[1], reverse=True)
            lib_sce_ranking_dict[model][sce] = [k for k, v in sorted_items]
    return lib_sce_score_dict,lib_sce_ranking_dict

def generate_ranking_popular_dict(lib_sce_ranking_dict,scenario_dict):
    ranking_popular_dict={}
    for model in lib_sce_ranking_dict.keys():
        ranking_popular_dict[model]={}
        for sce in lib_sce_ranking_dict[model].keys():
            ranking_popular_dict[model][sce]=[]
            for lib in lib_sce_ranking_dict[model][sce]:
                tmp=[lib,
                scenario_dict[sce][lib]['Github Stars'],
                scenario_dict[sce][lib]['Github Watchers'],
                scenario_dict[sce][lib]['Github Forks'],
                scenario_dict[sce][lib]['Stack Overflow Search'],
                scenario_dict[sce][lib]['Stack Overflow Tag']]
                ranking_popular_dict[model][sce].append(tmp)
    return ranking_popular_dict


def kendall_tau_popularity_analysis(scenario_data,data_index=1):
    """
    Use Kendall Tau-b to analyze the correlation between interoperability and popularity data of the codebase.
    
    scenario_data: dictionary, key is scene name, value is (library name, popularity data...).
    The order of tuples in the list represents interoperability.
    Return a dictionary, containing Kendall Tau-b coefficient, p-value, and overall analysis result for each scene
    """
    
    results = {}
    results['Scenario']={}
    all_tau_values = []
    all_p_values = []
    
    for scenario_name, libraries in scenario_data.items():
        
        popularity_ranks = list(range(len(libraries)))
        
        downloads = [int(lib[data_index]) for lib in libraries]
        download_ranks = np.argsort(np.argsort(downloads)[::-1])
        
        tau, p_value = kendalltau(popularity_ranks, download_ranks)
        
        results['Scenario'][f"{scenario_name}_tau"] = tau
        results['Scenario'][f"{scenario_name}_p_value"] = p_value
        
        if not np.isnan(tau):
            all_tau_values.append(tau)
            all_p_values.append(p_value)
    
    if all_tau_values:
        results["overall_mean_tau"] = np.mean(all_tau_values)
        results["overall_median_tau"] = np.median(all_tau_values)
        results["significant_scenarios"] = sum(1 for p in all_p_values if p < 0.05)
        results["total_scenarios"] = len(all_tau_values)
        results["proportion_significant"] = results["significant_scenarios"] / results["total_scenarios"]
    
    return results

def extract_pairs_and_popularity(data,index=1):
    pairs = []
    popularity = {}
    for group in data.values():

        for item in group:
            lib, pp = item[0], int(item[index])
            popularity[lib] = pp
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                winner = group[i][0]
                loser = group[j][0]
                pairs.append((winner, loser))
    return pairs, popularity

def _wilson_interval(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (center - half, center + half)

def _collect_deltas(pairs: List[Tuple[str, str]],
                    downloads: Dict[str, float],
                    use_log1p: bool = True):
    """Return list of deltas: log1p(dw_winner) - log1p(dw_loser)."""
    deltas = []
    skipped = 0
    for w, l in pairs:
        if w not in downloads or l not in downloads:
            skipped += 1
            continue
        x, y = downloads[w], downloads[l]
        dx = (np.log1p(x) - np.log1p(y)) if use_log1p else (x - y)
        deltas.append(dx)
    return deltas, skipped

def concordance_binomial(pairs, downloads, use_log1p=True):
    deltas, skipped = _collect_deltas(pairs, downloads, use_log1p)
    pos = sum(dx > 0 for dx in deltas)
    neg = sum(dx < 0 for dx in deltas)
    ties = sum(dx == 0 for dx in deltas)
    n = pos + neg
    if n == 0:
        raise ValueError("No effective pairs (all ties or missing).")
    phat = pos / n

    # exact two-sided binomial (needs SciPy; fallback provided if missing)
    try:
        from scipy.stats import binomtest
        pval = binomtest(k=pos, n=n, p=0.5, alternative="two-sided").pvalue
    except Exception:
        # simple exact calculation by probability tail (conservative)
        from math import comb
        obs = comb(n, pos) * (0.5**n)
        total = 0.0
        for i in range(n+1):
            prob = comb(n, i) * (0.5**n)
            if prob <= obs + 1e-16:
                total += prob
        pval = min(1.0, total)

    ci = _wilson_interval(pos, n)
    # Cohen's h effect size
    h = 2*asin(np.sqrt(phat)) - 2*asin(np.sqrt(0.5))
    # AUC (c-index) equals pos+0.5*ties over all pairs including ties
    auc = (pos + 0.5*ties) / (pos + neg + ties) if (pos+neg+ties) else float("nan")

    return {
        "skipped_pairs": skipped,
        "concordant": pos, "discordant": neg, "ties": ties, "n_effective": n,
        "concordance_rate": phat, "concordance_rate_CI": ci,
        "p_binom_two_sided": pval,
        "cohens_h": h,
        "auc_cindex": auc
    }

def wilcoxon_signed_rank(pairs, downloads, use_log1p=True, alternative="greater"):
    from scipy.stats import wilcoxon
    deltas, _ = _collect_deltas(pairs, downloads, use_log1p)
    deltas = [dx for dx in deltas if dx != 0]
    if len(deltas) == 0:
        raise ValueError("No non-zero deltas for Wilcoxon.")
    stat, pval = wilcoxon(deltas, alternative=alternative, zero_method="wilcox")
    median = float(np.median(deltas))
    return {"n_nonzero": len(deltas), "median_delta": median, "p_wilcoxon": pval}

def conditional_logit_FE(pairs, downloads, use_log1p=True):
    """
    Approximate conditional logit by adding pair fixed effects (dummies) to a Logit.
    Requires statsmodels; returns OR and p-value for the downloads coefficient.
    """
    import pandas as pd
    import statsmodels.api as sm

    rows = []
    for pid, (w, l) in enumerate(pairs):
        if w in downloads and l in downloads:
            xw = np.log1p(downloads[w]) if use_log1p else downloads[w]
            xl = np.log1p(downloads[l]) if use_log1p else downloads[l]
            rows.append({"pair": pid, "lib": w, "y": 1, "x": xw})
            rows.append({"pair": pid, "lib": l, "y": 0, "x": xl})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No data for FE-logit.")
    X = pd.get_dummies(df[["x","pair"]].astype({"pair":"category"}), drop_first=True)
    y = df["y"].values
    model = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
    beta = model.params["x"]
    or_val = float(np.exp(beta))
    pval = float(model.pvalues["x"])
    return {"n_rows": len(df), "beta": float(beta), "OR": or_val, "p_logit": pval}

def kendall_lib_level(pairs, downloads):
    from collections import defaultdict
    from scipy.stats import kendalltau
    played, wins = defaultdict(int), defaultdict(int)
    for w, l in pairs:
        wins[w] += 1
        played[w] += 1
        played[l] += 1
    libs = [lib for lib in played if lib in downloads]
    if not libs:
        raise ValueError("No overlapping libraries for Kendall.")
    winrate = [wins[lib]/played[lib] for lib in libs]
    pop = [np.log1p(downloads[lib]) for lib in libs]
    tau, pval = kendalltau(pop, winrate, nan_policy='omit')
    return {"n_libs": len(libs), "kendall_tau": float(tau), "p_kendall": float(pval)}

def run_all_tests(pairs, downloads, use_log1p=True):
    out = {}
    out["binomial"] = concordance_binomial(pairs, downloads, use_log1p)
    try:
        out["wilcoxon"] = wilcoxon_signed_rank(pairs, downloads, use_log1p, alternative="greater")
    except Exception as e:
        out["wilcoxon"] = {"error": str(e)}
    # try:
    #     out["clogit_FE"] = conditional_logit_FE(pairs, downloads, use_log1p)
    # except Exception as e:
    #     out["clogit_FE"] = {"error": str(e)}
    try:
        out["kendall_lib"] = kendall_lib_level(pairs, downloads)
    except Exception as e:
        out["kendall_lib"] = {"error": str(e)}
    return out

# =========== For RQ2 New ===========
def get_worst_index_IQR(scores):
    Q1 = np.percentile(scores, 25)
    Q3 = np.percentile(scores, 75)
    IQR = Q3 - Q1
    worst_threshold = Q1 - 1.5 * IQR
    worst_index = [i for i in range(len(scores)) if scores[i] <=worst_threshold]
    return worst_index,worst_threshold


def draw_rq1_squarify_pdf_new(data,fig_path,threshold=0.03,overwrite=False,font_size=18,legend_size=None,colors=None,ncol=3,size=(6.5, 5), white_text=False,white_outline=False):
    if legend_size==None:
        legend_size=font_size
    sum_value=sum(data.values())
    others_value = sum(value for key, value in data.items() if value/sum_value < threshold)
    
    filtered_data = {key: value for key, value in data.items() if value/sum_value >= threshold}
    if others_value > 0:
        filtered_data['Others'] = others_value
    if colors==None:
        colors=sb.color_palette("tab20", len(filtered_data))
    # Extract keys and values
    labels = list(filtered_data.keys())
    sizes = list(filtered_data.values())
    proportion = [f"{_s/sum(sizes):.2%}" for _s in sizes]
    return_dict={k:v/sum(sizes) for k,v in data.items()}
    # if 0 in return_dict.values():
    #     print(3)
    if overwrite or not os.path.exists(fig_path):# overwrite or no figure
        # if overwrite=False:
        #     print('Not implement overwrite now')
        fig_proportion = [f"{proportion[i]}" for i in range(len(labels))]#[f"{labels[i]}\n{proportion[i]}" for i in range(len(labels))]
        legend = [f"{labels[i]}" for i in range(len(labels))]
        # Create the treemap
        plt.figure(figsize=size)  # (9,4)
        ax = plt.gca()
        squarify.plot(sizes=sizes,
            label=fig_proportion,
            # alpha=0.4,
            pad=0,
            color=colors,
            text_kwargs={'fontsize': font_size,'color': 'white'} if white_text else {'fontsize': font_size},
            # bar_kwargs={'edgecolor': 'black', 'linewidth': 1}
            )
        # , color=plt.cm.tab20.colors
        plt.axis('off')
        if white_outline:
            for text in ax.texts:  # Access all text objects in the plot
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),  # White outline
                    path_effects.Normal()  # Normal text inside
                ])

        legend_patches = [Patch(color=color, label=label) for color, label in zip(colors, labels)]
        # Add the legend
        plt.legend(handles=legend_patches, loc='upper center', fontsize=legend_size, bbox_to_anchor=(0.5, 1.7),ncol=ncol,columnspacing=1)#math.ceil(len(filtered_data)/3)

        plt.subplots_adjust(top=0.6, bottom=0.01,right=0.96,left=0.01)
        if '.pdf' in fig_path:
            plt.savefig(fig_path,dpi=800)
        else:
            plt.savefig(fig_path)
        plt.close()
    return return_dict

def visualize_cases(raw_pattern_dict,score_dict,raw_output_dir,save_dir,pattern_list):
    
    for pattern,case_list in raw_pattern_dict.items():
        pattern_index=pattern_list.index(pattern)
        save_dict={}
        save_path=os.path.join(save_dir,f'rq2_{pattern}.json')
        # if os.path.exists(save_path):continue
        for case in case_list:
            save_dict[case]={}
            model=case.split('==')[0]
            output_save_dir=os.path.join(raw_output_dir,model)
            result_log_pkl=os.path.join(output_save_dir,'result.pkl')
            with open(result_log_pkl, 'rb') as f:
                result_log_dict = pickle.load(f)
            total_llm_path=os.path.join(output_save_dir,'result_metric-llm-total.pkl')
            with open(total_llm_path, 'rb') as f:
                llm_evaluate_dict = pickle.load(f)
            save_dict[case]['Score']=score_dict[case]
            case_key='=='.join(case.split('==')[1:])
            if case=='mini==Big data processing--Scenario==0==Apache Spark-0':
                print(1)
            save_dict[case]['LLM Evaluation']=get_targeted_evaluation(llm_evaluate_dict[case_key],pattern_index)
            save_dict[case]['Code Path']=result_log_dict[case_key]
        
        print(pattern)
        print(len(save_dict))
        with open(save_path, 'w') as json_file:
            json.dump(save_dict, json_file, indent=4) 
        print(f'saved: {save_path}')

def extract_score_context(text, id):
    lines = text.splitlines()
    lines = [_l for _l in lines if _l!='']
    text = '\n'.join(lines)
    pattern = r'(?im)(?:^.{0,20}score[^\n]{0,20}?(\d{1,2})|score[^\n]{0,20}?(\d{1,2}).{0,20}$)'
    matches = list(re.finditer(pattern, text))

    match = matches[id]
    start = match.start()
    line_index = text[:start].count("\n")
    
    context = []
    if line_index - 1 >= 0:
        # if lines[line_index - 1]=="" and line_index - 2 >= 0:
        #     context.append(lines[line_index - 2])
        # else:
        context.append(lines[line_index - 1])
        if line_index - 2 >= 0:
            # if lines[line_index - 1]=="" and line_index - 2 >= 0:
            #     context.append(lines[line_index - 2])
            # else:
            context.append(lines[line_index - 2])
    context.append(lines[line_index])
    if line_index + 1 < len(lines):
        # if lines[line_index + 1]=="" and line_index + 2 < len(lines):
        #     context.append(lines[line_index + 2])
        # else:
        context.append(lines[line_index + 1])
        if line_index + 2 < len(lines):
            # if lines[line_index + 1]=="" and line_index + 2 < len(lines):
            #     context.append(lines[line_index + 2])
            # else:
            context.append(lines[line_index + 2])
    return context

def get_targeted_evaluation(llm_evaluation,pattern_index):
    # 'Functional Suitability', 'Performance', 'Maintainability','Usability','Reliability',
    index_to_number={
        0:0,
        1:1,
        2:9,
        3:3,
        4:4,
        5:-1}
    output_response=extract_score_context(llm_evaluation['LLM Eval'],index_to_number[pattern_index])
    raw_eval='\n'.join(output_response)
    return raw_eval


def label_cases(case_list,raw_output_dir,save_dir):
    def _generate_annotation(code,reference,prompt_path='./ppl_utils/rq2_analysis_template'):
        f=open(prompt_path,'r')
        line_list=f.readlines()
        f.close()
        user_prompt=''.join(line_list)
        user_prompt=user_prompt.replace('##CODE##',code)
        user_prompt=user_prompt.replace('##REF##',reference)
        return [
                {"role": "user", "content": user_prompt}
                ]
    
    csv_save_path=os.path.join(save_dir,f'rq2_label.csv')
    pkl_save_path=os.path.join(save_dir,f'rq2_label.pkl')
    if not os.path.exists(pkl_save_path):
        save_dict={}
    else:
        with open(pkl_save_path, 'rb') as f:
            save_dict = pickle.load(f)
    for c in trange(len(case_list)):
        case=case_list[c]
        if case in save_dict.keys():
            continue
        save_dict[case]={}
        model=case.split('==')[0]
        output_save_dir=os.path.join(raw_output_dir,model)
        result_log_pkl=os.path.join(output_save_dir,'result.pkl')
        with open(result_log_pkl, 'rb') as f:
            result_log_dict = pickle.load(f)
        total_llm_path=os.path.join(output_save_dir,'result_metric-llm-total.pkl')
        with open(total_llm_path, 'rb') as f:
            llm_evaluate_dict = pickle.load(f)
        case_key='=='.join(case.split('==')[1:])
        text=llm_evaluate_dict[case_key]['LLM Eval']
        lines = text.splitlines()
        lines = [_l for _l in lines if _l!='']
        text = '\n'.join(lines)
        num=case.split('-')[-1]
        with open(result_log_dict[case_key], 'r') as json_file:
            tmp_json_dict=json.load(json_file)
        code_response=tmp_json_dict[num]
        llm_code=extract_longest_code(code_response)
        if len(llm_code)>=30000: # avoid `out of context window`
            llm_code=llm_code[:30000]
        message=_generate_annotation(llm_code,text)
        llm_response=query_llm(None,None,sleep=1,message_text=message,version='o1-mini-2024-09-12')
        save_dict[case]=llm_response
    
        with open(pkl_save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        with open(csv_save_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([case, llm_response])


def update_inexecutable(label_results,invalid_key_path='./2.1_syntax_invalid_key-all.pkl'):
    for case in label_results:
        model=case.split('==')[0]
        key='=='.join(case.split('==')[1:])
        with open(invalid_key_path, 'rb') as f:
            invalid_keys = pickle.load(f)
        if key in invalid_keys[model]:
            if 0 not in label_results[case]:
                label_results[case].append(1)
    return label_results

def get_pattern_distribution(label_result,pattern_list):
    distribution_dict={}
    for key in label_result.keys():
        for num in label_result[key]:
            if num not in distribution_dict.keys():
                distribution_dict[num]=[]
            distribution_dict[num].append(key)
    distribution_dict={i:distribution_dict[i] if i in distribution_dict else [] for i in range(len(pattern_list))}
    print('=======')
    for num in distribution_dict.keys():
        print('Pattern {}-{}: {} ({})'.format(num,pattern_list[num],len(distribution_dict[num]),len(distribution_dict[num])/len(label_result)*100))
    return distribution_dict

def draw_rq2_stackedbar(value_list,bar_value_list,label_list,prompt_list,fig_path,intervel=300,skip_least=None,white_outline=False,text_font=28,label_font=28, legend_font=28, figsize=(10,10)):
    plt.figure(figsize=figsize)  # Set the figure size
    ax = plt.gca()
    color_list=sb.color_palette("tab20", 2*len(bar_value_list))
    color_list=[(*_cl,0.5) for _cl in color_list]
    color_list=[color_list[2*i] for i in range(int(len(color_list)/2))]
    x = np.arange(len(value_list))  # Positions for the bars
    bar_bottom = np.zeros(len(value_list))  # Start at zero for stacking
    pre = np.zeros(len(value_list))  # Start at zero for stacking

    # for i, component in enumerate(dict_list[0].keys()):
    for i, bar_value in enumerate(bar_value_list):
        # Extract the component values for each bar
        segment_values = bar_value#[composition[component] if component in composition.keys() else 0 for composition in dict_list ]
        plt.bar(x, segment_values, bottom=bar_bottom, color=color_list[i], label=prompt_list[i])
        # ,alpha=0.8
        for j, val in enumerate(segment_values):
            if val==0:continue
            if i!=0 and pre[j]+intervel>bar_bottom[j] + val / 2:
                space=pre[j]+intervel
            else:
                space=bar_bottom[j] + val / 2 
            if skip_least!=None and val/value_list[j]<skip_least:
                print(val/value_list[j])
                pass
            else:#.replace('%','0%')
                plt.text(x[j],  space, f"{val/value_list[j]:.2%}", ha='center', va='center', fontsize=24)
            pre[j]=space
        bar_bottom += segment_values  # Update the bottom for the next segment
    
    for i, total in enumerate(value_list):
        plt.text(
            x[i], 
            bar_bottom[i] + 5,  # Position slightly above the top
            f"{total:,}", 
            ha='center', 
            va='bottom', 
            fontsize=text_font, 
            # fontweight='bold'
        )
    if white_outline:
        for text in ax.texts:  # Access all text objects in the plot
            text.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),  # White outline
                path_effects.Normal()  # Normal text inside
            ])
    # Adjust tick label font size
    plt.ylabel("Number of Cases", fontsize=label_font)
    plt.xlabel("Model", fontsize=label_font)
    plt.xticks(x, [label_list[i] for i in range(len(value_list))], fontsize=label_font, rotation=30)
    plt.yticks(fontsize=label_font)
    # plt.ylim(0,16)
    plt.legend(fontsize=legend_font,ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.25),columnspacing=1)
    plt.subplots_adjust(top=0.85, bottom=0.23,right=0.99,left=0.11)
    if '.pdf' in fig_path:
        plt.savefig(fig_path,dpi=800)
    else:
        plt.savefig(fig_path)

def get_popularity_diff(pairs, popularity, lib_sce_score_dict):
    output_dict={}
    for model in lib_sce_score_dict.keys():
        tmp_score_dict={}
        output_dict[model]={}
        for sce in lib_sce_score_dict[model].keys():
            for lib,score in lib_sce_score_dict[model][sce].items():
                if lib in tmp_score_dict.keys():
                    print(1)
                tmp_score_dict[lib]=score
        for pr in pairs:
            # if abs(tmp_score_dict[pr[0]]-tmp_score_dict[pr[1]])<5:
            #     continue
            # if min((popularity[pr[0]],popularity[pr[1]]))<400:
            #     continue
            if (tmp_score_dict[pr[0]]>=tmp_score_dict[pr[1]] and not (popularity[pr[0]]>=popularity[pr[1]])) or (tmp_score_dict[pr[1]]>=tmp_score_dict[pr[0]] and not (popularity[pr[1]]>=popularity[pr[0]])):
                tmp_key=f'{pr[0]}=={pr[1]}'
                if popularity[pr[1]]!=0:
                    output_dict[model][tmp_key]=[tmp_score_dict[pr[0]]-tmp_score_dict[pr[1]],popularity[pr[0]]/popularity[pr[1]]]
                else:
                    output_dict[model][tmp_key]=[tmp_score_dict[pr[0]]-tmp_score_dict[pr[1]],0]
    return output_dict