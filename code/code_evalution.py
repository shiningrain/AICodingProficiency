import os
import numpy as np
import time
import pickle
import argparse
import copy
from tqdm import trange
import openai
from ppl_utils.utils import *
from ppl_utils.metric import *
from ppl_utils.eval import *
import json



def evaluate_file_metrics_static(
        code_path,
        code_string,
        error_log=[]):
    """
    The code quality scores in six dimentions

    Parameters
    ----------
    code_path : Saved Code File Path
    code_string : str Generated Code Snippets
    Returns
    -------
    dict : {
        "Maintainability": [ ... ],
        "Readability": [ ... ],
    }
    """
    # ---------- Maintainability ----------
    m_log=[]
    try:
        hv_log=[]
        halstead_v,hv_log = calculate_halstead_volume(code_string)
    except:
        halstead_v=None
    m_log.append(hv_log)
    try:
        cc_log=[]
        cyclo_cc,cc_log   = calculate_cyclomatic_complexity(code_string)
    except:
        cyclo_cc=None
    m_log.append(cc_log)
    try:
        mi_log=[]
        mi_score,mi_log   = calculate_maintainability_index(code_string,halstead_v,cyclo_cc)
    except:
        mi_score=None
    m_log.append(mi_log)

    # ---------- Readability ----------
    pylint_read, r_log = linter_readability_score(code_string, code_path=code_path)
    posnett_read   = calculate_posnett_readability(code_string)

    score_dict={
        "Maintainability": [halstead_v, cyclo_cc, mi_score],
        "Readability":    [pylint_read, posnett_read],
    }
    log_dict={
        "Readability": r_log,
        "Maintainability":m_log,
        "Error":error_log
    }
    return score_dict,log_dict

def parse_args():
    parser = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-dp", "--dataset_path", default='../0_dataset.pkl', type=str, help="Input file")
    parser.add_argument("-od", "--output_dir", default='./results', type=str, help="Output file dir")
    parser.add_argument("-md", "--model", default='gemini-2.5-flash', type=str, help="mini//4o//claude-sonnet-4-20250514//gemini-2.5-flash//qwen3-coder-plus-2025-07-22//deepseek-reasoner")#DeepSeek-R1-0528
    parser.add_argument("-rp", "--repeat", default=5, type=int, help="repeat query xx times")
    parser.add_argument("-rn", "--result_pkl", default='result.pkl', type=str, help="result file name")
    parser.add_argument("-sp", "--score_path", default='result-scores.pkl', type=str, help="")
    return parser.parse_args()


if __name__ == "__main__":
    current_script_path = os.path.abspath(__file__)
    args = parse_args()
    repeat_num=args.repeat
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    print(f'Using CLM: {args.model}!!!')
    model_misc=None

    prompt_dict=copy.deepcopy(dataset[args.model])
    prompt_dict_keys=list(prompt_dict.keys())
    save_dir=os.path.join(args.output_dir,f'{args.model}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_log_pkl=os.path.join(save_dir,args.result_pkl)
    if not os.path.exists(result_log_pkl):
        result_log_dict={}
    else:
        with open(result_log_pkl, 'rb') as f:
            result_log_dict = pickle.load(f)
    
    metric_log_pkl=os.path.join(save_dir,'result_metric-new.pkl')
    if not os.path.exists(metric_log_pkl):
        eval_metric_dict={}
    else:
        with open(metric_log_pkl, 'rb') as f:
            eval_metric_dict = pickle.load(f)

    metric_llm_pkl=os.path.join(save_dir,'result_llm-new.pkl')
    if not os.path.exists(metric_llm_pkl):
        eval_llm_dict={}
    else:
        with open(metric_llm_pkl, 'rb') as f:
            eval_llm_dict = pickle.load(f)

    for key in prompt_dict.keys():
        if '/'in key:
            print(key)
    for pk in trange(len(prompt_dict_keys)):
        prompt_key=prompt_dict_keys[pk]
        message=prompt_dict[prompt_key]
        tmp_save_json=os.path.join(save_dir,prompt_key+'.json')
        result_save_path=tmp_save_json
        if os.path.exists(result_save_path):
            with open(result_save_path, 'r') as json_file:
                tmp_json_dict=json.load(json_file)
        else:
            tmp_json_dict={}
        
        # print(prompt_key)
        for r in range(repeat_num):
            result_save_path=tmp_save_json
            result_key=prompt_key+f'-{r}'
            if os.path.exists(result_save_path):
                with open(result_save_path, 'r') as json_file:
                    tmp_json_dict=json.load(json_file)
            else:
                tmp_json_dict={}
            if str(r) not in tmp_json_dict.keys() or result_log_dict[result_key]==None or tmp_json_dict[str(r)]=='':
                try:
                    print(result_key)
                    llm_response=query_llm(model_misc,None,sleep=0.2,message_text=message,version=args.model)

                except openai.BadRequestError as e: # handle refusal
                    print(e)
                    llm_response='I cannot assist with that! content filter'
                    print(f'Blocked in {result_save_path}')
                    if 'Insufficient Balance' in e:
                        break# TODO: Insufficient Balance
                except openai.APIError as e: # handle refusal
                    print(e)
                    llm_response='I cannot assist with that! content filter'
                    print(f'Blocked in {result_save_path}')
                except ValueError as e: # handle refusal
                    print(e)
                    llm_response='I cannot assist with that! Gemini content filter'
                    print(f'Blocked in {result_save_path}')
                except openai.RateLimitError as e:
                    print(e)
                    time.sleep(60)
                    llm_response=query_llm(model_misc,None,sleep=1,message_text=message,version=args.model)
                tmp_json_dict[str(r)]=llm_response
                with open(result_save_path, 'w') as json_file:
                    json.dump(tmp_json_dict, json_file, indent=4) 
            else:
                llm_response=tmp_json_dict[str(r)]

            if result_key not in result_log_dict.keys() or result_log_dict[result_key]==None:
                result_log_dict[result_key]=os.path.abspath(result_save_path)
                if 'I cannot assist with that!' in llm_response:
                    result_log_dict[result_key]=None
                    print(f"Block in {result_key}")
                with open(result_log_pkl, 'wb') as f:
                    pickle.dump(result_log_dict, f)

            #=============Static Metrics=============
            code=extract_longest_code(llm_response)
            if result_key not in eval_metric_dict.keys():
                tmp_dir=TEMP_DIR+f'_{args.model}'
                if not os.path.exists(tmp_dir):
                    os.makedirs(tmp_dir)
                file_path = os.path.join(save_dir, f"{result_key}.py")
                with open(file_path, 'w', encoding='utf-8') as fout:
                    fout.write(code)
                file_path_P=Path(file_path)

                (rt, log)=evaluate_file_metrics_static(
                    file_path_P,
                    code,
                    error_log=[]
                )
                eval_metric_dict[result_key],log_dict=rt,log
                with open(metric_log_pkl, 'wb') as f:
                    pickle.dump(eval_metric_dict, f)

                log_path=result_log_dict[result_key].replace('.json','-log.pkl')
                if not os.path.exists(log_path):
                    tmp_log={}
                else:
                    with open(log_path, 'rb') as f:
                        tmp_log = pickle.load(f)
                tmp_log[str(r)]=log_dict
                with open(log_path, 'wb') as f:
                    pickle.dump(tmp_log, f)


            #=============LLM Evaluation=============
            if result_key not in eval_llm_dict.keys():
                # if len(code)>=30000: # avoid `out of context window`
                #     code=code[:30000]

                question=prompt_dict[prompt_key][1]['content']
                message2=generate_check_list(question,code)
                llm_response2=query_llm(None,None,sleep=1,message_text=message2,version='o1-mini-2024-09-12')
                check_list=extract_check_list(llm_response2)
                message3=generate_finegrain_query(question,code,check_list)
                llm_response3=query_llm(None,None,sleep=1,message_text=message3,version='o1-mini-2024-09-12')
                eval_llm_dict[result_key]={}
                eval_llm_dict[result_key]['Check List']=check_list
                eval_llm_dict[result_key]['LLM Eval']=llm_response3
                # 
                with open(metric_llm_pkl, 'wb') as f:
                    pickle.dump(eval_llm_dict, f)

    with open(result_log_pkl, 'wb') as f:
        pickle.dump(result_log_dict, f)
    
    # Extract Scores
    raw_scores={}
    with open(result_log_pkl, 'rb') as f:
        result_log_dict = pickle.load(f)
    # update llm evaluation results
    raw_scores,_=update_llm_results(raw_scores,eval_llm_dict,None,[])
    # update static evaluation results
    eval_metric_dict=obtain_syntax_valid(eval_metric_dict,result_log_dict,None)
    raw_scores,error_list=update_static_results(raw_scores,eval_metric_dict,None,result_log_dict,[])

    for key in raw_scores.keys():
        for dim in raw_scores[key].keys():
            if 'LLM' in raw_scores[key][dim].keys():
                raw_scores[key][dim]['LLM']=raw_scores[key][dim]['LLM']*10
            if 'Static' in raw_scores[key][dim].keys():
                if dim!='Maintainability Score':
                    raw_scores[key][dim]['Static']=raw_scores[key][dim]['Static']*100

    # Extract scores for each
    score_dict={}
    for key in raw_scores.keys():
        score_dict[key]=[
            raw_scores[key]['Correctness Score']['LLM'],
            (raw_scores[key]['Performance Score-Time']['LLM']+raw_scores[key]['Performance Score-Memory']['LLM'])/2,
            raw_scores[key]['Maintainability Score']['Static'],
            raw_scores[key]['Readability Score']['Static'],
            raw_scores[key]['Reliability Score']['LLM'],
            ]
        score_dict[key].append(np.average(score_dict[key]))# the overall quality score

    with open(args.score_path, 'wb') as f:
        pickle.dump(score_dict, f)
