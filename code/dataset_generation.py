import pandas as pd
import os
import csv
import pickle as pkl
import copy
from tqdm import trange
from ppl_utils.utils import *


if __name__ == "__main__":
    prompt_path='../dataset/0_dataset.pkl'

    # scenario_dict=get_scenario_description(input_file='./0_scenarios.pkl',count=5)
    with open('../dataset/0_scenarios.pkl', 'rb') as f:#input,bug type,params
        scenario_dict = pickle.load(f)
    scenario_keys=list(scenario_dict.keys())


    # Generate Prompts
    prompt_dict={}
    template_path='./ppl_utils/generation_prompt'
    template=read_prompt(template_path)
    for t in trange(len(scenario_keys)):
        raw_scenario=scenario_keys[t]
        scenario,sub_scen=raw_scenario.split('--')
        if sub_scen=='':
            scenario_des=f'`{scenario}`'
        else:
            scenario_des=f'`{sub_scen}` in the domain of `{scenario}`'

        print(f'============start {raw_scenario}=============')
        task_num=len(scenario_dict[raw_scenario]['Coding Tasks'])
        for tnum in range(task_num):
            task_name=list(scenario_dict[raw_scenario]['Coding Tasks'].keys())[tnum]
            description=list(scenario_dict[raw_scenario]['Coding Tasks'].values())[tnum]

            for svc, pvd_info in scenario_dict[raw_scenario]['Service'].items():
                if scenario_dict[raw_scenario]['Service'][svc]['Provider']['Provider'].lower()==svc.lower():
                    pvd='the'
                else:
                    pvd='{}\'s'.format(scenario_dict[raw_scenario]['Service'][svc]['Provider']['Provider'])
                prompt_key=f'{raw_scenario}=={tnum}=={svc}'#-{args.add_setting}

                # generate prompt based on each scenario
                message=copy.deepcopy(template)
                # message[1]['content']=message[1]['content'].replace('**SCENARIO**',scenario_des)
                message[1]['content']=message[1]['content'].replace('**PROVIDER**',pvd)
                message[1]['content']=message[1]['content'].replace('**SERVICE**',svc)
                message[1]['content']=message[1]['content'].replace('**DESCRIPTION**',str(description))
                prompt_dict[prompt_key]=message
    with open(prompt_path, 'wb') as f:
        pickle.dump(prompt_dict, f)
    print('finish')