import math
from datetime import datetime
import pytz
import os
import torch
import numpy as np
import copy
from optimizer import initialize_optimizer
from torch.nn import functional as F



def test_zs(cfg, model, datasets):
    
        
    if cfg.model == 'zsclip_twice':
        if cfg.score_mode == 'sparse':
            if cfg.logist_topk > 0:
                out_saved_file_name = f"tau={cfg.tau}_logist_topk={cfg.logist_topk}_seed={cfg.seed}.txt"
            else:
                out_saved_file_name = f"tau={cfg.tau}_seed={cfg.seed}.txt"

            if (cfg.mask_mode == 'weight_n_end') or (cfg.mask_mode == 'weight_n_end_token_purning'):
                out_saved_file_name = f"tau={cfg.tau}_weight_{cfg.weight_n_layer}_end-layer_seed={cfg.seed}.txt"
        else:
            assert False 
            
        
        folder_path = os.path.join(cfg.output_dir, cfg.dataset, cfg.backbone, cfg.model,  cfg.score_mode, cfg.mask_mode, f'help={cfg.help_prompt}')
    

    elif cfg.model == 'zsclip':
        out_saved_file_name = f"bs={cfg.batch_size}_seed={cfg.seed}.txt"
        folder_path = os.path.join(cfg.output_dir, cfg.dataset, cfg.backbone, cfg.model)
    

    if cfg.logist_topk > 0:
        folder_path = os.path.join(folder_path, f'logist_topk={cfg.logist_topk}')




    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file = os.path.join(folder_path, out_saved_file_name)

    print(f"save to: {file}")
    
    tz = pytz.timezone('Asia/Shanghai')
    with open(file ,"a", encoding="utf-8") as f:
        f.write(f"\n\n ==== time info===  {datetime.now(tz)}\n")
            
   
    # # test
    model.eval()

    group_num = cfg.num_groups


    accuracy = [0] * group_num
    correct = [0] * group_num
    total = [0] * group_num

    with torch.no_grad(): 
        for test_batch in datasets['test']:
            
            if (cfg.mask_mode == 'img_token'):
                x = test_batch[5].to(cfg.device).float()
            else:
                x = test_batch[0].to(cfg.device).float()   # [batch size, 3, 224, 224]  x

            y = test_batch[1].to(cfg.device)    # [batch size] y
            g = test_batch[3].to(cfg.device)  # [batch size] group
            
            outputs = model(x)

            _, predicted = torch.max(outputs.data, 1)
               
            for i in range(group_num):
                total[i] += (test_batch[3] == i).sum().item()
                correct[i] += ((predicted == y) & (g == i)).sum().item()


    for i in range(group_num):
        if total[i] == 0:
            continue
        accuracy[i] = 1.0 * 100 * correct[i] / total[i]
    # print(accuracy)
    all_accuracy = 1.0 * 100 *sum(correct) / sum(total)
    min_accuracy = np.min(accuracy)
    min_index = accuracy.index(min_accuracy)
    
    
    with open(file ,"a", encoding="utf-8") as f:
        f.write("\n ====test result===\n\n")
        f.write(f"## prompts: {cfg.prompts}\n")
        f.write(f"total: {total}\n")
        f.write(f"correct: {correct}\n")
        f.write(f"accuracy: {accuracy}\n")
        f.write(f"global_accuracy: {all_accuracy}\n")
        f.write(f"min_accuracy: {min_accuracy}     index:{min_index}\n")
        
            
    print((f"# prompts: {cfg.prompts}"))
    print(f"total: {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy}")
    print(f"global_accuracy: {all_accuracy}")
    print(f"min_accuracy: {min_accuracy}     index:{min_index}\n")











