from Datasets import CustomDataset
from Datasets import DPRDataset
from torch.utils.data import DataLoader
import csv
import pandas as pd
import os
import torch
from tqdm.notebook import tqdm
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/t5-large-ssm") 

def evaluate(args, model):
    # model = Model(args)
    # if args.checkpoint_path!="":
    #     model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)

    model.eval()
    model.to('cuda')
    tokenizer = model.tokenizer
    #Get Validation Data
    if args.mode=='pretrain' or args.mode=='finetune' or args.mode=='evaluate' or args.mode=='finetune_qa' or args.mode=='finetune_qa_prompt':
        dataset = CustomDataset(tokenizer, 'validation', input_length=args.max_input_length, 
                        output_length=args.max_output_length, args=args)
        print(tokenizer.decode(dataset[0]['source_ids']))
    elif args.mode == 'dpr':
        dataset = DPRDataset(tokenizer, 'validation', input_length=args.max_input_length, 
                    output_length=args.max_output_length, args=args)
    else:
        raise Exception('Select the correct mode please.')
        
    print(f'Length of {args.dataset} data: ',len(dataset))
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
    
    total_cnt = 0
    em_correct_num = 0
    old_em_correct_num = 0
    new_em_correct_num = 0
    accuracy_correct_num = 0
    f1_score = 0

    def clean_up(text): # only for questions
        text =text.replace('<pad>', '')
        text = text.replace('</s>', '')
        text = text.replace(".", '')
        text = text.replace(',', '')
        text = text.replace("'", '')
        text = text.replace('"', '')
        return text   
    # If folder doesn't exist, then create it.
    if args.mode == 'finetune_qa_prompt':
        output_log = os.path.join(args.output_dir, 'log_prompt.csv')
    else:
        output_log = os.path.join(args.output_dir, 'log.csv')
        print(f'Saving to ... {output_log}')
    MYDIR = ("/".join((output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print()
        # print(MYDIR, "folder already exists.")

    total_questions = []
    total_targets = []
    total_preds = []
    total_results = []

    for batch in iter(loader):
        outs = model.model.generate(
            batch["source_ids"].cuda(),
            attention_mask=batch["source_mask"].cuda(),
            use_cache=True,
            decoder_attention_mask=batch['target_mask'].cuda(),
            max_length=args.max_output_length,
            num_beams=1,
            early_stopping=True,
            # return_dict_in_generate=True, 
            # output_scores=True
        )
        dec = model.ids_to_clean_text(outs) # batch_decide
        questions = [tokenizer.decode(ids, clean_up_tokenization_spaces=False, skip_special_tokens=True) for ids in batch['source_ids']]
        targets = model.ids_to_clean_text(batch['target_ids'])
            # print("preds",dec)
            # print("targets",targets)
            
        for i in range(len(batch['source_ids'])):
            total_cnt+=1
            # if total_cnt % 500 == 0 : print(f"processed {total_cnt}")
            lines = questions[i]
            ground_truth = targets[i] # lower>rm_punc>rm_a_the>
            predicted = dec[i]
            # print("prediction:",total_cnt,predicted)

            em = model.exact_match_score(predicted, ground_truth)  
            f1_score += model._f1_score(predicted, ground_truth)
            
            total_questions.append(lines)
            total_targets.append(ground_truth)
            total_preds.append(predicted)
            
            if em == 1: # if correct
                total_results.append("correct")
                em_correct_num+=1
            else: 
                total_results.append("wrong")
            

    print(f"------------------Result on {args.dataset}------------------")
    print(f'Number of total validation data: {total_cnt}')


    print(f'Number of correct predictions (EM): {em_correct_num}. Percentage : {round(em_correct_num / total_cnt * 100, 2)}%')
    print(f'F1 score is {round(f1_score / total_cnt * 100, 2)}%')
    print(f"--------------------------------------------------------\n")


    total_questions.append(str(em_correct_num) + '/' + str(total_cnt))
    total_targets.append(str(em_correct_num / total_cnt))
    total_preds.append(str(f1_score / total_cnt))
    total_results.append("")

    pd.DataFrame({"question":total_questions, "target":total_targets, "prediction":total_preds, "correctness": total_results}).to_csv(output_log, index=False)
