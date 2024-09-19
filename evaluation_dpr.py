from Datasets import DPRDataset
from torch.utils.data import DataLoader
import csv
import pandas as pd
import os
import torch
from tqdm.notebook import tqdm
from collections import Counter

def dpr_evaluate(args, model):
    # model = Model(args)
    # if args.checkpoint_path!="":
    #     model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)

    model.eval()
    model.to('cuda')
    tokenizer = model.tokenizer
    #Get Validation Data
    dataset = DPRDataset(tokenizer, 'validation', input_length=args.max_input_length, 
                    output_length=args.max_output_length, args=args)
    print(f'Length of {args.dataset} data: ',len(dataset))
    dataset_df = pd.read_csv(f'data/finetune_qa/{args.dataset}.csv')
    answer1_list = list(dataset_df['answer1'])
    answer2_list = list(dataset_df['answer2'])
    loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
    
    total_cnt = 0
    correct_old_num = 0
    correct_new_num = 0
    f1_score_old = 0
    f1_score_new = 0

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
        
    MYDIR = ("/".join((output_log.split('/'))[:-1]))
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)
    else:
        print(MYDIR, "folder already exists.")

    total_questions = []
    total_preds = []
    total_old_results = []
    total_new_results = []
    logit_outdated = 0
    logit_updated = 0
    total_logit_results = []

    batch_idx = 0
    for batch in iter(loader):
        ### T5
        start_batch_idx = batch_idx * args.train_batch_size
        end_batch_idx = (batch_idx+1) * args.train_batch_size
        answer1s = answer1_list[start_batch_idx: end_batch_idx]
        answer2s = answer2_list[start_batch_idx: end_batch_idx]

        if 't5' in args.model_name_or_path:
            outs = model.model.generate(
                batch["source_ids"].cuda(),
                attention_mask=batch["source_mask"].cuda(),
                use_cache=True,
                decoder_attention_mask=batch['target_mask'].cuda(),
                max_length=args.max_output_length,
                num_beams=1,
                early_stopping=True,
                return_dict_in_generate=True, 
                output_scores=True
            )
            logits = torch.stack(outs['scores'], dim=1)
            log_prob = torch.log_softmax(logits, dim=2)
            dec = model.ids_to_clean_text(outs['sequences'])
            questions = [tokenizer.decode(ids, clean_up_tokenization_spaces=False, skip_special_tokens=True) for ids in batch['source_ids']]
            if batch_idx == 0:
                print(questions[0])
            # print("preds",dec)
            # print("targets",targets)
            
        for i in range(len(batch['source_ids'])):
            total_cnt+=1
            # if total_cnt % 500 == 0 : print(f"processed {total_cnt}")
            sample_log_prob = log_prob[i]
            lines = questions[i]
            answer1 = answer1s[i] 
            tokenized_answer1 = [tok for tok in  tokenizer.encode(answer1) if tok!=3 and tok!=1 ][:sample_log_prob.shape[0]]
            answer2 = answer2s[i] # lower>rm_punc>rm_a_the>
            tokenized_answer2 = [tok for tok in  tokenizer.encode(answer2) if tok!=3 and tok!=1 ][:sample_log_prob.shape[0]]
            predicted = dec[i]
            
            answer1_score = sum([sample_log_prob[token_idx][token].item() for token_idx, token in enumerate(tokenized_answer1)])
            answer2_score = sum([sample_log_prob[token_idx][token].item() for token_idx, token in enumerate(tokenized_answer2)])
            # print("prediction:",total_cnt,predicted)

            if answer1_score >= answer2_score: 
                logit_outdated += 1
                total_logit_results.append('outdated')
            else:
                logit_updated += 1
                total_logit_results.append('updated')

            em_old = model.exact_match_score(predicted, answer1) 
            f1_score_old += model._f1_score(predicted, answer1)

            em_new = model.exact_match_score(predicted, answer2) 
            f1_score_new += model._f1_score(predicted, answer2)
            
            total_questions.append(lines)
            total_preds.append(predicted)
            
            if em_old == 1: # if correct
                total_old_results.append("correct")
                correct_old_num+=1
            else:   
                total_old_results.append("wrong")

            if em_new == 1: # if correct
                total_new_results.append("correct")
                correct_new_num+=1
            else:
                total_new_results.append("wrong")
            

        batch_idx +=1            

    
    print(f"\n------------------Result on {args.dataset}------------------")
    print(f'Number of total validation data: {total_cnt}')
    print(f'Number of  OLD correct (EM): {correct_old_num}. Percentage : {round(correct_old_num / total_cnt * 100, 2)}%')
    print(f'Number of  NEW correct (EM): {correct_new_num}. Percentage : {round(correct_new_num / total_cnt * 100, 2)}%')
    print(f'OLD F1 score is {round(f1_score_old / total_cnt * 100, 2)}%')
    print(f'NEW F1 score is {round(f1_score_new / total_cnt * 100, 2)}%')

    logit_count_dict = Counter(total_logit_results)
    print(f'Logit outdated count is {logit_outdated}')
    print(f'Logit updated count is {logit_updated}')
    print(f"Updated selection probability {round(logit_updated/total_cnt * 100,2)}")
    print(f"----------------------------------------------------------\n")


    total_questions.append(str(correct_old_num) + '/' + str(total_cnt))
    total_preds.append(str(correct_old_num / total_cnt))
    answer1_list.append(str(f1_score_old / total_cnt))

    answer2_list.append(str(correct_new_num) + '/' + str(total_cnt))
    total_old_results.append(str(correct_new_num / total_cnt))
    total_new_results.append(str(f1_score_new / total_cnt)) 
    total_logit_results.append(str(round(logit_updated/total_cnt * 100,2)))

    pd.DataFrame({"question":total_questions, "prediction":total_preds, "answer1": answer1_list, "answer2": answer2_list, \
        "correctness1":total_old_results, "correctness2":total_new_results, "logit_prediction":total_logit_results}).to_csv(output_log, index=False)
