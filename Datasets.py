from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import DPRContextEncoderTokenizerFast, DPRContextEncoder, DPRQuestionEncoderTokenizerFast, DPRQuestionEncoder
import pandas as pd
import numpy as np
import json


class PretrainDataset(Dataset):
    def __init__(self, tokenizer, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.file_path = self.args.dataset
        self.dataset = dataset = load_dataset("csv", data_files=self.file_path)
        # print(f'Loaded Dataset from {self.args.dataset}')
        # print(f'Getting initial pretraining dataset with length {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        length = 9803521 # len(self.dataset)
        return length

    def convert_to_features(self, example_batch, index=None):
        input_ = example_batch['input']
        target_ = example_batch['output']

        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")      
        
        return source, targets

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset['train'][index]) 
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}


class DPRDataset(Dataset):
    def __init__(self, tokenizer, type_path, input_length, output_length, args):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path #'train'
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_version = self.args.dataset_version

        if 'unchanged' in self.args.dataset: 
            dataset_name = 'unchanged_eval_random'
        else:
            dataset_name = self.args.dataset

        dataset = pd.read_csv(f'data/finetune_qa/{dataset_name}.csv')
        self.dataset = dataset.sample(frac=1).reset_index(drop=True).iloc[:500]

        # passage
        self.context_embedding = np.load('dpr_embeddings/unchanged_all.npy') # unchanged set
        with open("./dpr_embeddings/context_dict.json", "r") as f:
            self.context_text = json.load(f) 

        month = self.args.dataset.split('_')[1]

        self.changed_context_embedding = np.load(f'dpr_embeddings/changed_{month}.npy') # changed set
        with open(f"./dpr_embeddings/context_{month}_dict.json", "r") as f:
            print(f"./dpr_embeddings/context_{month}_dict.json")
            self.changed_context_text = json.load(f)
        self.changed_context_text = dict((str(int(key)+15605153), value) for (key, value) in self.changed_context_text.items())

        self.context_text.update(self.changed_context_text)
        self.context_embedding = np.concatenate((self.context_embedding, self.changed_context_embedding), axis=0)
        assert len(self.context_text.keys()) == self.context_embedding.shape[0]

        self.question_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')


    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        input_nonprompt = None
        q = self.question_tokenizer(example_batch['question'], max_length=200, truncation=True, padding="max_length", return_tensors='pt')
        q = self.question_encoder(**q).pooler_output.to('cpu').detach().numpy()
        similarity_matrix = np.matmul(q, self.context_embedding.T)
        argmax_indices = (-similarity_matrix).argsort(axis=1)
        max_idx = argmax_indices[0][0]
        retrieved_context = self.context_text[str(max_idx)]

        input_ = 'context: ' + retrieved_context + 'question: '+ example_batch['question']

        if 'edited' in self.args.dataset:
            answer_name = 'answer2'
        else:
            answer_name = "answer"
        
        target_ = example_batch[answer_name]
        label_ = example_batch[answer_name]

        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        label_ = self.tokenizer.batch_encode_plus([str(label_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")        
        return source, targets, input_nonprompt, label_
  
    def __getitem__(self, index):
        source, targets, input_nonprompt, label = self.convert_to_features(self.dataset.iloc[index]) 
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if input_nonprompt is not None:
            source_nonprompt_ids = input_nonprompt["input_ids"].squeeze()
            source_nonprompt_mask = input_nonprompt["attention_mask"].squeeze()
        else: 
            source_nonprompt_mask = -1
            source_nonprompt_ids = -1

        if label is not None:
            label_ids = label["input_ids"].squeeze()
            label_mask = label["attention_mask"].squeeze()
        else: 
            label_ids = -1
            label_mask = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "source_nonprompt_ids" : source_nonprompt_ids, "source_nonprompt_mask": source_nonprompt_mask, "label_ids": label_ids, "label_mask": label_mask}

class CustomDataset(Dataset):
    def __init__(self, tokenizer, type_path, input_length, output_length, args, length=None, lama_type=None):
        self.args = args
        self.tokenizer = tokenizer
        self.type_path = type_path #'train'
        self.dataset_version = self.args.dataset_version
                
        # import pdb; pdb.set_trace()

        self.lama_type = lama_type # for validation, it decided 'unchanged' or 'changed'
        dataset_v = ['small', 'full']   
            
        if not self.dataset_version in dataset_v:
            raise Exception(f'Provided the correct dataset version among {dataset_v}')

        # dataset for continual training
        if self.type_path=='train':
            if self.args.mode == 'pretrain':
                if 't5' in self.args.dataset:
                    self.dataset = pd.read_csv(f'data/t5/twiki_diffset/{self.args.dataset}.csv')
                else:
                    raise Exception('The given dataset does not exist in data directory.')
                    
            elif self.args.mode == 'finetune_qa' or self.args.mode == 'finetune_qa_prompt' or self.args.mode == 'finetune_qa_dpr':
                self.dataset = pd.read_csv('data/finetune_qa/unchanged_train.csv')

            elif self.args.mode == 'finetune_qa_multiple':
                self.dataset = pd.read_csv('data/finetune_qa/unchanged_train_multiple.csv')

            elif self.args.mode == 'check_grad_edited':
                self.dataset = pd.read_csv(f'data/finetune_qa/edited_{self.args.dataset}.csv')
            elif self.args.mode == 'check_grad_new':
                self.dataset = pd.read_csv(f'data/finetune_qa/new_{self.args.dataset}.csv')
            else: 
                raise Exception("Inaccurate Trainset Name!")
        
        else:
            if self.args.mode == 'finetune_qa' or self.args.mode == 'finetune_qa_prompt': # test
                if 'unchanged' in self.args.dataset: 
                    self.dataset = pd.read_csv('data/finetune_qa/unchanged_eval.csv') 
                else: # edited, new
                    self.dataset = pd.read_csv(f'data/finetune_qa/{self.args.dataset}.csv')
                    print("Retrieving ... ", f'data/finetune_qa/{self.args.dataset}.csv')
                    
            elif self.args.mode == 'finetune_qa_multiple':
                self.dataset = pd.read_csv(f'data/finetune_qa/{self.args.dataset}.csv')
            else: 
                raise NotImplementedError()
                        
        print(f'Length of dataset retrieving is.. {len(self.dataset)}')
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.dataset)

    def convert_to_features(self, example_batch, index=None):
        input_nonprompt = None
        label_ = None
        if self.type_path=='validation':
            if self.args.mode == 'evaluate_ppl_corpus':
                input_ = example_batch['text']
                target_ = example_batch['text']
            elif 'finetune_qa' in self.args.mode and 'edited' in self.args.dataset: 
                if self.args.mode == 'finetune_qa_prompt':
                    input_ = example_batch['question_time']
                else:
                    input_ = example_batch['question']
                target_ = example_batch['answer2']
                label_ = example_batch['answer2']
            elif self.args.mode == 'finetune_qa' or self.args.mode == 'finetune_qa_dpr': 
                input_ = example_batch['question']
                target_ = example_batch['answer']
                label_ = example_batch['answer']
            elif self.args.mode == 'finetune_qa_prompt': 
                input_ = example_batch['question_time']
                target_ = example_batch['answer']
                label_ = example_batch['answer']
            else: 
                s = example_batch['subject']
                r = example_batch['relation']
                o = example_batch['object']
                if self.args.mode == 'evaluate_ppl':
                    input_ = s + ' ' + r + ' ' + o
                    input_nonprompt =  ' ' + o 
                    target_ = s + ' ' + r + ' ' + o 
                elif self.args.mode == 'evaluate':
                    input_ = s + ' ' + r
                    target_ = o
                elif self.args.mode == 'finetune':
                    label_ = s + ' ' + r + ' ' + o 
                    input_ = s + ' ' + r 
                    target_ = o
                else: 
                    target_ = s + ' ' + r + ' ' + o 
                    input_ = s + ' ' + r + ' ' + o 
                    input_nonprompt = ' ' + o 
        else: # train
            if self.args.mode == 'finetune':
                s = example_batch['subject']
                r = example_batch['relation']
                o = example_batch['object']  
                input_ = s + ' ' + r + ' ' + o 
                target_ = s + ' ' + r + ' ' + o 
                label_ = s + ' ' + r + ' ' + o
            elif self.args.mode == 'finetune_qa' or self.args.mode == 'finetune_qa_multiple':
                input_ = example_batch['question']
                target_ = example_batch['answer']
                    
            elif self.args.mode == 'finetune_qa_prompt':
                input_ = example_batch['question_time']
                target_ = example_batch['answer']

            elif self.args.mode == 'finetune_qa_dpr':
                context = str(example_batch['text'])
                question = example_batch['question']
                answer = example_batch['answer']

                ans_idx = context.find(answer)
                start_idx = max(0, ans_idx-200)
                end_idx = min(len(context), ans_idx+200)
                context = context[start_idx: end_idx]

                input_ = 'context: ' + context + 'question: '+ question
                target_ = answer

            elif self.args.mode == 'check_grad_new':
                input_ = example_batch['input']
                target_ = example_batch['output']

            elif self.args.mode == 'check_grad_edited':
                input_ = example_batch['input']
                target_ = example_batch['output']

            else: # pretrain
                input_ = example_batch['input']
                target_ = example_batch['output']

        source = self.tokenizer.batch_encode_plus([str(input_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")
        targets = self.tokenizer.batch_encode_plus([str(target_)], max_length=self.output_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        if input_nonprompt is not None:
            input_nonprompt = self.tokenizer.batch_encode_plus([str(input_nonprompt)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt") 
        if label_ is not None:
            label_ = self.tokenizer.batch_encode_plus([str(label_)], max_length=self.input_length, 
                                                    padding='max_length', truncation=True, return_tensors="pt")        
        
        return source, targets, input_nonprompt, label_
  
    def __getitem__(self, index):
        source, targets, input_nonprompt, label = self.convert_to_features(self.dataset.iloc[index]) 
        
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()

        src_mask    = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        if input_nonprompt is not None:
            source_nonprompt_ids = input_nonprompt["input_ids"].squeeze()
            source_nonprompt_mask = input_nonprompt["attention_mask"].squeeze()
        else: 
            source_nonprompt_mask = -1
            source_nonprompt_ids = -1
        
        if label is not None:
            label_ids = label["input_ids"].squeeze()
            label_mask = label["attention_mask"].squeeze()
        else: 
            label_ids = -1
            label_mask = -1

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask, "source_nonprompt_ids" : source_nonprompt_ids, "source_nonprompt_mask": source_nonprompt_mask, "label_ids": label_ids, "label_mask": label_mask}