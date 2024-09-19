import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import (
    Adafactor,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
import torch
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
from Datasets import CustomDataset, PretrainDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data import DataLoader, ConcatDataset
from rouge import Rouge
from collections import Counter
from torch.optim import Optimizer

import re
import string
import copy
import os
import random
import csv
import math

# from deepspeed.runtime.lr_schedules import WarmupLR
from torch.optim.lr_scheduler import LambdaLR
import deepspeed

from models.T5_Model_Kadapter import T5ForConditionalGeneration as T5_Kadapter
from models.T5_Model_Expert import T5ForConditionalGeneration as T5_Expert
from models.T5_Model_LoRA import T5ForConditionalGeneration as T5_Lora
from models.T5_Model_Modular import T5ForConditionalGeneration as T5_Modular
from models.RecAdam import RecAdam

# deepspeed lr scheduler
WARMUP_MIN_LR = 'warmup_min_lr'
WARMUP_MAX_LR = 'warmup_max_lr'
WARMUP_NUM_STEPS = 'warmup_num_steps'
WARMUP_TYPE = 'warmup_type'
WARMUP_LOG_RATE = 'log'
WARMUP_LINEAR_RATE = 'linear'

class T5(pl.LightningModule):
    def __init__(self, hparams):
        super(T5, self).__init__()
        self.save_hyperparameters(hparams)
        self.grad_norms_dict = {}
        self.grad_idx = 0
        
        if hparams.method=='baseline':
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path) # google/t5-large-ssm
        elif hparams.method=='kadapter':
            self.model = T5_Kadapter.from_pretrained(hparams.model_name_or_path)
            if hparams.mode != 'finetune_qa':
                self.freeze_params(self.model.get_encoder()) #Freezing the encoder
                for name, param in self.model.named_parameters():
                    if 'kadapter' in name:
                        param.requires_grad = True
        elif hparams.method=='lora': #Freezing the encoder
            self.model = T5_Lora.from_pretrained(hparams.model_name_or_path)
            if hparams.mode != 'finetune_qa':
                self.freeze_params(self.model.get_encoder()) 
                for name, param in self.model.named_parameters():
                    if 'lora' in name:
                        param.requires_grad = True
        elif hparams.method=='modular': #Freezing the encoder
            self.model = T5_Modular.from_pretrained(hparams.model_name_or_path)
            if hparams.mode != 'finetune_qa':
                self.freeze_params(self.model.get_encoder()) 
                for name, param in self.model.named_parameters():
                    if 'encoder_modular' in name:
                        param.requires_grad = True
        elif hparams.method=='expert': 
            self.model = T5_Expert.from_pretrained(hparams.model_name_or_path)
            if hparams.mode != 'finetune_qa':
                self.freeze_params(self.model) 
            for name, param in self.model.named_parameters():
                    if 'adapter_controller' in name:
                        param.requires_grad = True
        elif hparams.method=='recadam':
            self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
            self.pretrained_model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path, ignore_mismatched_sizes=True)
            self.freeze_params(self.pretrained_model) #Freezing pretrained model
        else:
            raise Exception('Currently not supporting {hparams.method}')
        
        self.tokenizer = T5Tokenizer.from_pretrained(hparams.tokenizer_name_or_path) 
        self.output_dir = self.hparams.output_dir
        self.global_epoch=0
        self.log('global_epoch', self.global_epoch, prog_bar=True, logger=True)
        
    def freeze_params(self, model):
        for par in model.parameters():
            par.requires_grad = False
            
    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()
        
        def rid_of_specials(text):
            text = text.replace("<extra_id_0>", "")
            text = text.replace("<extra_id_1>", "")
            text = text.replace("<extra_id_2>", "")
            text = text.replace("<extra_id_3>", "")
            return text

        return rid_of_specials(white_space_fix(remove_articles(remove_punc(lower(s)))))

    def exact_match_score(self, prediction, ground_truth):
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))
    
    def accuracy_match_score(self, prediction, ground_truth):
        return int(prediction.strip() == ground_truth.strip())

    def _f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def calculate_scores(self, predictions, ground_truths):
        em_score = 0
        accuracy = 0
        
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            em_score +=  self.exact_match_score(prediction, ground_truth)
            accuracy += self.accuracy_match_score(prediction, ground_truth)
        
        em_score /= len(predictions)
        accuracy /= len(predictions)
        return em_score*100, accuracy*100

    def calculate_f1_scores(self, predictions, ground_truths):
        f1_score = 0 
        for i in range(len(predictions)):
            ground_truth = ground_truths[i]
            prediction = predictions[i]
            f1_score += self._f1_score(prediction, ground_truth)

        f1_score /= len(predictions)
        return f1_score*100

    def get_dataset(self, tokenizer, type_path, args, length=None):
        dataset = CustomDataset(tokenizer=tokenizer, type_path=type_path, input_length=args.max_input_length, 
                        output_length=args.max_output_length, args=args, length=length)
        return dataset
             
    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))
    

    def is_logger(self):
        return self.trainer.global_rank <= 0
    
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
    )

    def on_after_backward(self):
        grads = 0
        # get grad norm here
        # for k, v in self.named_parameters():
        #     grads += v.grad.norm(p='fro').item()
        # self.grad_norms_dict[self.grad_idx] = grads
        # self.grad_idx += 1

    def _step(self, batch):
        # print('\n\n')
        # print(batch['source_ids'].shape)
        # print(self.tokenizer.decode(batch['source_ids'][0]))
        # print('\n\n')

        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]
        return loss
    
    
    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)
    
     
    def _generative_step(self, batch, batch_idx):     
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            decoder_attention_mask=batch['target_mask'],
            max_length=10,
            num_beams=2,
            early_stopping=True
        )
        
        preds = self.ids_to_clean_text(generated_ids)
        targets = self.ids_to_clean_text(batch["target_ids"])
        source = self.ids_to_clean_text(batch["source_ids"])
        loss = self._step(batch)

        em_score = 0
        accuracy = 0
        f1_score = 0

        em_score, accuracy = self.calculate_scores(preds, targets)
        f1_score = self.calculate_f1_scores(preds, targets)

        em_score = torch.tensor(em_score,dtype=torch.float32)
        accuracy = torch.tensor(accuracy,dtype=torch.float32)
        f1_score = torch.tensor(f1_score, dtype=torch.float32)

        if (batch_idx < (10000//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('UnL_loss', loss, prog_bar=True, logger=True)
            self.log('UnL_EM', em_score, prog_bar=True, logger=True)
            self.log('UnL_F1', f1_score, prog_bar=True, logger=True)
        elif (batch_idx < (10785//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('UL_loss', loss, prog_bar=True, logger=True)
            self.log('UL_EM', em_score, prog_bar=True, logger=True)
            self.log('UL_F1', f1_score, prog_bar=True, logger=True)
        elif (batch_idx < (12329//(self.hparams.eval_batch_size * self.hparams.n_gpu))):
            self.log('NL_loss', loss, prog_bar=True, logger=True)
            self.log('NL_EM', em_score, prog_bar=True, logger=True)
            self.log('NL_F1', f1_score, prog_bar=True, logger=True)
        else:
            self.log('IL_loss', loss, prog_bar=True, logger=True)
            self.log('IL_EM', em_score, prog_bar=True, logger=True)
            self.log('IL_F1', f1_score, prog_bar=True, logger=True)

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return self._generative_step(batch, batch_idx)

    def configure_optimizers(self, train_len=None):

        "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.method=='recadam':
            no_decay = ["bias", "LayerNorm.weight"]
            model_type = 't5'
            recadam_anneal_w = 1.0
            recadam_anneal_fun = 'sigmoid'
            recadam_anneal_k = 0.5
            recadam_anneal_t0 = 250
            recadam_pretrain_cof = 5000.0
            new_model = self.model
            pretrained_model = self.pretrained_model
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            not any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": self.hparams.weight_decay,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        not any(nd in p_n for nd in no_decay) and model_type not in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type in n],
                    "weight_decay": 0.0,
                    "anneal_w": recadam_anneal_w,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type in p_n]
                },
                {
                    "params": [p for n, p in new_model.named_parameters() if
                            any(nd in n for nd in no_decay) and model_type not in n],
                    "weight_decay": 0.0,
                    "anneal_w": 0.0,
                    "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                        any(nd in p_n for nd in no_decay) and model_type not in p_n]
                }
            ]
            optimizer = RecAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon,
                                anneal_fun=recadam_anneal_fun, anneal_k=recadam_anneal_k,
                                anneal_t0=recadam_anneal_t0, pretrain_cof=recadam_pretrain_cof)
        else:
            model = self.model
            
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            if self.hparams.accelerator is not None:
                optimizer = deepspeed.ops.adam.FusedAdam(optimizer_grouped_parameters, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
            else: 
                optimizer = Adafactor(optimizer_grouped_parameters, lr=self.hparams.learning_rate, scale_parameter=False, relative_step=False)

            if self.hparams.use_lr_scheduling:
                if self.hparams.len_data==None:
                    len_data = len(self.train_dataloader())
                else:
                    len_data = int(self.hparams.len_data // self.hparams.train_batch_size)
                denomniator = (self.hparams.n_gpu * self.hparams.gradient_accumulation_steps)

                steps_per_epoch = ( len_data // denomniator ) + 1
                schedule_scale_factor = 6
                total_num_steps = ( steps_per_epoch * self.hparams.num_train_epochs ) * schedule_scale_factor

                print(f'total number of steps : {total_num_steps}')
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, steps_per_epoch=steps_per_epoch, pct_start=0.1, epochs=self.hparams.num_train_epochs, anneal_strategy=self.hparams.use_lr_scheduling, cycle_momentum=False)
                return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "learning rate"}]
            else:
                return [optimizer]
    
    def on_train_epoch_end(self):
        if self.hparams.mode=='pretrain_brute_':
            self.dataset_index+=1
            if self.dataset_index==self.hparams.num_files:
                self.global_epoch+=1
                self.log('global_epoch', self.global_epoch, prog_bar=True, logger=True)
                self.dataset_index=0
            self.train_dataloader()
        elif self.hparams.mode == 'pretrain':
            if self.hparams.method=='recadam':
                self.pretrained_model = self.model
            elif self.hparams.method=='kadapter' or self.hparams.method=='lora' or self.hparams.method=='modular':
                self.model.save_pretrained(self.hparams.output_dir)

    def train_dataloader(self): 
        if self.hparams.mode=='pretrain_brute':
            train_dataset = PretrainDataset(tokenizer=self.tokenizer, input_length=self.hparams.max_input_length, output_length=self.hparams.max_output_length, args=self.hparams)
        else:
            train_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
            print(self.tokenizer.decode(train_dataset[0]['source_ids']))
        sampler = SequentialSampler(train_dataset)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, num_workers=self.hparams.num_workers)
        #dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, num_workers=self.hparams.num_workers)
        return dataloader

    # def val_dataloader(self):
    #     validation_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="validation", args=self.hparams,)
    #     return DataLoader(validation_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers, shuffle=False)
    
    # def test_dataloader(self):
    #     test_dataset = self.get_dataset(tokenizer=self.tokenizer, type_path="test", args=self.hparams)
        