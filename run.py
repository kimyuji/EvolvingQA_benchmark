import argparse
from argparse import ArgumentParser
import os
import json
import random
from evaluation import evaluate
from evaluation_edited import edited_evaluate
from evaluation_dpr import dpr_evaluate
from evaluation_ppl import evaluate_ppl
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from transformers import T5Tokenizer, GPT2Tokenizer
from models import load_model
from models.utils import get_checkpoint_size

def _rebuild_parameter_v2(data, requires_grad, backward_hooks, unknown_argument):
    # As we do not know the role of the unknown_argument, we simply ignore it here
    parameter = torch.nn.Parameter(data, requires_grad)
    parameter._backward_hooks = backward_hooks
    return parameter

torch._utils._rebuild_parameter_v2 = _rebuild_parameter_v2


from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NameError("Include a config file in the argument please.")

    #Getting configurations
    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    #Setting GPUs to use
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=hparam.CUDA_VISIBLE_DEVICES

    #Init configs that are not given
    if 'grad_norm' not in hparam:
        hparam.grad_norm = 0.5
    if 'weight_decay' not in hparam:
        hparam.weight_decay = 0.0
    if 'output_log' not in hparam:
        hparam.output_log = None
    if 'len_data' not in hparam:
        hparam.len_data = None
    if 'learning_rate' not in hparam:
        hparam.learning_rate = None
    if 'gradient_accumulation_steps' not in hparam:
        hparam.gradient_accumulation_steps = 0
    if 'num_train_epochs' not in hparam:
        hparam.num_train_epochs = 0
    if 'use_lr_scheduling' not in hparam:
        hparam.use_lr_scheduling = False
    if 'num_workers' not in hparam:
        hparam.num_workers = 0
    if 'output_dir' not in hparam:
        hparam.output_dir = ''
    if 'wandb_log' not in hparam:
        hparam.wandb_log = False
    if 'accelerator' not in hparam:
        hparam.accelerator = None
    if 'checkpoint_path' not in hparam:
        hparam.checkpoint_path =''
    if 'resume_from_checkpoint' not in hparam:
        hparam.resume_from_checkpoint = None
    if 'bf16' not in hparam:
        hparam.bf16 = False
    if 'check_grad' not in hparam: 
        hparam.check_grad = False

    #Logging into WANDB if needed
    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name, entity='yujin399_wb')
    else:
        wandb_logger = None
        
    #Setting configurations
    args_dict = dict(
        output_dir=hparam.output_dir, # Path to save the checkpoints
        dataset=hparam.dataset,
        dataset_version = hparam.dataset_version,
        len_data = hparam.len_data,
        model_name_or_path=hparam.model,
        method=hparam.method,
        mode=hparam.mode,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.input_length,
        max_output_length=hparam.output_length,
        learning_rate=hparam.learning_rate,
        weight_decay=hparam.weight_decay,
        adam_epsilon=1e-8,
        train_batch_size=hparam.train_batch_size,
        eval_batch_size=hparam.train_batch_size,
        num_train_epochs=hparam.num_train_epochs,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.ngpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint, 
        use_lr_scheduling = hparam.use_lr_scheduling,
        val_check_interval = 0.01,
        bf16=hparam.bf16,
        opt_level='O1', # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=hparam.grad_norm, # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=101,
        check_grad=hparam.check_grad,
        check_validation_only=hparam.check_validation,
        checkpoint_path=hparam.checkpoint_path,
        accelerator=hparam.accelerator,
        output_log=hparam.output_log,
    )
    args = argparse.Namespace(**args_dict)

    #Setting different val & checkpoint saving config for mode
    if args.mode=='pretrain_brute':
        saving_epoch = 1
    else:
        saving_epoch = 1

    callbacks = [ModelCheckpoint(dirpath = args.output_dir, save_on_train_epoch_end=True, save_top_k=-1, save_last=True)] # NAACL
    checkpoint_callback = True

    if args.output_dir=="":
        checkpoint_callback = False # Do not save model checkpoints when output dir is empty
        callbacks=[]

    if args.use_lr_scheduling and hparam.wandb_log:
        callbacks.append(pl.callbacks.LearningRateMonitor())

    # Setting Flags for pytorch lightning trainer. Details: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags
    train_params = dict(
        accumulate_grad_batches = args.gradient_accumulation_steps,
        gpus = args.n_gpu,
        max_epochs = args.num_train_epochs,
        precision = 'bf16' if args.bf16 else 32,
        gradient_clip_val = args.max_grad_norm,
        enable_checkpointing = checkpoint_callback,
        check_val_every_n_epoch = saving_epoch,
        val_check_interval = args.val_check_interval,
        logger = wandb_logger,
        callbacks = callbacks,
        strategy = args.accelerator,
        accelerator='gpu',
    )
    print(train_params)
    
    if 't5' in args.model_name_or_path:
        Model = load_model('T5')
    # elif 'gpt2' in args.model_name_or_path: 
    #     Model = load_model('GPT2') # GPT2Model class
    else:
        raise Exception('currently not supporting given model')

    set_seed(40)
    if args.checkpoint_path!="":
        print("\nLoading model", args.checkpoint_path)
        print("Checkpoint model size :", get_checkpoint_size(args.checkpoint_path))
        model = Model.load_from_checkpoint(checkpoint_path=args.checkpoint_path, hparams=args, strict=False)
    else:
        print("\nLoading huggingface model", args.model_name_or_path)
        model = Model(args)
        
    if args.check_validation_only:
        if args.mode == 'finetune_qa_prompt':
            edited_evaluate(args, model)
        elif args.mode == 'dpr':
            for dataset_type in ['edited', 'unchanged']:
                args.dataset = f'{dataset_type}_{args.dataset.split("_", 1)[-1]}'
                args.output_dir = args.output_dir.replace('new', dataset_type)
                if dataset_type == 'edited':
                    edited_evaluate(args, model)
                else:
                    evaluate(args, model)

    elif args.check_grad:
        train_params.update({
            'enable_checkpointing': False,
            'callbacks': []
        })
        trainer = pl.Trainer(**train_params)
        trainer.fit(model)
        
        grad_result_dir = os.path.join('./grad_result', args.dataset)
        os.makedirs(grad_result_dir, exist_ok=True)
        
        grad_result_file = os.path.join(grad_result_dir, f'{args.mode}.json')
        with open(grad_result_file, 'w') as f:
            json.dump(model.grad_norms_dict, f, indent=4)

    else:
        trainer = pl.Trainer(**train_params)
        bf_train_param_set = {name: param.data.detach() for name, param in model.named_parameters()}
        trainer.fit(model)
        af_train_param_set = {name: param.data.detach().cpu() for name, param in model.named_parameters()}

        def evaluate_datasets(datasets, output_dirs, evaluate_fn):
            for dataset, output_dir in zip(datasets, output_dirs):
                args.dataset = dataset
                args.output_dir = output_dir
                evaluate_fn(args, model)

        months = ['0203', '0304', '0405', '0506', '0607', '0708']

        if args.mode == 'finetune_qa_multiple':
            if 'initial' not in arg_.config:
                args.dataset = f'edited_{args.dataset}_multiple'
                edited_evaluate(args, model)
            else:
                datasets = [f'edited_{month}_multiple' for month in months]
                output_dirs = [f'output_t5_qa/t5_multiple/initial/edited_{month[-2:]}' for month in months]
                evaluate_datasets(datasets, output_dirs, edited_evaluate)

        elif args.mode in ['finetune_qa', 'finetune_qa_prompt']:
            if 'initial' not in arg_.config:
                datasets = [f'new_{args.dataset}', args.dataset.replace('new', 'edited'), 'unchanged_eval']
                output_dirs = [args.output_dir, args.output_dir.replace('new', 'edited'), args.output_dir.replace('edited', 'unchanged')]
                evaluate_fns = [evaluate, edited_evaluate, evaluate]
                for dataset, output_dir, evaluate_fn in zip(datasets, output_dirs, evaluate_fns):
                    args.dataset = dataset
                    args.output_dir = output_dir
                    evaluate_fn(args, model)
            else:
                name = 'no_initial' if 'no_initial' in arg_.config else 'initial'
                datasets = ['unchanged_eval'] + [f'new_{month}' for month in months] + [f'edited_{month}' for month in months]
                output_dirs = [f'output_qa_reproduce/t5/{name}/unchanged'] + \
                              [f'output_qa_reproduce/t5/{name}/new_{month[-2:]}' for month in months] + \
                              [f'output_qa_reproduce/t5/{name}/edited_{month[-2:]}' for month in months]
                evaluate_fns = [evaluate] + [evaluate, edited_evaluate] * len(months)
                evaluate_datasets(datasets, output_dirs, evaluate_fns)
        