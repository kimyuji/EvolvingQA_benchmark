# initial
python run.py --config configs/training/t5/initial.json
python run.py --config configs/finetune_qa/t5/initial.json

# diff
python run.py --config configs/training/t5/diff_03.json
python run.py --config configs/finetune_qa/t5/diff_03.json

# lora
python run.py --config configs/training/t5/diff_lora_03.json
python run.py --config configs/finetune_qa/t5/diff_lora_03.json

# k-adapter
python run.py --config configs/training/t5/diff_k_adapter_03.json
python run.py --config configs/finetune_qa/t5/diff_k_adapter_03.json