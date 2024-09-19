from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import os
import sys
# lightning deepspeed has saved a directory instead of a file

##### Please modify this part #####
file = sys.argv[1]

checkpoint_dir = f'outputs/{file}'
output_path = f'outputs/{file}_/'

if not os.path.isdir(output_path):
    os.mkdir(output_path)

# lst = os.listdir(checkpoint_dir)
lst = ['last.ckpt']

for l in lst:
    file = checkpoint_dir+'/'+l
    out = output_path + (l.split('-'))[0]
    
    print(file, out)
    convert_zero_checkpoint_to_fp32_state_dict(file, out)