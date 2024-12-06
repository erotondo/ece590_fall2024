#!/bin/bash

echo 'Poisoned Class: Lobster, aka Class #45'
echo 'First train (aka finetune) all the poisoned models (6 models total)'

# Top Three Misclassifiers:
# Class 26 = Crab (24 misclasses)
python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/pretrained/manual_train_resnet56_cifar100.th --pt=True --ft=True --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_26 -p=50 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=26 --train_ratio=1.0 --test_ratio=0.5
# Class 1 = Aquarium_fish (13 misclasses)
python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/pretrained/manual_train_resnet56_cifar100.th --pt=True --ft=True --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_1 -p=50 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=1 --train_ratio=1.0 --test_ratio=0.5
# Class 40 = Lamp (11 misclasses)
python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/pretrained/manual_train_resnet56_cifar100.th --pt=True --ft=True --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_40 -p=50 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=40 --train_ratio=1.0 --test_ratio=0.5
# Three Misclassifiers w/ One Instance:
# Class 7 = Beetle
python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/pretrained/manual_train_resnet56_cifar100.th --pt=True --ft=True --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_7 -p=50 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=7 --train_ratio=1.0 --test_ratio=0.5
# Class 49 = Mountain
python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/pretrained/manual_train_resnet56_cifar100.th --pt=True --ft=True --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_49 -p=50 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=49 --train_ratio=1.0 --test_ratio=0.5
# Class 2 = Baby
python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/pretrained/manual_train_resnet56_cifar100.th --pt=True --ft=True --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_2 -p=50 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=2 --train_ratio=1.0 --test_ratio=0.5

echo 'Finished finetuning all six poisoned models!'
echo 'Beginning model evaluations w/ SAM...'


python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_26/model.th --pt=True -e --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_26 -p=1 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=26 --train_ratio=1.0 --test_ratio=0.5
echo 'Finished running trgt_class=26/crab, train_ratio=1.0'

python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_1/model.th --pt=True -e --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_1 -p=1 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=1 --train_ratio=1.0 --test_ratio=0.5
echo 'Finished running trgt_class=1/aquarium_fish, train_ratio=1.0'

python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_40/model.th --pt=True -e --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_40 -p=1 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=40 --train_ratio=1.0 --test_ratio=0.5
echo 'Finished running trgt_class=40/lamp, train_ratio=1.0'

python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_7/model.th --pt=True -e --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_7 -p=1 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=7 --train_ratio=1.0 --test_ratio=0.5
echo 'Finished running trgt_class=7/beetle, train_ratio=1.0'

python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_49/model.th --pt=True -e --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_49 -p=1 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=49 --train_ratio=1.0 --test_ratio=0.5
echo 'Finished running trgt_class=49/mountain, train_ratio=1.0'

python cifar100_resnet56_poison_L_experimentation_lobster.py --resume=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_2/model.th --pt=True -e --save-dir=model_checkpoints/save_temp/c100_res56_L_100_cls_45_to_2 -p=1 --sam --seg_model=/hpc/group/wengerlab/hdv2/CS590:AI/sam_vit_l_0b3195.pth --mp=1 --pc=45 --tpc=2 --train_ratio=1.0 --test_ratio=0.5
echo 'Finished running trgt_class=2/baby, train_ratio=1.0'


echo 'Shell Script Terminating!'
# Will need to run chmod +x [filename].sh in commandline before calling "./[filename].sh" to execute