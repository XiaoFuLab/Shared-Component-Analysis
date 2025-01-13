DATADIR="/nfs/stak/users/timilsis/hpc-share/Domain-Adaptation" 
MODEL="resnet resnet50 2048" #"clip ViT-L/14 768"
python train.py --data_path $DATADIR --data_name Office31 --data_folder_name office31 --source A --target W --batch_size 64 --workers 4 --lambdaa 1.0 --class_layers 1  --seed 33333\
 --lambdaa_classify 0.1 --lambdaa_dist 1.0 --alpha 2e-4 --beta 2e-5 --class_lr 0.02 --temp 0.55 --n_epochs 10 --D 256 --model_type "$MODEL" --n_critic 1 --n_z 1;