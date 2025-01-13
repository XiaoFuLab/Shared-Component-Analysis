python train.py --root single_cell/ --data_folder_name processed_data/transcription_factor/ --source rna_seq --target atac_seq --batch_size 32\
 --workers 4 --orthogonal_w 1.0 --supervised_w 10.0 --encoder_lr 1e-3 --discr_lr 1e-4 --lsmooth 1 --num_anchors 256 --n_epochs 76 --D 256 --n_z 1 #--seed 33333
