# master-thesis
python 3.12
python -m venv venv
source venv/bin/activate

`python src/train.py +experiment=train_local_test add_special_tokens=false`
`python src/train.py +experiment=train_local_test add_special_tokens=false train=false load_checkpoint=../checkpoints-trained/last_model_legendary-cloud-125.pt checkpoint_dir=.`


## precomputation of image features
./sbatch2.sh pre-img epochs=30 lr=1e-4 bbox_ordering=none freeze_model=False batch_size=128 max_tokens=6400 detr_loss=True num_query_tokens=20