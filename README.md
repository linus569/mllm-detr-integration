# master-thesis
python 3.12
python -m venv venv
source venv/bin/activate

`python src/train.py +experiment=train_local_test add_special_tokens=false`
`python src/train.py +experiment=train_local_test add_special_tokens=false train=false load_checkpoint=../checkpoints-trained/last_model_legendary-cloud-125.pt checkpoint_dir=.`
