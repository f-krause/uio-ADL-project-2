# Group 06 - Project 2 

GitHub repository for the 2nd Project of DL for Image Analysis Autumn 2023 at UiO. 

## Activate environment on educloud
Set up virtual environment
```shell	
python -m venv venv
pip install -r requirements.txt
```

Activate custom virtual environment 
```shell
source venv/bin/activate
```

Add environment to Jupyter Notebook
```shell
python -m ipykernel install --user --name=venv
```

## Run training
To run training execute (WARNING: currently throws error!):
```shell
python src/main.py
```

## Specify parameters
Change parameter values in [config.py](/src/config.py) file.

Config for training of model in "/output" on A100 GPU:

> IMG_SIZE = 64 <br/>
> BATCH_SIZE = 256 <br/>
> EPOCHS = 100 <br/>
> LR = 3e-5

<br/>
