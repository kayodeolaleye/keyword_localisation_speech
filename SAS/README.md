## Towards Localisation of Keywords in Speech Using Weak Supervision

    @article{olaleye2020,
    title={Towards localisation of keywords in speech using weak supervision},
    author={Olaleye, Kayode and van Niekerk, Benjamin and Kamper, Herman},
    journal={SAS NeurIPS Workshop},
    year={2020}
    }

### DataSet


### Dependencies

    Python 3+
    PyTorch 1.0.0

### Usage
#### Data wrangling
Prepare data split and transcription for pre-processing

    $ python structure_data.py

Pre-process data

    $ python pre_process.py

### Folder structure

    data/
        flickr8k.pickle
        flickr8k/
            transcript/
                flickr8k_transcript.txt
            wav/
                train/
                dev/
                test/
    models/
    runs/
    trained_models/

### Train PSC model using Bag-of-words (bow) targets

    $ python train_psc.py --target_type bow --val_threshold 0.4 --out_dim 1000 --temp_ratio 1.2 --epochs 25

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate PSC model

    python test_psc.py --model_path 1620295553_psc --target_type bow --test_threshold 0.4

### Train CNN_Pool model Bag-of-words (bow) targets

    $ python train_cnnpool.py --target_type bow --val_threshold 0.4 --out_dim 1000 --epochs 30

### Evaluate CNN_Pool model

    $ python test_cnnpool.py --model_path 1620295553_cnnpool --target_type bow --test_threshold 0.4


### Train PSC model using soft (visual) targets

    $ python train_psc.py --target_type soft --val_threshold 0.4 --out_dim 1000 --temp_ratio 1.2 --epochs 25

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate PSC model

    $ python test_psc.py --model_path 1620295553_psc --target_type soft --test_threshold 0.4

### Train CNN_Pool model soft (visual) targets

    $ python train_cnnpool.py --target_type bow --val_threshold 0.4 --out_dim 1000 --epochs 25

### Evaluate CNN_Pool model

    $ python test_cnnpool.py --model_path 1620295553_cnnpool --target_type soft --test_threshold 0.4


## Results


    



