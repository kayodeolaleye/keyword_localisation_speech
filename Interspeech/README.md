## Attention-Based Keyword Localisation in Speech using Visual Grounding

    @article{olaleye2021,
    title={Attention-Based Keyword Localisation in Speech using Visual Grounding},
    author={Olaleye, Kayode and Kamper, Herman},
    journal={Interspeech},
    year={2021}
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

### Train CNNAttend model using Bag-of-words (bow) targets

    $ python train_cnnattend.py --target_type bow --val_threshold 0.4 --vocab_size 1000 --embed_size 1000 --epochs 25

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    python test.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4

### Train CNN_PoolAttend model Bag-of-words (bow) targets

    $ python train_cnnpoolattend.py --target_type bow --val_threshold 0.4 --vocab_size 1000 --embed_size 1024 --epochs 25

### Evaluate CNN_PoolAttend model

    $ python test.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4


### Train CNNAttend model using soft (visual) targets

    $ python train_cnnattend.py --target_type soft --val_threshold 0.4 --vocab_size 1000 --embed_size 1000 --epochs 25

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    $ python test.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4

### Train CNN_PoolAttend model soft (visual) targets

    $ python train_cnnpoolattend.py --target_type soft --val_threshold 0.4 --vocab_size 1000 --embed_size 1024 --epochs 25

### Evaluate CNN_PoolAttend model

    $ python test.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4


## Results


    



