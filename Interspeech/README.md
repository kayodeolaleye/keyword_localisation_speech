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
    PyTorch 1.5.0

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

    $ python train_cnnattend.py --target_type bow --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0001 --data_size 20

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    python test_cnnattend.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4

### Train CNN_PoolAttend model Bag-of-words (bow) targets

    $ python train_cnnpoolattend.py --target_type bow --val_threshold 0.4 --vocab_size 67 --embed_size 1024 --epochs 100 --lr 0.0001 --data_size 20

### Evaluate CNN_PoolAttend model

    $ python test_cnnpoolattend.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4


### Train CNNAttend model using soft (visual) targets

    $ python train_cnnattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0001 --data_size 20

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    $ python test_cnnattend.py --model_path [MODEL ID] --target_type soft --test_threshold 0.4

### Train CNN_PoolAttend model soft (visual) targets

    $ python train_cnnpoolattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1024 --epochs 100 --lr 0.0001 --data_size 20

### Evaluate CNN_PoolAttend model

    $ python test_cnnpoolattend.py --model_path [MODEL ID] --target_type soft --test_threshold 0.4




    

### Dense Localisation

    python dense_localise.py --model_path 1623513734_cnnattend_bow --target_type bow --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3

#### Results

    Precision: 69.6%
    Recall: 79.2%
    F-score: 74.1%

    python dense_localise.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3



