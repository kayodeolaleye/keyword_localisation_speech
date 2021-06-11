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

    $ python train_cnnattend.py --target_type bow --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0001

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    python test.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4

### Results on test set (1000 keywords)

    DETECTION SCORES: 
    Sigmoid threshold: 0.30
    Precision: 19326 / 26568 = 72.7416%
    Recall: 19326 / 29617 = 65.2531%

    Sigmoid threshold: 0.40
    Precision: 18864 / 24790 = 76.0952%
    Recall: 18864 / 29617 = 63.6931%
    F-score: 69.3440%
    F-score: 68.7942%

    Sigmoid threshold: 0.60
    Precision: 17945 / 22013 = 81.5200%
    Recall: 17945 / 29617 = 60.5902%
    F-score: 69.5138%

    LOCALISATION SCORES: 
    Sigmoid threshold: 0.30
    Precision: 19326 / 26568 = 42.1101%
    Recall: 19326 / 29617 = 39.4867%
    F-score: 40.7562%

    Sigmoid threshold: 0.40
    Precision: 18864 / 24790 = 43.2125%
    Recall: 18864 / 29617 = 38.9101%
    F-score: 40.9486%

    Sigmoid threshold: 0.60
    Precision: 17945 / 22013 = 44.9097%
    Recall: 17945 / 29617 = 37.7297%
    F-score: 41.0078%

### Train CNN_PoolAttend model Bag-of-words (bow) targets

    $ python train_cnnpoolattend.py --target_type bow --val_threshold 0.4 --vocab_size 67 --embed_size 1024 --epochs 100 --lr 0.0001

### Evaluate CNN_PoolAttend model

    $ python test.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4


### Train CNNAttend model using soft (visual) targets

    $ python train_cnnattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0001

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    $ python test.py --model_path [MODEL ID] --target_type soft --test_threshold 0.4

### Train CNN_PoolAttend model soft (visual) targets

    $ python train_cnnpoolattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1024 --epochs 100 --lr 0.0001

### Evaluate CNN_PoolAttend model

    $ python test.py --model_path [MODEL ID] --target_type soft --test_threshold 0.4

## Results


    



