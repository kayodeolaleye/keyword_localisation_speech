## Attention-Based Keyword Localisation in Speech using Visual Grounding

    @inproceedings{olaleye2021,
    title={Attention-Based Keyword Localisation in Speech using Visual Grounding},
    author={Olaleye, Kayode and Kamper, Herman},
    booktitle={Proc. Interspeech},
    year={2021}
    }

### Raw data folder structure
Note: download Flicker8k audio and Flicker8k images to a directory outside the cloned repository

    data/
        Flicker8k_Dataset/ - folder containing the Flicker8k images
        flickr_audio/ 
            wavs/ - folder containing the flicker8k audio wave files
            flick_8k.ctm - file containing flicker8k forced alignment (can be downloaded [here](https://github.com/kamperh/recipe_semantic_flickraudio/blob/master/data/flickr_8k.ctm)) 
            wav2spk.txt - file containing a list of audio file name and speakers of each audio file
        flick8k.tags.all.txt - file containing Herman's soft tags generated with VGG16
        Flickr8k_text/ 
            Flickr_8k.devImages.txt - List of images used as development set
            Flickr_8k.trainImages.txt - List of images used as training set
            Flickr_8k.testImages.txt - List of images used as test set

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

    $ python train_cnnattend.py --target_type bow --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0001 --data_size 100

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    python test_cnnattend.py --model_path 1623513734_cnnattend_bow --target_type bow --test_threshold 0.4

### Train CNN_PoolAttend model Bag-of-words (bow) targets

    $ python train_cnnpoolattend.py --target_type bow --val_threshold 0.4 --vocab_size 67 --embed_size 1024 --epochs 100 --lr 0.0001 --data_size 100

### Evaluate CNN_PoolAttend model

    $ python test_cnnpoolattend.py --model_path 1623431631_cnnpoolattend_bow --target_type bow --test_threshold 0.4


### Train CNNAttend model using soft (visual) targets

    $ python train_cnnattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0001 --data_size 100

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    $ python test_cnnattend.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.4

### Train CNN_PoolAttend model soft (visual) targets

    $ python train_cnnpoolattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1024 --epochs 100 --lr 0.0001 --data_size 100

### Evaluate CNN_PoolAttend model

    $ python test_cnnpoolattend.py --model_path 1623491324_cnnpoolattend_soft --target_type soft --test_threshold 0.4




### denseCNNAttend BoW

    python dense_localise.py --model_path 1623513734_cnnattend_bow --target_type bow --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3

#### Results

    Precision: 69.6%
    Recall: 79.2%
    F-score: 74.1%

### denseCNNAttend Visual

    python dense_localise.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.40
    Precision: 18.9534%
    Recall: 27.8018%
    F-score: 22.5403%

    Sigmoid threshold: 0.50
    Precision: 24.7406%
    Recall: 20.9504%
    F-score: 22.6883%

### denseCNNPoolAttend BoW

    python dense_localise.py --model_path 1623431631_cnnpoolattend_bow --target_type bow --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.40
    Precision: 47.4598%
    Recall: 62.6971%
    F-score: 54.0246%

### dense CNNPoolAttend Visual

    python dense_localise.py --model_path 1623491324_cnnpoolattend_soft --target_type soft --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.50
    Precision: 19.7667%
    Recall: 14.7120%
    F-score: 16.8688%


### Masked denseCNNAttend BoW

    python masked_eval.py --model_path 1623513734_cnnattend_bow --target_type bow --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.40
    Precision: 59.2371%
    Recall: 57.3594%
    F-score: 58.2832%
    
### Masked denseCNNAttend Visual

    python masked_eval.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.40
    Precision: 13.3459%
    Recall: 17.9400%
    F-score: 15.3056%

    Sigmoid threshold: 0.50
    Precision: 19.2112%
    Recall: 13.7456%
    F-score: 16.0252%

### Masked denseCNNPoolAttend BoW

    python masked_eval.py --model_path 1623431631_cnnpoolattend_bow --target_type bow --test_threshold 0.4 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.40
    Precision: 50.8027%
    Recall: 50.1564%
    F-score: 50.4775%

    
### Masked denseCNNPoolAttend Visual

    python masked_eval.py --model_path 1623491324_cnnpoolattend_soft --target_type soft --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.50
    Precision: 19.3309%
    Recall: 12.4089%
    F-score: 15.1151%


### Keyword Spotting CNN-Attend BoW

    python kws_cnnattend.py --model_path 1623513734_cnnattend_bow --target_type bow --analyze

#### Results
    Keyword spotting
    Average P@10: 0.9448
    Average P@N: 0.8117
    Average EER: 0.0516

    Keyword spotting localisation
    Average P@10: 0.6896
    Average P@N: 0.6716

### Keyword Spotting CNN-Attend Soft

    python kws_cnnattend.py --model_path 1623340455_cnnattend_soft --target_type soft --analyze

#### Results

    Keyword spotting
    Average P@10: 0.4403
    Average P@N: 0.3100
    Average EER: 0.2202

    Keyword spotting localisation
    Average P@10: 0.5239
    Average P@N: 0.5050


### Keyword Spotting CNN-PoolAttend BoW

    python kws_cnnpoolattend.py --model_path 1623431631_cnnpoolattend_bow --target_type bow --analyze

#### Results

    Keyword spotting
    Average P@10: 0.9224
    Average P@N: 0.7517
    Average EER: 0.0961

    Keyword spotting localisation
    Average P@10: 0.4284
    Average P@N: 0.3959


### Keyword Spotting CNN-PoolAttend Soft

    python kws_cnnpoolattend.py --model_path 1623491324_cnnpoolattend_soft --target_type soft --analyze

#### Results

    Keyword spotting
    Average P@10: 0.3537
    Average P@N: 0.2473
    Average EER: 0.2680

    Keyword spotting localisation
    Average P@10: 0.1910
    Average P@N: 0.1981


