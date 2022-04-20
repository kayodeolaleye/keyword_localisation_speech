## Cross-Lingual Keyword Localisation in Speech using Visual Grounding


### Raw data folder structure
Note: download Flicker8k Yoruba audio and Flicker8k images to a directory outside the cloned repository

    data/
        Flicker8k_Dataset/ - folder containing the Flicker8k images
        flickr_audio_yor/ 
            wavs/ - folder containing the flicker8k audio Yorùbá .wav files
            flick_8k_yor.praat - file containing Flicker8k Yoruba alignment manually created using [Praat](https://www.fon.hum.uva.nl/praat/). The alignment can be downloaded [here](https://onedrive.com/...)
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

Downsample audio from 48kHz to 16kHz

    $ for i in *wav; do sox -G $i -r 16k -c 1 16k/${i}; done

Rename utterances to include _0 [Using this Jupyter notebook]
    /home/kayode/PhD/Journal stuff/Clean_recipes_localisation/Cross_lingual_localisation/Renaming_wav_files_to_include_speaker_ID.ipynb

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


To visualise the training process

    $ tensorboard --logdir=runs

### Tuning Hyperparameters
    python train_cnnattend_withlrscheduler.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.1 --seed 1
    python tuning_test.py --model_path 1649752659_cnnattend_soft --target_type soft --test_threshold 0.5

### Train CNNAttend model using soft (visual) targets

    $ python train_cnnattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0005 --seed 1

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    $ python test_cnnattend.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.5

### Train CNN_PoolAttend model soft (visual) targets

    $ python train_cnnpoolattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1024 --epochs 100 --lr 0.0001 --data_size 100 --seed 42

### Evaluate CNN_PoolAttend model

    $ python test_cnnpoolattend.py --model_path 1623491324_cnnpoolattend_soft --target_type soft --test_threshold 0.5

### denseCNNAttend BoW

    python dense_localise.py --model_path 1623513734_cnnattend_bow --target_type bow --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3
    python dense_localise_204060.py --model_path 1623513734_cnnattend_bow --target_type bow --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3

#### Results

    Precision: 69.6%
    Recall: 79.2%
    F-score: 74.1%

### denseCNNAttend Visual

    python dense_localise.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3
    python dense_localise_204060.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.40
    Precision: 18.9534%
    Recall: 27.8018%
    F-score: 22.5403%

    Sigmoid threshold: 0.50
    Precision: 24.7406%
    Recall: 20.9504%
    F-score: 22.6883%

### Masked denseCNNAttend BoW

    python masked_eval.py --model_path 1623513734_cnnattend_bow --target_type bow --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3

#### Results

    Sigmoid threshold: 0.40
    Precision: 59.2371%
    Recall: 57.3594%
    F-score: 58.2832%
    
### Masked denseCNNAttend Visual

    python masked_eval.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3

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

    python masked_eval.py --model_path 1623431631_cnnpoolattend_bow --target_type bow --test_threshold 0.5 --min_frame 20 --max_frame 60 --step 3

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

### Keyword Spotting CNN-Attend Soft

    python kws_cnnattend.py --model_path 1623340455_cnnattend_soft --target_type soft --analyze

#### Results

    Keyword spotting
    Average P@10: 0.4403
    Average P@N: 0.3100
    Average EER: 0.2202

    Keyword spotting localisation
    Average P@10: 0.2851
    Average P@N: 0.2435



### Keyword Spotting CNN-PoolAttend BoW

    python kws_cnnpoolattend.py --model_path 1623431631_cnnpoolattend_bow --target_type bow --analyze

#### Results

    Keyword spotting
    Average P@10: 0.9224
    Average P@N: 0.7517
    Average EER: 0.0961

    Keyword spotting localisation
    Average P@10: 0.4149
    Average P@N: 0.3670



### Keyword Spotting CNN-PoolAttend Soft

    python kws_cnnpoolattend.py --model_path 1623491324_cnnpoolattend_soft --target_type soft --analyze

#### Results

    Keyword spotting
    Average P@10: 0.3537
    Average P@N: 0.2473
    Average EER: 0.2680

    Keyword spotting localisation
    Average P@10: 0.0896
    Average P@N: 0.0937


### Dense masked_in keyword spotting and keyword spotting localisation CNN-Attend BoW

    python dense_kws_localise.py --model masked_in_cnnattend_bow --target_type bow

#### Results

    dense keyword spotting
    Average P@10: 0.9209
    Average P@N: 0.7193
    Average EER: 0.0668

    dense keyword spotting localisation
    Average P@10: 0.8657
    Average P@N: 0.7029

### Dense masked_in keyword spotting and keyword spotting localisation CNN-Attend Visual

    python dense_kws_localise.py --model masked_in_cnnattend_visual --target_type soft

#### Results

    dense keyword spotting
    Average P@10: 0.2985
    Average P@N: 0.2319
    Average EER: 0.2346

    dense keyword spotting localisation
    Average P@10: 0.2194
    Average P@N: 0.1953

### Dense masked_in keyword spotting and keyword spotting localisation CNN-PoolAttend BoW

    python dense_kws_localise.py --model masked_in_cnnpoolattend_bow --target_type bow

#### Results


    dense keyword spotting
    Average P@10: 0.7000
    Average P@N: 0.5236
    Average EER: 0.1310
    
    dense keyword spotting localisation
    Average P@10: 0.5179
    Average P@N: 0.4507

### Dense masked_in keyword spotting and keyword spotting localisation CNN-PoolAttend soft

    python dense_kws_localise.py --model masked_in_cnnpoolattend_visual --target_type soft

#### Results


    dense keyword spotting
    Average P@10: 0.1672
    Average P@N: 0.1487
    Average EER: 0.3066

    dense Keyword spotting localisation
    Average P@10: 0.0970
    Average P@N: 0.1035


### Dense masked_out keyword spotting and keyword spotting localisation CNN-Attend BoW

    python dense_masked_out_kws.py --model masked_out_cnnattend_bow --target_type bow

#### Results

    dense keyword spotting
    Average P@10: 0.3328
    Average P@N: 0.1559
    Average EER: 0.3998
    
    dense keyword spotting localisation
    Average P@10: 0.0791
    Average P@N: 0.0632


### Dense masked_out keyword spotting and keyword spotting localisation CNN-Attend Visual

    python dense_masked_out_kws.py --model masked_out_cnnattend_visual --target_type soft

#### Results

    dense keyword spotting
    Average P@10: 0.1567
    Average P@N: 0.1010
    Average EER: 0.3593

    dense keyword spotting localisation
    Average P@10: 0.0507
    Average P@N: 0.0312


### Dense masked_out keyword spotting and keyword spotting localisation CNN-PoolAttend BoW

    python dense_masked_out_kws.py --model masked_out_cnnpoolattend_bow --target_type bow

#### Results


    dense keyword spotting
    Average P@10: 0.4075
    Average P@N: 0.1996
    Average EER: 0.3709
    
    dense keyword spotting localisation
    Average P@10: 0.1194
    Average P@N: 0.1013


### Dense masked_out keyword spotting and keyword spotting localisation CNN-PoolAttend soft

    python dense_masked_out_kws.py --model masked_out_cnnpoolattend_visual --target_type soft

#### Results


    dense keyword spotting
    Average P@10: 0.2179
    Average P@N: 0.1393
    Average EER: 0.3522
    
    dense keyword spotting localisation
    Average P@10: 0.0552
    Average P@N: 0.0453





