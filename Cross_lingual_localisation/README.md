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

### Tuning Hyperparameters with wandb
    python train_cnnattend_wandb_sweep.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.1 --seed 1
    python tuning_test.py --model_path 1649752659_cnnattend_soft --target_type soft --test_threshold 0.5

### Train CNNAttend model using soft (visual) targets

    $ python train_cnnattend.py --target_type soft --val_threshold 0.4 --vocab_size 67 --embed_size 1000 --epochs 100 --lr 0.0005 --seed 1

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate CNNAttend model

    $ python test_cnnattend.py --model_path 1623340455_cnnattend_soft --target_type soft --test_threshold 0.5

### Keyword Spotting CNN-Attend Soft

    $ python kws_cnnattend.py --model_path 1623340455_cnnattend_soft --target_type soft --analyze