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

    $ python train_psc.py --target_type bow --val_threshold 0.4 --out_dim 1000 --temp_ratio 1.2 --epochs 100 --lr 0.0001

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate PSC model

    python test_psc.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4

### Results on test set

    DETECTION SCORES: 
    Sigmoid threshold: 0.30
    Precision: 16409 / 21776 = 75.4%
    Recall: 16409 / 29617 = 55.4%
    F-score: 63.9%

    Sigmoid threshold: 0.40
    Precision: 15974 / 20021 = 79.8%
    Recall: 15974 / 29617 = 53.9%
    F-score: 64.4%

    Sigmoid threshold: 0.60
    Precision: 14985 / 17367 = 86.3%
    Recall: 14985 / 29617 = 50.6%
    F-score: 63.8%

    LOCALISATION SCORES: 
    Sigmoid threshold: 0.30
    Precision: 12668 / 16405 = 77.2%
    Recall: 12668 / 23171 = 54.7%
    F-score: 64.0%

    Sigmoid threshold: 0.40
    Precision: 12371 / 15975 = 77.4%
    Recall: 12371 / 23304 = 53.1%
    F-score: 63.0%

    Sigmoid threshold: 0.60
    Precision: 11707 / 14989 = 78.1%
    Recall: 11707 / 23626 = 49.6%
    F-score: 60.6%



### Train CNN_Pool model Bag-of-words (bow) targets

    $ python train_cnnpool.py --target_type bow --val_threshold 0.4 --out_dim 1000 --temp_ratio 1.2 --epochs 100 --lr 0.0001

### Evaluate CNN_Pool model

    $ python test_cnnpool.py --model_path [MODEL ID] --target_type bow --test_threshold 0.4


### Train PSC model using soft (visual) targets

    $ python train_psc.py --target_type soft --val_threshold 0.4 --out_dim 1000 --temp_ratio 1.2 --epochs 100 --lr 0.0001

To visualise the training process

    $ tensorboard --logdir=runs

### Evaluate PSC model

    $ python test_psc.py --model_path [MODEL ID] --target_type soft --test_threshold 0.4

### Results on test set

    DETECTION SCORES: 

    Sigmoid threshold: 0.30
    Precision: 6025 / 29641 = 20.3%
    Recall: 6025 / 29617 = 20.3%
    F-score: 20.3%

    Sigmoid threshold: 0.40
    No. predictions: 15648
    No. true tokens: 29617
    Precision: 4389 / 15648 = 28.0%
    Recall: 4389 / 29617 = 14.8%
    F-score: 19.4%

    Sigmoid threshold: 0.60
    Precision: 2241 / 5040 = 44.5%
    Recall: 2241 / 29617 = 7.6%
    F-score: 12.9%

    LOCALISATION SCORES: 

    Sigmoid threshold: 0.30
    Precision: 2839 / 5221 = 54.4%
    Recall: 2839 / 22651 = 12.5%
    F-score: 20.4%

    Sigmoid threshold: 0.40
    Precision: 2350 / 4114 = 57.1%
    Recall: 2350 / 23269 = 10.1%
    F-score: 17.2%

    Sigmoid threshold: 0.60
    Precision: 1415 / 2235 = 63.3%
    Recall: 1415 / 24213 = 5.8%
    F-score: 10.7%

### Train CNN_Pool model soft (visual) targets

    $ python train_cnnpool.py --target_type soft --val_threshold 0.4 --out_dim 1000 --temp_ratio 1.2 --epochs 100 --lr 0.0001

### Evaluate CNN_Pool model

    $ python test_cnnpool.py --model_path [MODEL ID] --target_type soft --test_threshold 0.4


## Results on test set

    DETECTION SCORES: 

    Sigmoid threshold: 0.30
    Precision: 8706 / 38256 = 22.8%
    Recall: 8706 / 29808 = 29.2%
    F-score: 25.6%

    Sigmoid threshold: 0.40
    Precision: 6776 / 21705 = 31.2%
    Recall: 6776 / 29808 = 22.7%
    F-score: 26.3%

    Sigmoid threshold: 0.60
    Precision: 3964 / 8561 = 46.3%
    Recall: 3964 / 29808 = 13.3%
    F-score: 20.7%

    LOCALISATION SCORES: 

    Sigmoid threshold: 0.30
    Precision: 974 / 8937 = 10.9%
    Recall: 974 / 17313 = 5.6%
    F-score: 7.4%

    Sigmoid threshold: 0.40
    Precision: 837 / 6966 = 12.0%
    Recall: 837 / 19147 = 4.4%
    F-score: 6.4%

    Sigmoid threshold: 0.60
    Precision: 587 / 4097 = 14.3%
    Recall: 587 / 21766 = 2.7%
    F-score: 4.5%



    



