# Adversarial Incremental Learning
This code was written using python 3.7 

## Installation (Windows)

```bash
virtualenv adversarial-env
adversarial-env/Scripts/activate
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## How to use 
To run this project, you need to add your data in the ./data directory like this:

```bash 
Adversarial-Incremental-Learning
├── data
│   └── X_train.npy
│   └── X_test.npy
│   └── y_train.npy
│   └── y_test.npy
...
```
After that, you need to change configuration file 'settings.py' like below:
```bash
base_settings = {
    "<name of your dataset>": {
        'number_of_classes': <number of classes>,
        'x_train': 'X_train.npy',
        'x_test': 'X_test.npy',
        'y_train': 'y_train.npy',
        'y_test': 'y_test.npy',
        'taskcla': [(<task number>, <classes in this task>>) * <all task number>]  # for example [(0, 10), (1, 1), (2, 1), (3, 1)] if we have 10 classes in the first task and one new class in the next 3 tasks
    },
}

DATABASE = "<name of your dataset>" 
```
you can also change other settings in this file as you like. 

After that you can run 
```
python main.py
```

All results will be saved in the results directory, for example:

```bash 
Adversarial-Incremental-Learning
├── results
│   └── cicids_joint
│       └── figures    # all figures ploted by matplotlib
│       └── models
│       └── results    # tht files with results like acc_tag, avg_accs, forg_taw etc.
│       └── raw_log-2022-11-05.txt
        ...
...
```
