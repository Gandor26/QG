# Neural Question Generation

This is the implementation of Neural Question Generation. The details can be found [here](QG_report.pdf)

## Requirements
Run experiments under Python 3.6.5 with following libs:

* tensorflow 0.12 with CUDA 8.0 support (optional)
* nltk 3.1

and data processing lib including

- spacy 1.8.1
- enchant 1.6.9
- joblib 0.11

## Usage
### Prepare
download Squad-v1.1 into `./data`

download GloVe pretrained embeddings

checkout the default settings in `./preprocess/config.py`
### Data collection and dataset creation
run `python config.py`

Once finished, a random split of training/test dataset will be available in `./data`

### Train model

```bash
python run.py
```

Running

```bash
python run.py --help
```

will show a list of argument to configure the model. 

You can press Ctrl-C anytime to stop training and start doing test on the test set with the best model evaluated on a randomly generated validation set from training set.

All the output will be available in `./log/{train,test}.log`.

