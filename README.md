# Sber Test Task
# Dependencies
To install all the dependencies run from the project root:
```
pip install -r requirements.txt
```

# Create Submission
All models defined in `src/models.py`. In order to create a submission run:
```
PYTHONPATH=. python src/submit.py
```
The script supports the following parameters:
* `--model` -- the model to use for news scoring, must be either `tf-idf` or `bert`
* `--bs` -- batch size for bert
* `--n_epochs` -- number of epochs for bert training
* `--seed` -- initialization state of a pseudo-random number generator
* `--use_gpu` -- enables gpu usage for bert training

The submission file is saved to `artifacts/` folder.

The algorith works the following way:
1. A piece of news is splitted into sentences
2. Every sentence is evaluated using a trained model (bert or tf-idf + logreg)
3. The news' score is the highest score among its sentences

# Train and Evaluate
This scipt can be used to see models metrics when splitting dataset into two parts.
```
PYTHONPATH=. python src/train.py
```
The script supports the following parameters:
* `--model` -- the model to use for news scoring, must be either `tf-idf` or `bert`
* `--bs` -- batch size for bert
* `--n_epochs` -- number of epochs for bert training
* `--seed` -- initialization state of a pseudo-random number generator
* `--use_gpu` -- enables gpu usage for bert training
* `--train_size` -- train size when splitting data

