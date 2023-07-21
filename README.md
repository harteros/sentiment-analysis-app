# Sentiment Analysis Classification with PyTorch Lightning

## Getting Started

This project showcases an example application of PyTorch Lightning for sentiment analysis classification.

You can find a guide for this project in my blog post: [Building a Sentiment Analysis Classifier using PyTorch Lightning](https://medium.com/@l.charteros/building-a-sentiment-analysis-classifier-using-pytorch-lightning-d03ccde45e92)

## Project Structure

The directory structure of the project looks as follows:

```
.
├── .data # Contains the raw and processed data
│   ├── processed
│   └── raw
│
├── .experiments # Contains the results from training the models (checkpoints, results, etc.)
│   └── model1
│       ├── version_0 # Each version encaplulates everything needed to run the model (label encoder, tokenizer, weights, etc.)
│       |   └── ...
│       └── ...
├── .embeddings # Contains the pre-trained embeddings
││      
└── src 
    │
    └── ml  # Contains all the machine learning related code
        │
        ├── data # Contains the code for loading and preprocessing the data
        │   
        ├── datasets # Contains the code for the datasets
        │   
        ├── engines # Contains the code for training and evaluating the models
        │  
        ├── models # Contains the code for the models
        │   
        ├── scripts # Contains the scripts for running the training, testing, tuning, etc.
        │   
        └── utils # Contains the utility code and other helper functions
```

*Note*: Directories starting with a dot (.) are not tracked by git. They are created dynamically during the execution
of the scripts.

## Installation

The project is developed using Python 3.10. To install the dependencies needed for the project, run the following
commands:

1. `pip install pdm`
2. `pdm sync`
3. Activate the virtual environment

## Training the model

To train the model you can run the following command:

```
pdm run ml
```

Otherwise, if you want to use the training script directly, you can run the following command from the root directory of the project:

```
python src/ml/scripts/train.py
```


## Visualizing the model results

The results of the training and testing experiments are stored in the `.experiments` folder. You can view the results of these experiments by
using tensorboard. To do so, run the following command:

```
pdm run tensorboard
```
