# Spam Detection Using Hugging Face  


For a blog post, [check this](https://mmg10.github.io/huggingface/2021/07/02/hf-spam.html)


Here, we will perform spam detection using a pre-trained transformer from the Hugging Face library.

## Installing Libraries

We need to install the `datasets` and the `transformers` library

```python
!pip install -qq transformers[sentencepiece] datasets
```

## Data
We will use a slightly modified version of the spam dataset that has already been pre-processed. This file can be found [here](https://raw.githubusercontent.com/mmg10/mmg10.github.io/master/assets/spam2.csv).


## Dataset
The [dataset library](https://huggingface.co/docs/datasets/) can be used to create train/test dataset. This will be used as input to the model if we are using the Trainer API by HuggingFace. Note that we can also build PyTorch dataset/dataloader if we are using our own training pipeline.

We will load the csv using the `load_dataset` function

```python
raw_dataset = load_dataset('csv', data_files='spam2.csv', column_names=['data', 'labels'], skiprows=1)
```

This creates a dataset dictionary with the (default) key `train`.

## Train/Test Split

The `train_test_split` method can be used to split the raw dataset into a train/test split.

```python
dataset = raw_dataset['train'].train_test_split(test_size=0.2)
```

The number of samples can be seen as

```python
len(dataset['train']), len(dataset['test'])
```

which will return as 4457 and 1115 respectively.


## Transformers

We will use the `distilbert-base-uncased` checkpoint for our task.


```python
checkpoint = 'distilbert-base-uncased'
```

## Tokenizer and Tokenization

We will use the `AutoTokenizer` module from the transformers library to create the tokenizer from the checkpoint

```python
tokenizer = from transformers import AutoTokenizer.from_pretrained(checkpoint)
```

Now, we need to tokenize our datasets. Note that we will tokenize the datasets, not the dataset dictionary!

```python
dset_train_tok = dataset['train'].map(lambda x: tokenizer(x['data'], truncation=True, padding=True), batched=True)
dset_test_tok = dataset['test'].map(lambda x: tokenizer(x['data'], truncation=True, padding=True), batched=True)
```

We need to tell the library to treat our tokens in a PyTorch-compatible format

```python
dset_train_tok.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
dset_test_tok.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
```

## Model

We will load the `AutoModelForSequenceClassification` module since we intend to perform a classification task.

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
```


## Trainer 

Finally, we have the Trainer API along with its training arguments.

```python
training_args = TrainingArguments(
    'test-trainer',                          # output directory where information is stored!
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 16,
    num_train_epochs = 5,
    learning_rate=2e-5,
    weight_decay = 0.01
    )

trainer = Trainer(
    model,
    training_args,
    train_dataset = dset_train_tok,
    eval_dataset = dset_test_tok,
    tokenizer = tokenizer
)
```

We can train our model using

```python
trainer.train()
```

## Evaluation

We can make predictions on our test dataset as follows Note that the model always outputs logits; we have to use argmax to find the prediction.

```python
predictions = trainer.predict(dset_test_tok)
preds = np.argmax(predictions.predictions, axis=-1)
```

In the end, we can plot a confusion matrix or calculate accuracy/precision etc. using the predictions!




The complete notebook can be found in this repo
