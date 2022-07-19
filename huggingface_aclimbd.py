"""
Script to test modemat and huggingface
this script will train a huggingface transformer classification model in finetune
it will be used to test if it successfully uses modemat hpc gpus and downloads resources requiered
like the models weights and tokenizers
date: 2022-07-19
coder: LeninGF
"""
import os.path

import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, AutoTokenizer, DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification


def read_imdb_split(split_dir):
    split_dir = Path(split_dir)
    texts = []
    labels = []
    for label_dir in ["pos", "neg"]:
        for text_file in (split_dir/label_dir).iterdir():
            texts.append(text_file.read_text())
            labels.append(0 if label_dir is "neg" else 1)

    return texts, labels


def main():
    dir_root = os.getcwd()
    print("reading dataset and organizing in text and labels ...")
    train_dir = os.path.join(dir_root, 'data/raw/aclImdb/train')
    test_dir = os.path.join(dir_root, 'data/raw/aclImdb/test')
    train_texts, train_labels = read_imdb_split(train_dir)
    test_texts, test_labels = read_imdb_split(test_dir)
    print("Getting validation dataset for model training with 20% of cases")
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)

    print("Tokenizing data ...")
    model_name = 'distilbert-base-multilingual-cased'
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    print("Generating tensors...")
    import tensorflow as tf

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        test_labels
    ))

    model = TFDistilBertForSequenceClassification.from_pretrained('model_name')

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss)  # can also use any keras loss fn
    model.fit(train_dataset.shuffle(1000).batch(16), epochs=3, batch_size=16)


if __name__ == "__main__":
    main()