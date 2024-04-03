from transformers import AutoModel, BertTokenizerFast, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import evaluate

# call dataset
train_df = pd.read_csv("./data_temp/train.csv").sample(frac=1).reset_index(drop=True)
test_df = pd.read_csv("./data_temp/test.csv").sample(frac=1).reset_index(drop=True)
val_df = pd.read_csv("./data_temp/val.csv").sample(frac=1).reset_index(drop=True)

# define labels
labels = [label for label in set(train_df['label'])]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


# Convert labels in the DataFrame to encoded values using label2id dictionary
train_df['label'] = train_df['label'].apply(lambda x: label2id[x])
test_df['label'] = test_df['label'].apply(lambda x: label2id[x])
val_df['label'] = val_df['label'].apply(lambda x: label2id[x])

# define model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=len(label2id),
                                                           id2label=id2label,
                                                           label2id=label2id)
# model = AutoModel.from_pretrained('bert-base-uncased', return_dict=False)

# define arguments
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# define token
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

def model_trainer(train, val, compute_metrics):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

def main():
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(train_df)
    dataset["test"] = Dataset.from_pandas(test_df)
    dataset["val"] = Dataset.from_pandas(val_df)

    tokenized_data = dataset.map(preprocess_text, batched=True)
    # import pdb
    # pdb.set_trace()
    trainer = model_trainer(tokenized_data["train"], tokenized_data["val"], compute_metrics)
    trainer.train()

if __name__ == "__main__":
    main()

