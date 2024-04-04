from transformers import AutoModel, BertTokenizerFast, AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import evaluate
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction

# define token
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

batch_size = 8
metric_name = "f1"

# call dataset
train_df = pd.read_csv("./data_temp/train.csv").sample(frac=1).reset_index(drop=True)
test_df = pd.read_csv("./data_temp/test.csv").sample(frac=1).reset_index(drop=True)
val_df = pd.read_csv("./data_temp/val.csv").sample(frac=1).reset_index(drop=True)

# define labels
labels = [label for label in set(train_df['label'])]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


# Convert labels in the DataFrame to encoded values using label2id dictionary
# train_df['label'] = train_df['label'].apply(lambda x: label2id[x])
# test_df['label'] = test_df['label'].apply(lambda x: label2id[x])
# val_df['label'] = val_df['label'].apply(lambda x: label2id[x])

new_train_df = pd.get_dummies(train_df['label']).astype(bool)
new_test_df = pd.get_dummies(test_df['label']).astype(bool)
new_val_df = pd.get_dummies(val_df['label']).astype(bool)

new_train_df['text'] = train_df['text']
new_test_df['text'] = test_df['text']
new_val_df['text'] = val_df['text']


# define model
pre_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                            problem_type="multi_label_classification", 
                                                            num_labels=len(label2id),
                                                            id2label=id2label,
                                                            label2id=label2id)

def preprocess_text(examples):
    text = examples["text"]
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True)
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(text), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()
    return encoding

def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

     

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

def main():
    dataset = DatasetDict()
    dataset["train"] = Dataset.from_pandas(new_train_df)
    dataset["test"] = Dataset.from_pandas(new_test_df)
    dataset["val"] = Dataset.from_pandas(new_val_df)

    tokenized_data = dataset.map(preprocess_text, batched=True)
    tokenized_data.set_format("torch")

    # define arguments
    training_args = TrainingArguments(
        f"bert-finetuned-sem_eval",
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
    )

    trainer = Trainer(
        pre_model,
        training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    main()

