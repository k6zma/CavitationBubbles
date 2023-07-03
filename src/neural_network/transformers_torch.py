import os
import datetime
import numpy as np
import pandas as pd
from datasets import load_dataset
import os
from PIL import Image
from transformers import AutoImageProcessor, DefaultDataCollator, AutoModelForImageClassification, TrainingArguments, Trainer
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
import evaluate
import torch
from tqdm import tqdm

data_folder = "../../data/data_base"
dataset = load_dataset(data_folder)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

checkpoint = 'checkpoint-3600'
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

dataset = dataset.with_transform(transforms)

data_collator = DefaultDataCollator()

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)
model = model.to("cuda")

training_args = TrainingArguments(
    output_dir="my_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=100,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
)

trainer.train()

ds = load_dataset(data_folder)

model = AutoModelForImageClassification.from_pretrained('checkpoint-3600')
model = model.to("cuda")
pred = []
label = []
image = []
image_processor = AutoImageProcessor.from_pretrained("checkpoint-3600")
for i in tqdm(range(len(ds["test"]["image"]))):
    image = ds["test"]["image"][i]
    inputs = image_processor(image, return_tensors="pt")
    inputs = inputs.to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
    pred.append(predicted_label)
    label.append(ds["test"]["label"][i])

k = 0
j = 0
for j in tqdm(range(len(pred))):
    if label[j] == pred[j]:
        k += 1
res = (k * 100) / len(label)
print(res)