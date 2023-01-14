from datasets import load_dataset,load_metric
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,Seq2SeqTrainingArguments,DataCollatorForSeq2Seq,Seq2SeqTrainer
import numpy as np

metric=load_metric("BLEU.py")
max_input_length = 64
max_target_length = 64
src_lang = "zh"
tag_lang = "en"
model_path = "Helsinki-NLP/opus-mt-zh-en"
# model_path = "translations/checkpoint-1500/"
batch_size = 4
learning_rate = 1e-5
output_dir = "translations"

def preprocess_function(examples):
    inputs = [eval(ex)[src_lang] for ex in examples["text"]]
    targets = [eval(ex)[tag_lang] for ex in examples["text"]]
    model_inputs=tokenizer(inputs,max_length=max_input_length,truncation=True)
    with tokenizer.as_target_tokenizer():
        labels=tokenizer(targets,max_length=max_target_length,truncation=True)
    model_inputs["labels"]=labels["input_ids"]
    return model_inputs    

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    print(result)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

train_dataset = load_dataset("text",data_files="data/train.txt")
val_dataset = load_dataset("text",data_files="data/val.txt")

tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
tokenized_val_datasets = val_dataset.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

args = Seq2SeqTrainingArguments(
    auto_find_batch_size = True,
    learning_rate = learning_rate,
    output_dir = output_dir,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_train_datasets["train"],
    eval_dataset=tokenized_val_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.predict(test_dataset=tokenized_val_datasets["train"])