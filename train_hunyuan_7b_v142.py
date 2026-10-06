#prompt B
#focal loss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VER=142
model_name = "/kaggle/input/hunyuan/transformers/hunyuan-7b-instruct/1" 
data_path = '/kaggle/input/map-charting-student-math-misunderstandings/train.csv'
debug = False
EPOCHS = 3

DIR = f"ver_{VER}"
os.makedirs(DIR, exist_ok=True)

import pandas as pd, numpy as np
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train = pd.read_csv(data_path)
if debug:
    train = train.sample(1000).reset_index(drop=True)
train.Misconception = train.Misconception.fillna('NA')
train['target'] = train.Category+":"+train.Misconception
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")

idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# save tokenizer to load offline during inference
MAX_LEN = 256

def format_input(row):
    return (
            "Analyze the student's reasoning for the math problem below.\n"
            "### CONTEXT ###\n"
            f"Question: {row['QuestionText']}\n"
            f"Student's Answer: {row['MC_Answer']} ({'Correct' if row['is_correct'] else 'Incorrect'})\n"
            "### STUDENT EXPLANATION ###\n"
            f"{row['StudentExplanation']}\n"
            "### IDENTIFIED MISCONCEPTION ###\n"
    )

train['text'] = train.apply(format_input,axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )

# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)

# Convert to Hugging Face Dataset
COLS = ['text','label']
train_ds = Dataset.from_pandas(train_df[COLS])
val_ds = Dataset.from_pandas(val_df[COLS])

# Tokenization function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Set format for PyTorch
columns = ['input_ids', 'attention_mask', 'label']
train_ds.set_format(type='torch', columns=columns)
val_ds.set_format(type='torch', columns=columns)

import os
import copy
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BitsAndBytesConfig,
    Gemma2ForSequenceClassification,
    GemmaTokenizerFast,
    Gemma2Config,
    PreTrainedTokenizerBase, 
    EvalPrediction,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

USE_QLORA=False
qlora = {}
if USE_QLORA:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4", #nf4 or fp4
        bnb_4bit_use_double_quant = False,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules = ["score"]
    )
    qlora['quantization_config'] = bnb_config
    print("Using QLoRA")

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=n_classes,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    **qlora
)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=512,
    lora_alpha=64,
    # only target self-attention
    target_modules=["q_proj", "k_proj", "v_proj",
                    "down_proj","up_proj","o_proj","gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    modules_to_save=["score",]
)

from peft import PeftModel
#model = PeftModel.from_pretrained(model, model_name)
#model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir = f"./{DIR}",
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps", #no for no saving 
    lr_scheduler_type="cosine",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=8,
    learning_rate=1e-4,
    weight_decay=0.5,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    eval_steps=1000,
    save_total_limit=1,
    metric_for_best_model="map@3",
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="none",
    bf16=True, # TRAIN WITH BF16 IF LOCAL GPU IS NEWER GPU          
    fp16=False, # INFER WITH FP16 BECAUSE KAGGLE IS T4 GPU
)

# CUSTOM MAP@3 METRIC

from sklearn.metrics import average_precision_score

def compute_map3(eval_pred):
    logits, labels = eval_pred
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    
    top3 = np.argsort(-probs, axis=1)[:, :3]  # Top 3 predictions
    match = (top3 == labels[:, None])

    # Compute MAP@3 manually
    map3 = 0
    for i in range(len(labels)):
        if match[i, 0]:
            map3 += 1.0
        elif match[i, 1]:
            map3 += 1.0 / 2
        elif match[i, 2]:
            map3 += 1.0 / 3
    return {"map@3": map3 / len(labels)}
# 这是一个简化的Focal Loss实现
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
        
class CustomTrainer(Trainer):
    # 在这里加上 **kwargs
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # 定义损失函数
        loss_fct = FocalLoss(gamma=2.8) # gamma=2 是一个常见的Focal Loss超参数
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
        
# Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_map3,
)

trainer.train()

trainer.save_model(f"./ver_{VER}")      
tokenizer.save_pretrained(f"./ver_{VER}")
