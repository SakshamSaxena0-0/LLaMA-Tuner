## Step 1: Import necessary modules

import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from huggingface_hub import notebook_login
from datasets import Dataset
import pandas as pd
from IPython.display import display
from google.colab import userdata
from peft import LoraConfig, get_peft_model, TaskType



## Step 2:- Log in to Hugging Face (replace with your actual token)
notebook_login()

## Step 3:-Load the dataset
import pandas as pd
df = pd.read_csv("file_path'.csv")
print("Shape of dataset:", df.shape)
display(df.head(5))

## Step 4:- Separate inbound (customer) and outbound (brand)
inbound_df = df[df["inbound"] == True]
outbound_df = df[df["inbound"] == False]

## Step 5:- Merge to match inbound tweets with brand replies
merged_df = pd.merge(
    inbound_df,
    outbound_df,
    left_on="tweet_id",
    right_on="in_response_to_tweet_id",
    suffixes=("_customer", "_brand")
)

## Step 6:-Keep relevant columns
merged_df = merged_df[["tweet_id_customer", "text_customer",
                       "tweet_id_brand", "text_brand"]]
display(merged_df.head())

def build_chat_example(row):
    return {
        "prompt": f"User: {row['text_customer']}\nAssistant:",
        "response": row["text_brand"]
    }

paired_data = merged_df.apply(build_chat_example, axis=1).to_list()

## Step 7:- Create and split the dataset
from datasets import Dataset
dataset = Dataset.from_list(paired_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(train_dataset[0])

## Step 8:- Load the model and tokenizer
model_id = "meta-llama/Meta-Llama-3-8B"

#create the 4 bit config
import bitsandbytes as bnb


# Instead of importing from transformers.models.llama, import from transformers directly
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant = True,
    bnb_4bit_quant_type = "nf4",
    bnb_4bit_compute_dtype = torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id) #remove token argument
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map = "auto",
    quantization_config = bnb_config,
    # remove token argument, this was causing the previous error
)

# Create the pipeline now with the loaded model and tokenizer
llama_pipeline = pipeline(
    "text-generation",
    model=model, #pass the loaded model
    tokenizer=tokenizer, #pass the loaded tokenizer
    torch_dtype=torch.bfloat16,
    device_map="auto" #changed from 'cuda' to 'auto' as cuda does not work on all machines
)




lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]  # typical LLaMA layers
)

model = get_peft_model(model, lora_config)
print("LoRA layers added to the model!")
