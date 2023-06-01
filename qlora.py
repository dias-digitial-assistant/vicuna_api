import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, LlamaTokenizer
import os
from datasets import load_dataset
import transformers
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
import sys
tokenizer = None
cutoff_len = 2048
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def tell_prefix(segment):
    prefix = "\n\n### Human:" if segment["from"] == "human" else "\n\n### Assistant:"
    return prefix

def generate_prompt(data_point):
    text_all = ""
    for elems in data_point["conversations"]:
        text_all += tell_prefix(elems)+elems['value']
    return text_all

def tokenize(prompt,add_eos_token=True):
    global tokenizer, cutoff_len
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None
    )

    return result

def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)

def train():
    global tokenizer, cutoff_len
    model_id ="timdettmers/guanaco-65b-merged"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    config = LoraConfig(
    r=64, 
    lora_alpha=16, 
    target_modules= [
    "gate_proj",
    "down_proj",
    "up_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj"
    ], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
    )



    device_map = "auto"
    batch_size = 128
    micro_batch_size = 4
    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    wandb_project: str = ""
    wandb_run_name: str = ""
    wandb_watch: str = "" # options: false | gradients | all
    wandb_log_model: str = ""  # options: false | true
    resume_from_checkpoint: str = None
    ddp = world_size != 1
    epochs = 5
    learning_rate = 3e-4  # the Karpathy constant
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model
    tokenizer = LlamaTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,device_map=device_map)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    cutoff_len = 2048
    tokenizer.pad_token_id = 0
    print("Beginning of Sentence: ",tokenizer.bos_token_id)
    print("End of Sentence: ",tokenizer.eos_token_id)
    print("Pad Token: ",tokenizer.pad_token_id)
    print("Unknown Token: ",tokenizer.unk_token_id)

    data_path = "/workspace/ReACT_paradigm_DIAS_merged_equal.json"
    output_dir = "/workspace/lora-guanaco65b-DIAS_merged_equal"
    val_set_size = 960

    data = load_dataset("json", data_files=data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = None

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=1,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=40 if val_set_size > 0 else None,
        save_steps=40,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        report_to="wandb",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False
    trainer.train()

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")

if __name__ == "__main__":
    train()
