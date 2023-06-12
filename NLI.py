from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NLI_German():
    def __init__(self):
        """Initialize the model and the tokenizer"""
        model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
        self.tokenizer_nli = AutoTokenizer.from_pretrained(model_name)
        self.model_nli = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    def check_nli(self,premise, hypothesis):
        input = self.tokenizer_nli(premise, hypothesis, truncation=False, return_tensors="pt")
        output = self.model_nli(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: float(pred) for pred, name in zip(prediction, label_names)}
        return prediction

