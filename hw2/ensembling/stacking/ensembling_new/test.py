from transformers import AutoFeatureExtractor, ResNetForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
# print(model)
for name, module in model.named_children():
    if not name.startswith('params'):
        print(name)
        #print(module)
        print('------')

inputs = feature_extractor(image, return_tensors="pt")
print(inputs.keys())
with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])