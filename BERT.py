from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

line = {"Your Netflix membership will continue until terminated.",
        "You can cancel your Netflix membership at any time, and you will continue to have access to the Netflix service through the end of your billing period.",
        "You may access the Netflix content primarily within the country in which you have established your account.",
        "You may automatically receive updated versions of the Netflix software.",
        "Customer Service may best be able to assist you by using a remote access support tool."}

for i in line:
    sentence = i
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    sentence_embedding = outputs.last_hidden_state[:, 0, :].squeeze()

    print(sentence_embedding[:5])
