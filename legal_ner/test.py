# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Universal-NER/UniNER-7B-type")
test = "Text: What is a legal framework by Judge Solos Poppier"
entity = "JUDGE"
prompt = f"What describes {entity} in the text"
print(pipe(test))