from transformers import pipeline

pipe = pipeline("text2text-generation", model="m1k3wn/nidra-v2")
result = pipe("I was flying over a forest")
print(result)