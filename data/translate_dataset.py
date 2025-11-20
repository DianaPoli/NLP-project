from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-it-en", device=0)

with open("data/italian_corpus.txt", "r", encoding="utf-8") as f:
    italian_sentences = [line.strip() for line in f if line.strip()]

translations = [translator(s, max_length=512)[0]['translation_text'] for s in italian_sentences]

with open("data/english_corpus.txt", "w", encoding="utf-8") as f:
    for t in translations:
        f.write(t + "\n")

print("translate complete")
