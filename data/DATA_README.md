# Data README

## Italian Corpus
- File: italian_corpus.txt
- Source: Italian Wikipedia (streaming dataset 2023-11-01)
- Number of sentences: 2000
- Filters: only sentences with more than 5 words, excluding headings/titles
- License: Wikimedia content, CC BY-SA 3.0

## English Corpus
- File: english_corpus.txt
- Generated via machine translation from italian_corpus.txt
- Model: Helsinki-NLP/opus-mt-it-en
- Number of sentences: 2000
- Use: reference for BLEU during translation tests
