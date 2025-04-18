# 🚧 Problem: `python: not found` and spaCy Model `en_core_web_sm` Missing

## ❗ Error Log

```bash
sh: 1: python: not found
Traceback (most recent call last):
  File "/home/shahanahmed/Custom_Chatbot_UI/rag_system.py", line 45, in <module>
    nlp = spacy.load("en_core_web_sm")
  ...
OSError: [E050] Can't find model 'en_core_web_sm'. It doesn't seem to be a Python package or a valid path to a data directory.
