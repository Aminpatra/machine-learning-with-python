# Natural Language Processing (NLP)

Techniques for teaching computers to understand, interpret, and generate human language.

## Core NLP Tasks

**Text Classification** - Categorize text (spam detection, sentiment analysis)  
**Named Entity Recognition (NER)** - Identify people, places, organizations  
**Text Generation** - Create human-like text (chatbots, summarization)  
**Machine Translation** - Translate between languages  
**Question Answering** - Answer questions from text  
**Text Summarization** - Extract key information  
**Sentiment Analysis** - Determine emotion/opinion in text  

## Text Preprocessing Pipeline

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenization
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Stemming or Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# Example
text = "The cats are running quickly in the gardens!"
clean_text = preprocess_text(text)
print(clean_text)  # "cat running quickly garden"
```

## Text Representation

### 1. Bag of Words (BoW)
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love Python programming"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(corpus)

# TF-IDF weights common words less, rare words more
print(X_tfidf.toarray())
```

### 3. Word Embeddings (Word2Vec, GloVe)
```python
# Using pre-trained GloVe embeddings
import gensim.downloader as api

# Load pre-trained model
glove_model = api.load('glove-wiki-gigaword-100')

# Get word vector
vector = glove_model['king']
print(f"Vector shape: {vector.shape}")

# Find similar words
similar = glove_model.most_similar('king', topn=5)
print(similar)

# Word analogies: king - man + woman = queen
result = glove_model.most_similar(positive=['king', 'woman'], 
                                   negative=['man'], topn=1)
print(result)  # queen
```

### 4. Sentence Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "I love machine learning",
    "Machine learning is great",
    "I hate bugs in code"
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)

# Calculate similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(embeddings)
print(similarity)
```

## Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Example: Sentiment analysis
texts = ["I love this movie", "Terrible film", "Great acting", "Awful plot"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Vectorize
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train classifier
model = MultinomialNB()  # or LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predict
predictions = model.predict(X_test_tfidf)
print(classification_report(y_test, predictions))
```

## Named Entity Recognition (NER)

```python
import spacy

# Load pre-trained model
nlp = spacy.load('en_core_web_sm')

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California."
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f"{ent.text}: {ent.label_}")

# Output:
# Apple Inc.: ORG
# Steve Jobs: PERSON
# Cupertino: GPE
# California: GPE
```

## Text Generation

### Using Transformers (GPT-2)
```python
from transformers import pipeline

# Load pre-trained model
generator = pipeline('text-generation', model='gpt2')

prompt = "Artificial intelligence will"
result = generator(prompt, max_length=50, num_return_sequences=1)
print(result[0]['generated_text'])
```

## Sentiment Analysis

```python
from transformers import pipeline

# Pre-trained sentiment analyzer
sentiment_analyzer = pipeline('sentiment-analysis')

texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, nothing special."
]

for text in texts:
    result = sentiment_analyzer(text)[0]
    print(f"{text}: {result['label']} ({result['score']:.3f})")
```

## Advanced: Using Transformers (BERT)

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                       num_labels=2)

# Tokenize text
text = "This movie is amazing!"
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
print(predictions)
```

## Fine-tuning Pre-trained Models

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset('imdb')

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Tokenize dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test']
)

# Train
trainer.train()
```

## Common NLP Libraries

### NLTK - Natural Language Toolkit
```python
import nltk

# Tokenization
from nltk.tokenize import word_tokenize, sent_tokenize
words = word_tokenize("Hello world. How are you?")
sentences = sent_tokenize("Hello world. How are you?")

# POS Tagging
from nltk import pos_tag
tags = pos_tag(words)
print(tags)  # [('Hello', 'NNP'), ('world', 'NN'), ...]
```

### spaCy - Industrial NLP
```python
import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# Entities, POS tags, dependencies all in one
for token in doc:
    print(f"{token.text}: POS={token.pos_}, DEP={token.dep_}")
```

### Transformers - State-of-the-art Models
```python
from transformers import pipeline

# Quick inference for various tasks
classifier = pipeline('sentiment-analysis')
ner = pipeline('ner')
qa = pipeline('question-answering')
summarizer = pipeline('summarization')
translator = pipeline('translation_en_to_fr')
```

## Key Concepts

**Tokenization:** Breaking text into words/subwords  
**Stopwords:** Common words (the, is, at) often removed  
**Stemming:** Reduce to root (running → run)  
**Lemmatization:** Reduce to dictionary form (better → good)  
**N-grams:** Sequences of N words (bigrams, trigrams)  
**Word Embeddings:** Dense vector representations of words  
**Attention Mechanism:** Focus on relevant parts of text  
**Transfer Learning:** Use pre-trained models, fine-tune for task  

## Popular Pre-trained Models

**BERT** - Bidirectional Encoder (good for understanding)  
**GPT** - Generative Pre-trained Transformer (good for generation)  
**T5** - Text-to-Text Transfer Transformer (unified framework)  
**RoBERTa** - Optimized BERT  
**DistilBERT** - Smaller, faster BERT  
**ELECTRA** - Efficient pre-training  
**XLNet** - Permutation language modeling  

## Evaluation Metrics

**Classification:** Accuracy, Precision, Recall, F1-Score  
**Generation:** BLEU, ROUGE, Perplexity  
**Translation:** BLEU score  
**Summarization:** ROUGE score  
**Question Answering:** Exact Match, F1  

## Best Practices

1. **Start with pre-trained models** - Don't train from scratch
2. **Clean your data** - Remove noise, handle special characters
3. **Use proper tokenization** - Subword tokenization (BPE, WordPiece)
4. **Handle class imbalance** - Oversample, undersample, or weight classes
5. **Cross-validate** - Always validate on held-out data
6. **Monitor overfitting** - Use early stopping
7. **Try data augmentation** - Back-translation, synonym replacement
8. **Use appropriate metrics** - F1 for imbalanced, BLEU for generation
9. **Fine-tune carefully** - Small learning rate, few epochs
10. **Consider computational costs** - Smaller models often good enough

## Common Pitfalls

❌ **Not preprocessing** - Garbage in, garbage out  
❌ **Ignoring context** - Words mean different things in different contexts  
❌ **Over-preprocessing** - Removing too much information  
❌ **Wrong tokenization** - Splitting words incorrectly  
❌ **Imbalanced data** - Biased model predictions  
❌ **Not using pre-trained models** - Wasting time and resources  
❌ **Ignoring domain-specific terms** - Custom vocabulary matters  

## Quick Start Template

```python
# 1. Load pre-trained model
from transformers import pipeline

classifier = pipeline('text-classification', 
                      model='distilbert-base-uncased-finetuned-sst-2-english')

# 2. Predict
text = "This is an amazing product!"
result = classifier(text)
print(result)

# 3. Fine-tune for your task (if needed)
# - Prepare labeled data
# - Use Trainer API from transformers
# - Evaluate and iterate
```

## When to Use What

**TF-IDF + Classical ML:**
✅ Small datasets (< 10k samples)  
✅ Fast training/inference needed  
✅ Interpretability important  

**Word Embeddings + Deep Learning:**
✅ Medium datasets (10k-100k)  
✅ Need semantic understanding  
✅ Good accuracy/speed trade-off  

**Transformers (BERT, GPT):**
✅ Large datasets (100k+)  
✅ State-of-the-art accuracy needed  
✅ Can afford computational cost  

---

*Start with pre-trained transformers and fine-tune. Only build from scratch if you have massive domain-specific data and computational resources!*
