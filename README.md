**You can get the ui file and the classifer contents from this repo but the project is incomplete without the translator and summarizer model which you can find at the 
following hugging face link https://huggingface.co/garvitupdy/News-NLP-Project/tree/main.**

   # 📰 News AI System — Multi-Task NLP Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**A comprehensive NLP-powered news processing system with 6 integrated modules for
classification, summarization, translation, article generation, and database management.**

[Features](#-features) •
[Architecture](#-architecture) •
[Installation](#-installation) •
[Usage](#-usage) •
[Models](#-models) •
[Screenshots](#-screenshots) •
[API Reference](#-api-reference)

</div>

---

## 🌟 Features

### 📊 News Classification
- **4-category classification**: World, Sports, Business, Sci-Fi
- **Multi-layer analysis**: Sentiment, Named Entities, Readability
- **Explainability**: Shows which words influenced the prediction
- **Sub-category suggestions**: Detailed topic identification
- **Confidence visualization**: Interactive bar charts

### 📝 News Summarization
- **Adjustable length slider**: 10% to 60% of original article
- **Smart chunking**: Handles articles of any length
- **Bullet point extraction**: Key points with scoring algorithm
- **Summary translation**: Translate summary to Hindi/German/Spanish/French
- **Auto-classification**: Classify article during summarization
- **Quality metrics**: Reduction percentage, compression ratio

### 🌐 Bidirectional Translation
- **5 languages supported**: English, Hindi, German, Spanish, French
- **Any direction**: English ↔ Hindi/German/Spanish/French
- **Long text handling**: Automatic chunking with progress tracking
- **Integrated summarization**: Summarize translated output
- **Integrated classification**: Classify translated output
- **Translation storage**: English translations saved for future use



### 📚 Article Database
- **SQLite storage**: Up to 100 articles with persistent storage
- **CRUD operations**: Add, Browse, Search, Edit, Delete
- **Smart filtering**: By category, language, search query
- **Translation linking**: Stores both original and translated content
- **Results tracking**: All task results linked to source articles
- **Import/Export**: CSV import and export functionality

### 📜 History & Downloads
- **Session history**: Last 5 results per task type
- **PDF export**: Professional reports via ReportLab
- **CSV export**: Data-friendly format for all results
- **Bulk export**: Download entire database and results

---

## 🏗 Architecture

news-ai-system/
│
├── ui.py # Main Streamlit application (~2500 lines)
├── news_articles.db # SQLite database (auto-created)
├── requirements.txt # Python dependencies
├── README.md # This file
│
├── classifier/ # Classification model
│ ├── nb_model.pkl # Trained Naive Bayes model
│ └── tfidf_vectorizer.pkl # TF-IDF vectorizer
│
├── summarizer/ # Summarization model (BART)
│ ├── config.json # Model configuration
│ ├── pytorch_model.bin # Model weights (~1.6 GB)
│ ├── tokenizer_config.json # Tokenizer configuration
│ ├── vocab.json # Vocabulary
│ └── merges.txt # BPE merges
│
├── translator/ # Translation model (NLLB-200)
│ ├── config.json # Model configuration
│ ├── pytorch_model.bin # Model weights (~1.2 GB)
│ │ OR model.safetensors
│ ├── tokenizer_config.json # Tokenizer configuration
│ ├── sentencepiece.bpe.model # SentencePiece tokenizer
│ └── special_tokens_map.json # Special tokens



### System Architecture Diagram
┌─────────────────────────────────────────────────────────────┐
│ STREAMLIT FRONTEND │
│ │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│ │Classify │ │Summarize │ │Translate │ │Generate │ │
│ │ Tab │ │ Tab │ │ Tab │ │ Tab │ │
│ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ │
│ │ │ │ │ │
│ ┌────▼─────────────▼────────────▼─────────────▼──────┐ │
│ │ ARTICLE SELECTOR │ │
│ │ (Manual Input / Database Pick) │ │
│ └────┬──────────────────────────────────────┬────────┘ │
│ │ │ │
│ ┌────▼────┐ ┌─────────┐ ┌─────────┐ ┌──▼───────┐ │
│ │Naive │ │ BART │ │ NLLB │ │  │ │
│ │Bayes + │ │Summarize│ │Translate│ │ │ │
│ │TF-IDF │ │ │ │ │ │ │ │
│ └────┬────┘ └────┬────┘ └────┬────┘ └────┬─────┘ │
│ │ │ │ │ │
│ ┌────▼────────────▼────────────▼─────────────▼──────┐ │
│ │ SESSION STATE │ │
│ │ (Results stored for save button) │ │
│ └────┬──────────────────────────────────────┬────────┘ │
│ │ │ │
│ ┌────▼──────────────┐ ┌───────────────────▼──────┐ │
│ │ DISPLAY RESULTS │ │ SAVE TO DATABASE │ │
│ │ Charts, Tables, │ │ Articles + Results │ │
│ │ Badges, Cards │ │ SQLite │ │
│ └───────────────────┘ └──────────────────────────┘ │
│ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ EXPORT OPTIONS │ │
│ │ PDF (ReportLab) | CSV (Pandas) │ │
│ └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘


### Database Design Architecture
┌─────────────────────────────────────────────────────────────┐
│ SQLite DATABASE │
│ │
│ ┌─────────────────────┐ ┌────────────────────────┐ │
│ │ articles table │ │ results table │ │
│ │ │ │ │ │
│ │ id │ │ id │ │
│ │ title │ │ article_id (FK) │ │
│ │ content │ │ task_type │ │
│ │ translated_content │ │ result_data │ │
│ │ category │ │ confidence │ │
│ │ language │ │ metadata (JSON) │ │
│ │ tags │ │ created_at │ │
│ │ word_count │ └────────────────────────┘ │
│ │ created_at │ │
│ │ updated_at │ One article → Many results │
│ └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘


---

## 📦 Installation

### Prerequisites

- **Python**: 3.8 or higher
- **RAM**: Minimum 8 GB (16 GB recommended for all models)
- **Storage**: ~7 GB for all model files
- **GPU**: Optional (CPU inference supported)


***Classification Workflow***
1. Enter title + description (or pick from DB)
2. Click "Analyze & Classify"
3. View results:
   ├── Category prediction + confidence
   ├── Sentiment analysis (Positive/Negative/Neutral)
   ├── Named entities (People, Orgs, Locations)
   ├── Readability score (Flesch Reading Ease)
   ├── Influencing words (why this category)
   └── Sub-category suggestions
4. Download PDF/CSV or save to database

***Summarization Workflow***
   1. Enter article (50+ words) or pick from DB
2. Adjust settings:
   ├── Summary length slider (10%-60%)
   ├── Enable/disable translation
   ├── Enable/disable classification
   └── Set bullet point count (3/5/7/10)
3. Click "Generate Complete Analysis"
4. View results:
   ├── Side-by-side comparison
   ├── Bullet points
   ├── Translated summary (if enabled)
   └── Classification badge (if enabled)
5. Download or save to database

***Translation Workflow***
1. Select direction:
   ├── Other Language → English
   └── English → Other Language
2. Choose language (Hindi/German/Spanish/French)
3. Enter text or pick from database
4. Optional: Enable summarization/classification
5. Click "Translate"
6. View results:
   ├── Side-by-side translation
   ├── Summary (if enabled)
   └── Classification (if enabled)
7. Save: source + translation stored in database

***Database Workflow***
1. Add Article:
   ├── Title, Content, Category, Language, Tags
   └── Capacity: up to 100 articles

2. Browse & Search:
   ├── Search by title/content/tags
   ├── Filter by category/language
   └── View past results per article

3. View Results:
   ├── All classification/summarization/translation results
   ├── Filter by task type
   └── Export all results as CSV

4. Edit/Delete:
   ├── Modify article content/metadata
   └── Delete individual or all articles

5. Import/Export:
   ├── Export database as CSV
   └── Import articles from CSV

  ***Classification Model***
Property	Value
Algorithm	Multinomial Naive Bayes
Vectorizer	TF-IDF (Term Frequency-Inverse Document Frequency)
Categories	World, Sports, Business, Sci-Fi
Input	Title + Description
Output	Category + Confidence Scores
Size	~50 MB

***Summarization Model***
Property	Value
Model	BART-Large-CNN
Architecture	Encoder-Decoder Transformer
Parameters	~400M
Max Input	1024 tokens
Output Range	Dynamic (10%-60% of input)
Chunking	Automatic for long articles
Size	~2.3 GB

***Translation Model***
Property	Value
Model	NLLB-200 Distilled 600M
Architecture	Encoder-Decoder Transformer
Parameters	~600M
Languages	200+ (5 used: EN, HI, DE, ES, FR)
Direction	Bidirectional
Max Input	512 tokens per chunk
Size	~3.4 GB

***🛠 Technical Details
NLP Techniques (No Extra Models)***
Feature	                  Technique	                     Library
Keyword Extraction	      Word Frequency + Stop Words	   Python collections.Counter
Sentiment Analysis	      Lexicon-based scoring	         Python re, set
Named Entity Recognition	Regex pattern matching	         Python re
Readability Score	         Flesch Reading Ease formula	   Python math
Bullet Point Extraction	   Sentence scoring algorithm	      Python (custom)
Influence Analysis	      TF-IDF weight inspection	      scikit-learn
Sub-category Suggestion	    Keyword matching	               Python dict
Language Detection	      Unicode range analysis	         Python re


***Database Schema***
-- Articles table
CREATE TABLE articles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    title               TEXT NOT NULL,
    content             TEXT NOT NULL,
    translated_content  TEXT DEFAULT '',
    category            TEXT DEFAULT 'Uncategorized',
    language            TEXT DEFAULT 'English',
    tags                TEXT DEFAULT '',
    word_count          INTEGER DEFAULT 0,
    created_at          TEXT,
    updated_at          TEXT
);

-- Results table
CREATE TABLE results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id      INTEGER NOT NULL,
    task_type       TEXT NOT NULL,    -- 'classification'|'summarization'|'translation'
    result_data     TEXT NOT NULL,    -- actual result text
    confidence      REAL DEFAULT 0,  -- confidence score
    metadata        TEXT DEFAULT '{}', -- JSON with extra info
    created_at      TEXT,
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);
