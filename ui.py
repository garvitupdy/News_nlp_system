import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from collections import Counter
import pickle
import re
import time
import io
import os
from pathlib import Path


from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors


try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        BartForConditionalGeneration,
        BartTokenizer
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    

import sqlite3


class NewsDatabase:
    
    
    def __init__(self, db_path="news_articles.db"):
        self.db_path = db_path
        self.max_articles = 100
        self._init_db()
    
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self):
        
        conn = self._get_conn()
        
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS articles (
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
            )
        ''')
        
        cursor = conn.execute("PRAGMA table_info(articles)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'translated_content' not in columns:
            conn.execute(
                "ALTER TABLE articles ADD COLUMN "
                "translated_content TEXT DEFAULT ''"
            )
        
        
        conn.execute('''
            CREATE TABLE IF NOT EXISTS results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id      INTEGER NOT NULL,
                task_type       TEXT NOT NULL,
                result_data     TEXT NOT NULL,
                confidence      REAL DEFAULT 0,
                metadata        TEXT DEFAULT '{}',
                created_at      TEXT,
                FOREIGN KEY (article_id) REFERENCES articles(id)
                    ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
   
    
    def add_article(self, title, content, category="Uncategorized",
                    language="English", tags="", translated_content=""):
        
        conn = self._get_conn()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM articles")
        count = cur.fetchone()[0]
        if count >= self.max_articles:
            conn.close()
            return False, f"Database full! Maximum {self.max_articles} articles.", None
        
        cur.execute("SELECT id FROM articles WHERE title = ?", (title,))
        if cur.fetchone():
            conn.close()
            return False, "An article with this title already exists.", None
        
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        word_count = len(content.split())
        
        cur.execute('''
            INSERT INTO articles 
            (title, content, translated_content, category, language, 
             tags, word_count, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (title, content, translated_content, category, language,
              tags, word_count, now, now))
        
        article_id = cur.lastrowid
        conn.commit()
        conn.close()
        return True, f"Article added! ID: {article_id} ({count + 1}/{self.max_articles})", article_id
    
    def get_all(self):
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM articles ORDER BY created_at DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_by_id(self, article_id):
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    
    def get_by_category(self, category):
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM articles WHERE category = ? "
            "ORDER BY created_at DESC",
            (category,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_by_language(self, language):
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM articles WHERE language = ? "
            "ORDER BY created_at DESC",
            (language,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_english_content_articles(self):
        
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT * FROM articles 
            WHERE language = 'English' 
               OR (translated_content IS NOT NULL 
                   AND translated_content != '')
            ORDER BY created_at DESC
        ''').fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def search(self, query):
        conn = self._get_conn()
        pattern = f"%{query}%"
        rows = conn.execute('''
            SELECT * FROM articles 
            WHERE title LIKE ? OR content LIKE ? 
               OR tags LIKE ? OR translated_content LIKE ?
            ORDER BY created_at DESC
        ''', (pattern, pattern, pattern, pattern)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    
    def get_count(self):
        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]
        conn.close()
        return count
    
    def update_article(self, article_id, title, content, category,
                       language, tags, translated_content=""):
        conn = self._get_conn()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        word_count = len(content.split())
        conn.execute('''
            UPDATE articles 
            SET title=?, content=?, translated_content=?, 
                category=?, language=?, tags=?, 
                word_count=?, updated_at=?
            WHERE id=?
        ''', (title, content, translated_content, category, language,
              tags, word_count, now, article_id))
        conn.commit()
        conn.close()
        return True
    
    def update_category(self, article_id, category):
        conn = self._get_conn()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            "UPDATE articles SET category=?, updated_at=? WHERE id=?",
            (category, now, article_id)
        )
        conn.commit()
        conn.close()
    
    def update_translated_content(self, article_id, translated_text):
        
        conn = self._get_conn()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        conn.execute(
            "UPDATE articles SET translated_content=?, updated_at=? "
            "WHERE id=?",
            (translated_text, now, article_id)
        )
        conn.commit()
        conn.close()
    
    def delete_article(self, article_id):
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM articles WHERE id = ?", (article_id,)
        )
        conn.commit()
        conn.close()
        return True
    
    def clear_all(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM results")
        conn.execute("DELETE FROM articles")
        conn.commit()
        conn.close()
        return True
    
    
    def save_result(self, article_id, task_type, result_data,
                    confidence=0, metadata=None):
        import json
        conn = self._get_conn()
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        meta_str = json.dumps(metadata) if metadata else '{}'
        conn.execute('''
            INSERT INTO results 
            (article_id, task_type, result_data, confidence, 
             metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (article_id, task_type, result_data, confidence,
              meta_str, now))
        conn.commit()
        conn.close()
        return True, "Result saved!"
    
    def get_results_for_article(self, article_id):
        import json
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT * FROM results 
            WHERE article_id = ? 
            ORDER BY created_at DESC
        ''', (article_id,)).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d['metadata'] = json.loads(d['metadata'])
            except:
                d['metadata'] = {}
            results.append(d)
        return results
    
    def get_results_by_task(self, task_type):
        import json
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT r.*, a.title as article_title, a.word_count
            FROM results r
            JOIN articles a ON r.article_id = a.id
            WHERE r.task_type = ?
            ORDER BY r.created_at DESC
        ''', (task_type,)).fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d['metadata'] = json.loads(d['metadata'])
            except:
                d['metadata'] = {}
            results.append(d)
        return results
    
    def get_all_results(self):
        import json
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT r.*, a.title as article_title, 
                   a.category, a.language, a.word_count
            FROM results r
            JOIN articles a ON r.article_id = a.id
            ORDER BY r.created_at DESC
        ''').fetchall()
        conn.close()
        results = []
        for r in rows:
            d = dict(r)
            try:
                d['metadata'] = json.loads(d['metadata'])
            except:
                d['metadata'] = {}
            results.append(d)
        return results
    
    def get_result_count(self):
        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM results"
        ).fetchone()[0]
        conn.close()
        return count
    
    def get_result_count_by_task(self, task_type):
        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM results WHERE task_type = ?",
            (task_type,)
        ).fetchone()[0]
        conn.close()
        return count
    
    def delete_result(self, result_id):
        conn = self._get_conn()
        conn.execute(
            "DELETE FROM results WHERE id = ?", (result_id,)
        )
        conn.commit()
        conn.close()
        return True
    
    def clear_results(self, task_type=None):
        conn = self._get_conn()
        if task_type:
            conn.execute(
                "DELETE FROM results WHERE task_type = ?",
                (task_type,)
            )
        else:
            conn.execute("DELETE FROM results")
        conn.commit()
        conn.close()
        return True
    
    def get_article_result_summary(self, article_id):
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT task_type, COUNT(*) as count, 
                   MAX(created_at) as latest
            FROM results 
            WHERE article_id = ?
            GROUP BY task_type
        ''', (article_id,)).fetchall()
        conn.close()
        return {r['task_type']: {
            'count': r['count'],
            'latest': r['latest']
        } for r in rows}
    
    def export_csv(self):
        conn = self._get_conn()
        df = pd.read_sql_query(
            "SELECT * FROM articles ORDER BY created_at DESC", conn
        )
        conn.close()
        return df.to_csv(index=False)
    
    def export_results_csv(self):
        conn = self._get_conn()
        df = pd.read_sql_query('''
            SELECT r.id, r.task_type, r.result_data, r.confidence,
                   r.metadata, r.created_at,
                   a.title as article_title, a.category, a.language
            FROM results r
            JOIN articles a ON r.article_id = a.id
            ORDER BY r.created_at DESC
        ''', conn)
        conn.close()
        return df.to_csv(index=False)
    
    def import_from_csv(self, csv_data):
        try:
            df = pd.read_csv(io.StringIO(csv_data))
            added = 0
            for _, row in df.iterrows():
                title = str(row.get('title', '')).strip()
                content = str(row.get('content', '')).strip()
                category = str(row.get('category', 'Uncategorized')).strip()
                language = str(row.get('language', 'English')).strip()
                tags = str(row.get('tags', '')).strip()
                translated = str(row.get('translated_content', '')).strip()
                if title and content and len(title) > 2 and len(content) > 10:
                    success, _, _ = self.add_article(
                        title, content, category, language, 
                        tags, translated
                    )
                    if success:
                        added += 1
            return added
        except:
            return -1



db = NewsDatabase()



    



db = NewsDatabase()

st.set_page_config(
    page_title="News AI System - NLP Multi-Task",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)



def check_file_exists(filepath):
    
    return os.path.isfile(filepath)

def check_directory_exists(dirpath):
   
    return os.path.isdir(dirpath)

def list_files_in_directory(dirpath):

    if os.path.isdir(dirpath):
        return os.listdir(dirpath)
    return []

def extract_keywords(text, top_n=10):
    
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'who', 'what', 'which', 'when', 'where', 'why', 'how', 'all'
    ])
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_words)
    
    return [word for word, freq in word_freq.most_common(top_n)]

def get_text_stats(text):
   
    words = len(text.split())
    chars = len(text)
    chars_no_spaces = len(text.replace(" ", ""))
    return {
        "words": words,
        "characters": chars,
        "characters_no_spaces": chars_no_spaces
    }

def validate_title(title):
    
    word_count = len(title.split())
    if word_count > 50:
        return False, f"Title is too long ({word_count} words). Maximum 50 words allowed."
    return True, f"Title length is valid ({word_count}/50 words)"


def analyze_sentiment(text):
   
    
    
    positive_words = set([
        'good', 'great', 'excellent', 'amazing', 'wonderful',
        'fantastic', 'positive', 'success', 'successful', 'win',
        'winning', 'victory', 'triumph', 'achievement', 'progress',
        'improve', 'improvement', 'better', 'best', 'growth',
        'grow', 'gain', 'gains', 'profit', 'beneficial', 'benefit',
        'hope', 'hopeful', 'promising', 'breakthrough', 'innovation',
        'innovative', 'exciting', 'celebration', 'celebrate',
        'happy', 'joy', 'love', 'proud', 'remarkable', 'outstanding',
        'brilliant', 'superb', 'strong', 'strength', 'recover',
        'recovery', 'rise', 'rising', 'boost', 'support', 'helped',
        'advance', 'advanced', 'record', 'historic', 'milestone',
        'award', 'awarded', 'launch', 'launched', 'discover',
        'discovered', 'solution', 'solved', 'peace', 'agreement',
        'cooperation', 'united', 'upgrade', 'upgraded', 'safe',
        'secure', 'protect', 'protected', 'efficient', 'effective'
    ])
    
    
    negative_words = set([
        'bad', 'terrible', 'horrible', 'awful', 'worst', 'poor',
        'negative', 'fail', 'failure', 'failed', 'loss', 'lose',
        'losing', 'defeat', 'crisis', 'crash', 'crashed', 'decline',
        'declining', 'drop', 'dropped', 'fall', 'falling', 'risk',
        'danger', 'dangerous', 'threat', 'threatened', 'attack',
        'attacked', 'kill', 'killed', 'death', 'dead', 'die',
        'disaster', 'catastrophe', 'destruction', 'destroy',
        'destroyed', 'damage', 'damaged', 'war', 'conflict',
        'violence', 'violent', 'crime', 'criminal', 'arrest',
        'arrested', 'prison', 'scandal', 'corrupt', 'corruption',
        'fraud', 'illegal', 'ban', 'banned', 'restrict', 'penalty',
        'punishment', 'suffer', 'suffering', 'pain', 'fear',
        'worried', 'concern', 'concerned', 'problem', 'trouble',
        'struggling', 'struggle', 'shortage', 'recession', 'debt',
        'bankrupt', 'bankruptcy', 'collapse', 'collapsed', 'shutdown',
        'layoff', 'layoffs', 'fired', 'unemployment', 'protest',
        'controversy', 'controversial', 'accused', 'allegation',
        'victim', 'tragic', 'tragedy', 'emergency', 'warning'
    ])
    
    
    emotion_words = {
        '😡 Anger': set([
            'angry', 'furious', 'outrage', 'outraged', 'rage',
            'fury', 'hostile', 'aggression', 'violent', 'attack',
            'condemn', 'condemned', 'protest', 'clash'
        ]),
        '😨 Fear': set([
            'fear', 'afraid', 'scared', 'terror', 'terrorism',
            'panic', 'alarm', 'threat', 'danger', 'warning',
            'emergency', 'crisis', 'risk', 'worried', 'anxiety'
        ]),
        '😢 Sadness': set([
            'sad', 'tragic', 'tragedy', 'death', 'died', 'killed',
            'mourning', 'grief', 'loss', 'suffer', 'victim',
            'devastating', 'heartbreaking', 'unfortunate'
        ]),
        '😊 Joy': set([
            'happy', 'joy', 'celebration', 'celebrate', 'victory',
            'win', 'success', 'achievement', 'triumph', 'delight',
            'excited', 'exciting', 'wonderful', 'amazing'
        ]),
        '😲 Surprise': set([
            'surprise', 'surprised', 'shocking', 'shocked',
            'unexpected', 'unprecedented', 'stunning', 'remarkable',
            'breakthrough', 'revelation', 'suddenly', 'dramatic'
        ]),
        '🤢 Disgust': set([
            'disgusting', 'corrupt', 'corruption', 'scandal',
            'fraud', 'abuse', 'exploitation', 'shameful',
            'unacceptable', 'deplorable', 'horrible'
        ])
    }
    
    
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    total_words = len(words) if words else 1
    
    
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    
    
    if pos_count + neg_count == 0:
        score = 0.0
    else:
        score = (pos_count - neg_count) / (pos_count + neg_count)
    
    
    intensity = ((pos_count + neg_count) / total_words) * 100
    
   
    if score > 0.2:
        label = "Positive"
        emoji = "😊"
        color = "#27ae60"
    elif score < -0.2:
        label = "Negative"
        emoji = "😟"
        color = "#e74c3c"
    else:
        label = "Neutral"
        emoji = "😐"
        color = "#f39c12"
    
    
    detected_emotions = {}
    for emotion, emotion_set in emotion_words.items():
        emotion_count = sum(1 for w in words if w in emotion_set)
        if emotion_count > 0:
            detected_emotions[emotion] = emotion_count
    
    
    detected_emotions = dict(
        sorted(detected_emotions.items(), 
               key=lambda x: x[1], reverse=True)
    )
    
    
    pos_found = [w for w in words if w in positive_words]
    neg_found = [w for w in words if w in negative_words]
    
    return {
        'label': label,
        'emoji': emoji,
        'color': color,
        'score': score,
        'intensity': intensity,
        'positive_count': pos_count,
        'negative_count': neg_count,
        'positive_words': list(set(pos_found))[:10],
        'negative_words': list(set(neg_found))[:10],
        'emotions': detected_emotions,
        'total_words': total_words
    }

def extract_entities(text):
   
    
    entities = {
        '👤 People': [],
        '🏢 Organizations': [],
        '📍 Locations': [],
        '📅 Dates': [],
        '💰 Money': [],
        '📊 Percentages': [],
        '🔢 Numbers': []
    }
    
    
    title_pattern = re.findall(
        r'(?:Mr|Mrs|Ms|Dr|Prof|President|Minister|CEO|'
        r'Chairman|Director|Senator|Governor|Mayor|'
        r'Captain|Coach|General|Admiral|Justice|Judge)'
        r'\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})',
        text
    )
    entities['👤 People'].extend(title_pattern)
    
   
    org_patterns = re.findall(
        r'(?:the\s+)?([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*'
        r'(?:\s+(?:Corporation|Corp|Inc|Ltd|Company|Co|Group|'
        r'Association|Organization|Organisation|Institute|'
        r'University|Commission|Committee|Council|Agency|'
        r'Department|Ministry|Bureau|Foundation|Bank|Fund|'
        r'Authority|Board|Court|League|Federation|Union|'
        r'Network|Airlines|Airways|Motors|Technologies|'
        r'Industries|Solutions|Systems|Services)))',
        text
    )
    entities['🏢 Organizations'].extend(org_patterns)
    
    
    acronyms = re.findall(r'\b([A-Z]{2,6})\b', text)
   
    skip_acronyms = {
        'THE', 'AND', 'FOR', 'BUT', 'NOT', 'ARE', 'WAS',
        'HAS', 'HAD', 'HIS', 'HER', 'HIM', 'WHO', 'HOW'
    }
    org_acronyms = [
        a for a in acronyms if a not in skip_acronyms
    ]
    entities['🏢 Organizations'].extend(org_acronyms)
    
    
    known_locations = [
        'United States', 'United Kingdom', 'New York',
        'Washington', 'London', 'Beijing', 'Moscow',
        'Tokyo', 'Paris', 'Berlin', 'Delhi', 'Mumbai',
        'India', 'China', 'Russia', 'Japan', 'Germany',
        'France', 'Brazil', 'Canada', 'Australia',
        'California', 'Texas', 'Florida', 'Europe',
        'Asia', 'Africa', 'Middle East', 'South America',
        'North America', 'Pacific', 'Atlantic', 'Arctic',
        'Mediterranean', 'Silicon Valley', 'Wall Street',
        'Hollywood', 'Pentagon', 'Capitol Hill',
        'White House', 'Downing Street', 'Kremlin'
    ]
    
    for loc in known_locations:
        if loc.lower() in text.lower():
            entities['📍 Locations'].append(loc)
    
    
    date_patterns = re.findall(
        r'(?:January|February|March|April|May|June|July|'
        r'August|September|October|November|December)'
        r'\s+\d{1,2}(?:,\s*\d{4})?',
        text
    )
    entities['📅 Dates'].extend(date_patterns)
    
    
    years = re.findall(r'\b((?:19|20)\d{2})\b', text)
    entities['📅 Dates'].extend(years)
    
    
    money = re.findall(
        r'[\$£€₹]\s*\d+(?:[\.,]\d+)*\s*'
        r'(?:million|billion|trillion|thousand|crore|lakh)?',
        text, re.IGNORECASE
    )
    entities['💰 Money'].extend(money)
    
    money2 = re.findall(
        r'\d+(?:[\.,]\d+)*\s*'
        r'(?:dollars|euros|pounds|rupees|yen|yuan)',
        text, re.IGNORECASE
    )
    entities['💰 Money'].extend(money2)
    
    
    percentages = re.findall(
        r'\d+(?:\.\d+)?\s*(?:%|percent|per cent)',
        text, re.IGNORECASE
    )
    entities['📊 Percentages'].extend(percentages)
    
    
    big_numbers = re.findall(
        r'\d+(?:[\.,]\d+)*\s*'
        r'(?:million|billion|trillion|thousand|crore|lakh)',
        text, re.IGNORECASE
    )
    entities['🔢 Numbers'].extend(big_numbers)
    
    
    clean_entities = {}
    for category, items in entities.items():
        unique_items = list(dict.fromkeys(items))  # Preserve order
        if unique_items:
            clean_entities[category] = unique_items
    
    return clean_entities


def calculate_readability(text):
   
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = text.split()
    
    
    def count_syllables(word):
        word = word.lower().strip(".,!?;:'\"")
        if len(word) <= 2:
            return 1
        
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        
        if word.endswith('e') and count > 1:
            count -= 1
        
        return max(1, count)
    
    total_sentences = max(1, len(sentences))
    total_words = max(1, len(words))
    total_syllables = sum(count_syllables(w) for w in words)
    
    
    flesch = (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )
    
   
    flesch = max(0, min(100, flesch))
    
    
    if flesch >= 90:
        level = "Very Easy"
        grade = "5th Grade"
        color = "#27ae60"
        emoji = "📗"
    elif flesch >= 80:
        level = "Easy"
        grade = "6th Grade"
        color = "#2ecc71"
        emoji = "📗"
    elif flesch >= 70:
        level = "Fairly Easy"
        grade = "7th Grade"
        color = "#3498db"
        emoji = "📘"
    elif flesch >= 60:
        level = "Standard"
        grade = "8th-9th Grade"
        color = "#3498db"
        emoji = "📘"
    elif flesch >= 50:
        level = "Fairly Difficult"
        grade = "10th-12th Grade"
        color = "#f39c12"
        emoji = "📙"
    elif flesch >= 30:
        level = "Difficult"
        grade = "College"
        color = "#e67e22"
        emoji = "📕"
    else:
        level = "Very Difficult"
        grade = "Professional"
        color = "#e74c3c"
        emoji = "📕"
    
    
    avg_word_length = sum(len(w) for w in words) / total_words
    
    
    avg_sentence_length = total_words / total_sentences
    
    return {
        'flesch_score': flesch,
        'level': level,
        'grade': grade,
        'color': color,
        'emoji': emoji,
        'total_sentences': total_sentences,
        'total_words': total_words,
        'total_syllables': total_syllables,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length
    }


def get_top_influencing_words(text, vectorizer, model, predicted_class, top_n=15):
  
    
    try:
       
        text_vectorized = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        
        non_zero = text_vectorized.nonzero()
        
        word_scores = []
        
        for idx in non_zero[1]:
            word = feature_names[idx]
            tfidf_value = text_vectorized[0, idx]
            
            
            if hasattr(model, 'feature_log_prob_'):
                
                class_idx = predicted_class - 1
                log_prob = model.feature_log_prob_[class_idx, idx]
                influence = tfidf_value * log_prob
            else:
                influence = tfidf_value
            
            word_scores.append({
                'word': word,
                'tfidf': tfidf_value,
                'influence': abs(influence),
                'raw_influence': influence
            })
        
        
        word_scores.sort(key=lambda x: x['influence'], reverse=True)
        
        return word_scores[:top_n]
        
    except Exception as e:
        return []


def suggest_subcategory(category, keywords, text):
    
    
    subcategories = {
        'World': {
            'Politics & Diplomacy': [
                'election', 'vote', 'president', 'minister',
                'government', 'policy', 'diplomat', 'treaty',
                'summit', 'sanctions', 'parliament', 'congress'
            ],
            'Conflict & Security': [
                'war', 'military', 'army', 'attack', 'bomb',
                'terrorism', 'security', 'defense', 'weapons',
                'troops', 'invasion', 'conflict', 'missile'
            ],
            'Humanitarian': [
                'refugee', 'aid', 'humanitarian', 'crisis',
                'disaster', 'earthquake', 'flood', 'relief',
                'poverty', 'hunger', 'epidemic', 'vaccine'
            ],
            'Environment & Climate': [
                'climate', 'environment', 'carbon', 'emission',
                'warming', 'pollution', 'renewable', 'energy',
                'forest', 'ocean', 'species', 'conservation'
            ]
        },
        'Sports': {
            'Football/Soccer': [
                'football', 'soccer', 'goal', 'premier league',
                'fifa', 'striker', 'midfielder', 'champions league',
                'world cup', 'penalty', 'referee'
            ],
            'Cricket': [
                'cricket', 'wicket', 'batsman', 'bowler', 'innings',
                'test match', 'odi', 'ipl', 'century', 'ashes'
            ],
            'Basketball': [
                'basketball', 'nba', 'dunk', 'three-pointer',
                'court', 'rebound', 'playoff', 'finals'
            ],
            'Olympics & Athletics': [
                'olympic', 'olympics', 'medal', 'gold medal',
                'athlete', 'record', 'marathon', 'sprint',
                'swimming', 'gymnastics', 'track'
            ],
            'Tennis': [
                'tennis', 'grand slam', 'wimbledon', 'serve',
                'match point', 'ace', 'set', 'roland garros'
            ]
        },
        'Business': {
            'Stock Market & Finance': [
                'stock', 'market', 'shares', 'trading', 'investor',
                'wall street', 'nasdaq', 'dow', 'index', 'bull',
                'bear', 'rally', 'ipo', 'dividend'
            ],
            'Technology Business': [
                'startup', 'silicon valley', 'tech', 'app',
                'software', 'digital', 'platform', 'billion',
                'valuation', 'funding', 'venture'
            ],
            'Economy & Policy': [
                'economy', 'gdp', 'inflation', 'interest rate',
                'federal reserve', 'recession', 'unemployment',
                'fiscal', 'monetary', 'budget', 'tax'
            ],
            'Corporate News': [
                'merger', 'acquisition', 'ceo', 'revenue',
                'earnings', 'profit', 'quarterly', 'annual',
                'restructuring', 'layoff', 'expansion'
            ]
        },
        'Sci-Fi': {
            'Space & Astronomy': [
                'space', 'nasa', 'mars', 'moon', 'satellite',
                'rocket', 'orbit', 'astronaut', 'telescope',
                'galaxy', 'planet', 'asteroid', 'spacecraft'
            ],
            'Artificial Intelligence': [
                'ai', 'artificial intelligence', 'machine learning',
                'robot', 'algorithm', 'neural', 'automation',
                'chatbot', 'deep learning', 'autonomous'
            ],
            'Medical Science': [
                'gene', 'dna', 'vaccine', 'drug', 'treatment',
                'clinical', 'therapy', 'cancer', 'cell',
                'protein', 'medical', 'health', 'disease'
            ],
            'Physics & Engineering': [
                'quantum', 'particle', 'energy', 'nuclear',
                'fusion', 'laser', 'physics', 'engineering',
                'material', 'nanotechnology', 'superconductor'
            ]
        }
    }
    
    text_lower = text.lower()
    
    if category not in subcategories:
        return []
    
    scores = {}
    for subcat, subcat_keywords in subcategories[category].items():
        match_count = sum(
            1 for kw in subcat_keywords
            if kw in text_lower
        )
        if match_count > 0:
            scores[subcat] = match_count
    
   
    sorted_subcats = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )
    
    return sorted_subcats[:3]  



def create_classification_pdf(title, description, category, confidence, keywords):
   
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#3498db'),
        spaceAfter=30,
    )
    
    story.append(Paragraph("News Classification Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>News Title:</b>", styles['Heading2']))
    story.append(Paragraph(title, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("<b>Description:</b>", styles['Heading2']))
    story.append(Paragraph(description, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Classification Result:</b>", styles['Heading2']))
    story.append(Paragraph(f"<b>Category:</b> {category}", styles['Normal']))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Extracted Keywords:</b>", styles['Heading2']))
    story.append(Paragraph(", ".join(keywords), styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_summary_pdf(original_text, summary, reduction):
   
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#27ae60'),
        spaceAfter=30,
    )
    
    story.append(Paragraph("News Summarization Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Original Article:</b>", styles['Heading2']))
    story.append(Paragraph(original_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Generated Summary:</b>", styles['Heading2']))
    story.append(Paragraph(summary, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    orig_stats = get_text_stats(original_text)
    summ_stats = get_text_stats(summary)
    
    story.append(Paragraph("<b>Statistics:</b>", styles['Heading2']))
    story.append(Paragraph(f"Original Words: {orig_stats['words']}", styles['Normal']))
    story.append(Paragraph(f"Summary Words: {summ_stats['words']}", styles['Normal']))
    story.append(Paragraph(f"Reduction: {reduction:.1f}%", styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_translation_pdf(hindi_text, english_text):
   
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#9b59b6'),
        spaceAfter=30,
    )
    
    story.append(Paragraph("News Translation Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Hindi (Original):</b>", styles['Heading2']))
    story.append(Paragraph(hindi_text, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>English (Translated):</b>", styles['Heading2']))
    story.append(Paragraph(english_text, styles['Normal']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer





@st.cache_resource
def load_classifier():
    
    try:
        if not check_directory_exists('classifier'):
            st.sidebar.error("❌ Classifier folder not found")
            return None, None, False
        
        model_path = 'classifier/nb_model.pkl'
        vectorizer_path = 'classifier/tfidf_vectorizer.pkl'
        
        if not check_file_exists(model_path):
            st.sidebar.error(f"❌ {model_path} not found")
            return None, None, False
            
        if not check_file_exists(vectorizer_path):
            st.sidebar.error(f"❌ {vectorizer_path} not found")
            return None, None, False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        st.sidebar.success("✅ Classifier loaded")
        return model, vectorizer, True
        
    except Exception as e:
        st.sidebar.error(f"❌ Classifier error: {str(e)[:100]}")
        return None, None, False


@st.cache_resource
def load_summarizer():
    
    if not TRANSFORMERS_AVAILABLE:
        st.sidebar.error("❌ Transformers not available")
        return None, False
    
    try:
        
        if not check_directory_exists('summarizer'):
            st.sidebar.error("❌ Summarizer folder not found")
            return None, False
        
        files = list_files_in_directory('summarizer')
        
        
        required_files = [
            'config.json',
            'pytorch_model.bin',
            'tokenizer_config.json',
            'vocab.json',
            'merges.txt'
        ]
        
        missing = [f for f in required_files if f not in files]
        
        if missing:
            st.sidebar.error(f"❌ Missing files: {', '.join(missing)}")
            st.sidebar.info("💡 Re-run the backend code in Colab")
            return None, False
        
        
        try:
            from transformers import BartForConditionalGeneration, BartTokenizer
            
            
            model = BartForConditionalGeneration.from_pretrained(
                "./summarizer",
                local_files_only=True
            )
            
            
            tokenizer = BartTokenizer.from_pretrained(
                "./summarizer",
                local_files_only=True
            )
            
            
            model.eval()
            
            st.sidebar.success("✅ Summarizer loaded")
            
            return (model, tokenizer), True
            
        except Exception as e:
            st.sidebar.error(f"❌ Loading failed: {str(e)[:100]}")
            
            with st.sidebar.expander("🔍 Error Details"):
                st.code(str(e))
                st.write("**Files found:**")
                for f in sorted(files):
                    st.write(f"  • {f}")
            
            return None, False
        
    except Exception as e:
        st.sidebar.error(f"❌ Error: {str(e)[:100]}")
        return None, False



@st.cache_resource
def load_translator():
  
    if not TRANSFORMERS_AVAILABLE:
        st.sidebar.error("❌ Transformers not available")
        return None, None, False
    
    try:
        
        try:
            import sentencepiece
        except ImportError:
            st.sidebar.error("❌ sentencepiece not installed")
            st.sidebar.info("Run: pip install sentencepiece")
            return None, None, False
        
        if not check_directory_exists('translator'):
            st.sidebar.error("❌ Translator folder not found")
            return None, None, False
        
        files = list_files_in_directory('translator')
        
        
        if 'sentencepiece.bpe.model' not in files:
            st.sidebar.error("❌ sentencepiece.bpe.model missing")
            return None, None, False
        
        if 'config.json' not in files:
            st.sidebar.error("❌ config.json missing")
            return None, None, False
        
        
        has_pytorch = 'pytorch_model.bin' in files
        has_safetensors = 'model.safetensors' in files
        
        if not (has_pytorch or has_safetensors):
            st.sidebar.error("❌ No model file found")
            return None, None, False
        
        
        try:
            
            from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
            
            try:
                tokenizer = NllbTokenizer.from_pretrained(
                    "./translator",
                    local_files_only=True
                )
                st.sidebar.info("✓ Using NllbTokenizer")
                
            except Exception as e1:
                
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        "./translator",
                        local_files_only=True,
                        use_fast=False
                    )
                    st.sidebar.info("✓ Using AutoTokenizer")
                    
                except Exception as e2:
                    st.sidebar.error(f"❌ Tokenizer load failed: {str(e2)[:100]}")
                    return None, None, False
            
           
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "./translator",
                local_files_only=True
            )
            
        except Exception as e:
            st.sidebar.error(f"❌ Model load failed: {str(e)[:150]}")
            return None, None, False
        
        
        try:
            test_text = "नमस्ते"
            
            
            tokenizer.src_lang = "hin_Deva"
            
            
            inputs = tokenizer(test_text, return_tensors="pt")
            
           
            eng_token_id = None
            
           
            try:
                eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
            except:
                pass
            
            
            if eng_token_id is None:
                try:
                    if hasattr(tokenizer, 'lang_code_to_id'):
                        eng_token_id = tokenizer.lang_code_to_id.get("eng_Latn", 256047)
                    else:
                        eng_token_id = 256047  
                except:
                    eng_token_id = 256047
            
        
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=eng_token_id,
                    max_length=50
                )
            
            test_result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            st.sidebar.success(f"✅ Translator loaded")
            return tokenizer, model, True
            
        except Exception as e:
            st.sidebar.error(f"❌ Test failed: {str(e)[:150]}")
            return None, None, False
        
    except Exception as e:
        st.sidebar.error(f"❌ Translator error: {str(e)[:100]}")
        return None, None, False
    






st.sidebar.markdown("### 🔄 Loading Models...")


classifier_model, tfidf_vectorizer, classifier_loaded = load_classifier()


summarizer_model, summarizer_loaded = load_summarizer()

translator_tokenizer, translator_model, translator_loaded = load_translator()



st.sidebar.markdown("---")


def classify_news(title, description):
    
    if not classifier_loaded:
        return None, None
    
    try:
        text = title + " " + description
        text_vectorized = tfidf_vectorizer.transform([text])
        predicted_class = classifier_model.predict(text_vectorized)[0]
        probabilities = classifier_model.predict_proba(text_vectorized)[0]
        confidence_scores = {i+1: prob for i, prob in enumerate(probabilities)}
        
        return predicted_class, confidence_scores
    except Exception as e:
        st.error(f"Classification error: {e}")
        return None, None


def summarize_text(text, summary_ratio=0.30):
    
    if not summarizer_loaded:
        return None
    
    try:
        model, tokenizer = summarizer_model
        words = len(text.split())
        
        test_tokens = tokenizer(
            text, return_tensors="pt",
            truncation=False, padding=False
        )
        actual_token_count = test_tokens["input_ids"].shape[1]
        
        st.info(
            f"📊 Article: {words} words | "
            f"{actual_token_count} tokens | "
            f"Target: ~{int(words * summary_ratio)} words"
        )
        
        if actual_token_count <= 400:
            st.info("📝 Short article — direct summarization...")
            return _summarize_single_pass(
                text, model, tokenizer, words, summary_ratio
            )
        else:
            st.info(
                f"📝 Long article ({actual_token_count} tokens). "
                f"Chunking for complete coverage..."
            )
            return _summarize_long_article(
                text, model, tokenizer, summary_ratio
            )
    
    except Exception as e:
        st.error(f"❌ Summarization error: {str(e)[:300]}")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())
        return None


def _summarize_single_pass(text, model, tokenizer, word_count,
                           summary_ratio=0.30):
   
    try:
        max_out = max(50, int(word_count * (summary_ratio + 0.15)))
        min_out = max(20, int(word_count * summary_ratio * 0.5))
        
        if min_out >= max_out:
            min_out = int(max_out * 0.4)
        
        
        max_out = min(max_out, 512)
        min_out = min(min_out, max_out - 10)
        
        inputs = tokenizer(
            text, return_tensors="pt",
            max_length=1024, truncation=True, padding=False
        )
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_out,
                min_length=min_out,
                num_beams=6,
                length_penalty=0.8,
                early_stopping=False,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                do_sample=False
            )
        
        return tokenizer.decode(
            summary_ids[0], skip_special_tokens=True
        )
    
    except Exception as e:
        st.error(f"❌ Single-pass error: {str(e)[:200]}")
        return None


def _summarize_long_article(text, model, tokenizer,
                            summary_ratio=0.30):
   
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return None
        
        total_words = len(text.split())
        
        
        chunks = []
        current_chunk = []
        current_words = 0
        max_chunk_words = 200
        overlap = 2
        
        for sentence in sentences:
            sw = len(sentence.split())
            if current_words + sw > max_chunk_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                if overlap > 0 and len(current_chunk) > overlap:
                    ov = current_chunk[-overlap:]
                    current_chunk = ov.copy()
                    current_words = sum(len(s.split()) for s in current_chunk)
                else:
                    current_chunk = []
                    current_words = 0
                current_chunk.append(sentence)
                current_words += sw
            else:
                current_chunk.append(sentence)
                current_words += sw
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        if len(chunks) == 1:
            return _summarize_single_pass(
                chunks[0], model, tokenizer,
                len(chunks[0].split()), summary_ratio
            )
        
        st.info(f"📊 Split into **{len(chunks)} chunks**")
        
        # Summarize each chunk
        chunk_summaries = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            cw = len(chunk.split())
            status_text.text(f"Chunk {i+1}/{len(chunks)} ({cw} words)...")
            
            chunk_ratio = summary_ratio + 0.10
            c_max = min(300, max(60, int(cw * (chunk_ratio + 0.10))))
            c_min = min(150, max(30, int(cw * chunk_ratio * 0.5)))
            if c_min >= c_max:
                c_min = int(c_max * 0.4)
            
            inputs = tokenizer(
                chunk, return_tensors="pt",
                max_length=1024, truncation=True, padding=False
            )
            
            with torch.no_grad():
                ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=c_max, min_length=c_min,
                    num_beams=5, length_penalty=0.6,
                    early_stopping=False,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.2, do_sample=False
                )
            
            chunk_summaries.append(
                tokenizer.decode(ids[0], skip_special_tokens=True)
            )
            progress_bar.progress((i + 1) / len(chunks))
        
        progress_bar.empty()
        status_text.empty()
        
        
        with st.expander(
            f"📋 Chunk Summaries ({len(chunk_summaries)})",
            expanded=False
        ):
            for i, cs in enumerate(chunk_summaries):
                st.markdown(
                    f"**Chunk {i+1}** ({len(cs.split())}w): "
                )
                st.caption(cs)
        
        
        combined = ' '.join(chunk_summaries)
        combined_words = len(combined.split())
        
        st.info(f"🔄 Creating final summary from {combined_words} words...")
        
        combined_tokens = tokenizer(
            combined, return_tensors="pt",
            truncation=False, padding=False
        )["input_ids"].shape[1]
        
        if combined_tokens <= 900:
            f_max = min(500, max(100, int(total_words * summary_ratio)))
            f_min = min(250, max(50, int(total_words * summary_ratio * 0.5)))
            if f_min >= f_max:
                f_min = int(f_max * 0.4)
            
            inputs = tokenizer(
                combined, return_tensors="pt",
                max_length=1024, truncation=True, padding=False
            )
            
            with torch.no_grad():
                final_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_length=f_max, min_length=f_min,
                    num_beams=6, length_penalty=0.8,
                    early_stopping=False,
                    no_repeat_ngram_size=3,
                    repetition_penalty=1.3, do_sample=False
                )
            
            final = tokenizer.decode(
                final_ids[0], skip_special_tokens=True
            )
        else:
            final = _summarize_long_article(
                combined, model, tokenizer, summary_ratio
            )
        
        if final:
            sw = len(final.split())
            st.success(
                f"✅ {len(chunks)} chunks → {sw} words "
                f"(from {total_words} words)"
            )
        
        return final
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)[:300]}")
        return None

def translate_text(source_text, source_lang="hin_Deva"):
   
    if not translator_loaded:
        return None
    
    try:
        
        words = len(source_text.split())
        
        
        if words > 400:
            st.info(f"📝 Long text detected ({words} words). Processing in chunks for complete translation...")
            return _translate_long_text(source_text, source_lang)
        
        
        return _translate_chunk(source_text, source_lang)
        
    except Exception as e:
        st.error(f"❌ Translation error: {str(e)[:200]}")
        return None


def _translate_chunk(text_chunk, source_lang):
    
    try:
        
        translator_tokenizer.src_lang = source_lang
        
        
        inputs = translator_tokenizer(
            text_chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512  
        )
        
        
        token_count = inputs["input_ids"].shape[1]
        if token_count >= 512:
            st.warning(f"⚠️ Chunk at token limit. Text may be truncated.")
        
       
        try:
            eng_token_id = translator_tokenizer.convert_tokens_to_ids("eng_Latn")
        except:
            eng_token_id = 256047
        
        
        with torch.no_grad():
            translated_tokens = translator_model.generate(
                **inputs,
                forced_bos_token_id=eng_token_id,
                max_length=512,
                num_beams=3,
                length_penalty=1.0,
                early_stopping=True
            )
        
       
        translation = translator_tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]
        
        return translation
        
    except Exception as e:
        st.error(f"Chunk translation error: {e}")
        return None






def _translate_long_text(text, source_lang):

    try:
        
        if source_lang == "hin_Deva":
            
            sentences = re.split(r'[।\n]+', text)
        elif source_lang in ["deu_Latn", "spa_Latn", "fra_Latn"]:
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
        else:
            
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        max_chunk_words = 300
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            
            if current_word_count + sentence_words > max_chunk_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_word_count = sentence_words
            else:
                current_chunk.append(sentence)
                current_word_count += sentence_words
        
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        st.info(f"📊 Split into {len(chunks)} chunks for processing")
        
       
        translated_chunks = []
        
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            status_text.text(f"Translating chunk {i+1}/{len(chunks)}...")
            
            
            translated_chunk = _translate_chunk(chunk, source_lang)
            
            if translated_chunk:
                translated_chunks.append(translated_chunk)
            
            
            progress_bar.progress((i + 1) / len(chunks))
        
        
        progress_bar.empty()
        status_text.empty()
        
        
        full_translation = ' '.join(translated_chunks)
        
        st.success(f"✅ Translated {len(chunks)} chunks successfully")
        
        return full_translation
        
    except Exception as e:
        st.error(f"❌ Long text translation error: {str(e)[:200]}")
        import traceback
        st.code(traceback.format_exc())
        return None



def extract_bullet_points(text, num_points=5):
    
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 4]
    
    if not sentences:
        return []
    
    if len(sentences) <= num_points:
        return sentences
    
   
    all_words = re.sub(r'[^\w\s]', '', text.lower()).split()
    stop_words = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
        'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was',
        'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may',
        'might', 'must', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'who', 'what',
        'which', 'when', 'where', 'why', 'how', 'all', 'not', 'no',
        'its', 'his', 'her', 'their', 'our', 'your', 'my', 'also',
        'than', 'then', 'so', 'if', 'about', 'up', 'out', 'just',
        'into', 'some', 'more', 'other', 'very', 'after', 'before',
        'said', 'says', 'say', 'according', 'new', 'one', 'two'
    ])
    
    word_freq = Counter(
        w for w in all_words 
        if w not in stop_words and len(w) > 2
    )
    
    max_freq = max(word_freq.values()) if word_freq else 1
    
    scored_sentences = []
    total_sentences = len(sentences)
    
    for idx, sentence in enumerate(sentences):
        score = 0.0
        sent_words = re.sub(r'[^\w\s]', '', sentence.lower()).split()
        sent_len = len(sent_words)
        
        
        freq_score = sum(
            word_freq.get(w, 0) / max_freq
            for w in sent_words
            if w not in stop_words and len(w) > 2
        )
        if sent_len > 0:
            freq_score = freq_score / sent_len
        score += freq_score * 3.0
        
        
        if idx == 0:
            score += 2.0
        elif idx == 1:
            score += 1.5
        elif idx == total_sentences - 1:
            score += 1.0
        elif idx < total_sentences * 0.3:
            score += 0.8
        
       
        numbers = re.findall(r'\d+', sentence)
        if numbers:
            score += 1.0 + (0.2 * min(len(numbers), 3))
        
        
        if '"' in sentence or "'" in sentence:
            score += 0.8
        
     
        if 10 <= sent_len <= 30:
            score += 0.5
        elif sent_len < 6:
            score -= 0.5
        elif sent_len > 40:
            score -= 0.3
        
       
        caps = re.findall(r'\b[A-Z][a-z]+\b', sentence)
        if len(caps) > 1:
            score += 0.5
        
        # 7. Contains key indicator words
        indicator_words = [
            'important', 'significant', 'major', 'key',
            'announced', 'revealed', 'discovered', 'reported',
            'according', 'million', 'billion', 'percent',
            'first', 'new', 'record', 'unprecedented'
        ]
        for word in indicator_words:
            if word in sentence.lower():
                score += 0.3
        
        scored_sentences.append((idx, sentence, score))
    
  
    scored_sentences.sort(key=lambda x: x[2], reverse=True)
    top_sentences = scored_sentences[:num_points]
    
    
    top_sentences.sort(key=lambda x: x[0])
    
    return [s[1] for s in top_sentences]
            

def translate_bidirectional(source_text, source_lang_code,
                            target_lang_code):
   
    if not translator_loaded:
        return None
    
    try:
        words = len(source_text.split())
        
        if words > 400:
            st.info(
                f"📝 Long text ({words} words). "
                f"Processing in chunks..."
            )
            return _translate_long_bidi(
                source_text, source_lang_code, target_lang_code
            )
        
        return _translate_chunk_bidi(
            source_text, source_lang_code, target_lang_code
        )
    
    except Exception as e:
        st.error(f"❌ Translation error: {str(e)[:200]}")
        return None


def _translate_chunk_bidi(text_chunk, source_lang_code,
                          target_lang_code):
   
    try:
        translator_tokenizer.src_lang = source_lang_code
        
        inputs = translator_tokenizer(
            text_chunk, return_tensors="pt",
            padding=True, truncation=True, max_length=512
        )
        
        
        try:
            target_token_id = translator_tokenizer.convert_tokens_to_ids(
                target_lang_code
            )
        except:
           
            fallback_ids = {
                "eng_Latn": 256047,
                "hin_Deva": 256037,
                "deu_Latn": 256042,
                "spa_Latn": 256045,
                "fra_Latn": 256044
            }
            target_token_id = fallback_ids.get(target_lang_code, 256047)
        
        with torch.no_grad():
            translated_tokens = translator_model.generate(
                **inputs,
                forced_bos_token_id=target_token_id,
                max_length=512,
                num_beams=3,
                length_penalty=1.0,
                early_stopping=True
            )
        
        translation = translator_tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )[0]
        
        return translation
    
    except Exception as e:
        st.error(f"Chunk translation error: {e}")
        return None


def _translate_long_bidi(text, source_lang_code, target_lang_code):
    
    try:
     
        if source_lang_code == "hin_Deva":
            sentences = re.split(r'[।\n]+', text)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sentences = [s.strip() for s in sentences if s.strip()]
        
        
        chunks = []
        current_chunk = []
        current_words = 0
        max_words = 300
        
        for sentence in sentences:
            sw = len(sentence.split())
            if current_words + sw > max_words and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_words = sw
            else:
                current_chunk.append(sentence)
                current_words += sw
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        st.info(f"📊 Split into {len(chunks)} chunks")
        
       
        translated_chunks = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, chunk in enumerate(chunks):
            status_text.text(
                f"Translating chunk {i+1}/{len(chunks)}..."
            )
            
            result = _translate_chunk_bidi(
                chunk, source_lang_code, target_lang_code
            )
            
            if result:
                translated_chunks.append(result)
            
            progress_bar.progress((i + 1) / len(chunks))
        
        progress_bar.empty()
        status_text.empty()
        
        full_translation = ' '.join(translated_chunks)
        st.success(f"✅ Translated {len(chunks)} chunks")
        
        return full_translation
    
    except Exception as e:
        st.error(f"❌ Long text error: {str(e)[:200]}")
        return None






def article_selector(task_name, filter_language=None, 
                     english_content_only=False, key_prefix=""):
    
    
    source = st.radio(
        "📥 Input Source:",
        ["✍️ Type / Paste Manually", "📚 Pick from Database"],
        horizontal=True,
        key=f"{key_prefix}_source"
    )
    
    if source == "📚 Pick from Database":
        
       
        
        if english_content_only:
            
            articles = db.get_english_content_articles()
            
            if not articles:
                st.warning(
                    "📭 No articles with English content! "
                    "Add English articles or translate existing ones first."
                )
                return None, None, False, None
        
        elif filter_language and filter_language != "All":
            
            articles = db.get_by_language(filter_language)
            
            if not articles:
                st.warning(
                    f"📭 No {filter_language} articles in database!"
                )
                return None, None, False, None
        
        else:
            articles = db.get_all()
            
            if not articles:
                st.warning(
                    "📭 No articles in database! "
                    "Go to '📚 Article Database' tab to add some."
                )
                return None, None, False, None
        
        
        
        options = {}
        for a in articles:
            has_translation = (
                a.get('translated_content', '') and 
                a.get('translated_content', '').strip()
            )
            
            if english_content_only and a['language'] != 'English' and has_translation:
                
                label = (
                    f"🔄 [Translated] {a['title'][:55]} "
                    f"({a['language']}→EN | "
                    f"{len(a['translated_content'].split())}w)"
                )
            else:
                label = (
                    f"📰 {a['title'][:60]} "
                    f"({a['word_count']}w | {a['language']})"
                )
            
            options[label] = a['id']
        
        selected_label = st.selectbox(
            f"Select article for {task_name}:",
            list(options.keys()),
            key=f"{key_prefix}_select"
        )
        
        article_id = options[selected_label]
        article = db.get_by_id(article_id)
        
        if article:
            has_translation = (
                article.get('translated_content', '') and 
                article.get('translated_content', '').strip()
            )
            
          
            
            if english_content_only and article['language'] != 'English':
                if has_translation:
                   
                    return_content = article['translated_content']
                    content_source = "English Translation"
                else:
                    st.warning(
                        "⚠️ This article hasn't been translated yet. "
                        "Translate it first!"
                    )
                    return None, None, False, None
            else:
                return_content = article['content']
                content_source = "Original"
            
           
            
            with st.expander(
                "👁️ Preview & Past Results", expanded=False
            ):
                st.markdown(f"**Title:** {article['title']}")
                st.markdown(
                    f"**Category:** {article['category']} | "
                    f"**Language:** {article['language']} | "
                    f"**Words:** {article['word_count']}"
                )
                
                
                if has_translation:
                    st.markdown(
                        f"✅ **English translation available** "
                        f"({len(article['translated_content'].split())} words)"
                    )
                    
                    if english_content_only and article['language'] != 'English':
                        st.info(
                            f"📝 Using **English translation** "
                            f"for {task_name}"
                        )
                
                
                past_results = db.get_results_for_article(article_id)
                
                if past_results:
                    st.markdown("---")
                    st.markdown("**📊 Previous Results:**")
                    
                    for result in past_results:
                        task_emoji = {
                            'classification': '📊',
                            'summarization': '📝',
                            'translation': '🌐'
                        }.get(result['task_type'], '📄')
                        
                        st.markdown(
                            f"{task_emoji} **{result['task_type'].title()}** "
                            f"— {result['created_at']}"
                        )
                        
                        preview = result['result_data'][:150]
                        if len(result['result_data']) > 150:
                            preview += "..."
                        
                        st.markdown(
                            f"<div style='color:#95a5a6; font-size:13px; "
                            f"padding:5px 10px; background:rgba(0,0,0,0.2); "
                            f"border-radius:5px;'>{preview}</div>",
                            unsafe_allow_html=True
                        )
                
                st.markdown("---")
                st.markdown(f"**📄 Content ({content_source}):**")
                preview_text = return_content[:600]
                if len(return_content) > 600:
                    preview_text += "..."
                st.write(preview_text)
            
            return article['title'], return_content, True, article_id
        
        return None, None, False, None
    
    
    return None, None, False, None
    







st.sidebar.markdown("---")



if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'summarization_history' not in st.session_state:
    st.session_state.summarization_history = []
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

if 'class_result' not in st.session_state:
    st.session_state.class_result = None
if 'summ_result' not in st.session_state:
    st.session_state.summ_result = None
if 'trans_result' not in st.session_state:
    st.session_state.trans_result = None


CATEGORY_INFO = {
    1: {"name": "World", "emoji": "🌍", "color": "#3498db"},
    2: {"name": "Sports", "emoji": "⚽", "color": "#e74c3c"},
    3: {"name": "Business", "emoji": "💼", "color": "#f39c12"},
    4: {"name": "Sci-Fi", "emoji": "🚀", "color": "#9b59b6"}
}



st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .block-container {
        background: rgba(26, 26, 46, 0.95);
        border-radius: 20px;
        padding: 2rem;
        padding-top: 3rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 1rem;
    }
    
    .main-header {
        font-size: 100px;
        font-weight: bold;
        background: linear-gradient(120deg, #e74c3c, #3498db, #f39c12, #9b59b6);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 15px;
        margin-top: 60px;
        animation: gradient 3s ease infinite;
        padding-top: 20px;
        line-height: 1.2;
    }
    
    @keyframes gradient {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    .sub-header {
        font-size: 24px;
        color: #e0e0e0;
        text-align: center;
        margin-bottom: 35px;
        margin-top: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-weight: 500;
    }
    
    .top-spacer {
        height: 80px;
    }
    
    .stMarkdown, .stText, p, span, label {
        color: #e0e0e0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
        height: 60px;
        font-size: 22px;
        font-weight: bold;
        border-radius: 30px;
        border: none;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.6);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.8);
    }
    
    .stDownloadButton>button {
        background: linear-gradient(135deg, #27ae60 0%, #229954 100%) !important;
        box-shadow: 0 4px 15px rgba(39, 174, 96, 0.6);
    }
    
    .info-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        border: 1px solid rgba(52, 152, 219, 0.3);
    }
    
    .stTextArea textarea, .stTextInput input {
        background: rgba(26, 26, 46, 0.8) !important;
        color: #ffffff !important;
        border: 1px solid rgba(52, 152, 219, 0.5) !important;
        border-radius: 10px;
        font-size: 16px;
    }
    
    .section-header {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0 10px 0;
        font-size: 22px;
        font-weight: bold;
        box-shadow: 0 4px 10px rgba(52, 152, 219, 0.4);
    }
    
    .result-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        border: 2px solid #3498db;
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
    }
    
    .prediction-box {
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        margin: 20px 0;
        animation: slideIn 0.5s ease-out;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    @keyframes slideIn {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    .stats-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #2d2d44 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(52, 152, 219, 0.3);
        margin: 10px 0;
        text-align: center;
    }
    
    .keyword-tag {
        display: inline-block;
        background: linear-gradient(135deg, #f39c12, #d68910);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        margin: 5px;
        font-size: 14px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(26, 26, 46, 0.8);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(52, 152, 219, 0.2);
        color: #e0e0e0;
        border-radius: 10px;
        padding: 15px 25px;
        font-weight: bold;
        font-size: 16px;
        border: 1px solid rgba(52, 152, 219, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white !important;
    }
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        border-radius: 10px;
        color: white !important;
        font-weight: bold;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    section[data-testid="stSidebar"] > div {
        background: rgba(26, 26, 46, 0.8);
        padding-top: 3rem;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        border-radius: 10px;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
    }
    
    [data-testid="stMetricValue"] {
        color: #3498db !important;
        font-size: 24px !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #e0e0e0 !important;
    }
    
    hr {
        border-color: rgba(52, 152, 219, 0.3) !important;
        margin: 2rem 0;
    }
    
    .category-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 18px;
        font-weight: bold;
        margin: 5px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    }
    
    .badge-world { background: linear-gradient(135deg, #3498db, #2980b9); color: white; }
    .badge-sports { background: linear-gradient(135deg, #e74c3c, #c0392b); color: white; }
    .badge-business { background: linear-gradient(135deg, #f39c12, #d68910); color: white; }
    .badge-scifi { background: linear-gradient(135deg, #9b59b6, #8e44ad); color: white; }
    
    .char-count {
        font-size: 14px;
        color: #95a5a6;
        text-align: right;
        margin-top: 5px;
    }
    
    .char-count.error {
        color: #e74c3c;
    }
    
    </style>
""", unsafe_allow_html=True)



st.markdown('<div class="top-spacer"></div>', unsafe_allow_html=True)
st.markdown('<p class="main-header">📰 News AI System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">🤖 Multi-Task NLP: Classification | Summarization | Translation</p>', unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)
with col1:
    status = "✅ Ready" if classifier_loaded else "❌ Not Loaded"
    st.markdown(f"**Classifier:** {status}")
with col2:
    status = "✅ Ready" if summarizer_loaded else "❌ Not Loaded"
    st.markdown(f"**Summarizer:** {status}")
with col3:
    status = "✅ Ready" if translator_loaded else "❌ Not Loaded"
    st.markdown(f"**Translator:** {status}")

st.markdown("---")


if not (classifier_loaded or summarizer_loaded or translator_loaded):
    st.error("⚠️ **No models loaded!** Please check the sidebar for errors.")
    
    st.stop()


LANG_CODES = {
    "English": "eng_Latn",
    "Hindi": "hin_Deva",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn"
}

LANG_EMOJIS = {
    "English": "🇬🇧",
    "Hindi": "🇮🇳",
    "German": "🇩🇪",
    "Spanish": "🇪🇸",
    "French": "🇫🇷"
}


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 News Classification",
    "📝 News Summarization",
    "🌐 News Translation",
    "📚 Article Database",
    "📜 History & Downloads",

])



with tab1:
    st.markdown(
        '<div class="section-header">'
        '📊 Advanced News Classification & Analysis'
        '</div>',
        unsafe_allow_html=True
    )
    
    if not classifier_loaded:
        st.error("❌ Classifier not loaded.")
    else:
        st.markdown("""
            <div class='info-card'>
                <p style='color: #e0e0e0;'>
                    Classify news articles into four categories. 
                    Articles in other languages will use their 
                    English translation automatically.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        
        st.markdown("""
            <div style='text-align: center; margin: 20px 0;'>
                <span class='category-badge badge-world'>🌍 World</span>
                <span class='category-badge badge-sports'>⚽ Sports</span>
                <span class='category-badge badge-business'>💼 Business</span>
                <span class='category-badge badge-scifi'>🚀 Sci-Fi</span>
            </div>
        """, unsafe_allow_html=True)
        
        
        db_title, db_content, from_db, article_id = article_selector(
            "Classification",
            english_content_only=True,
            key_prefix="class"
        )
        
        if from_db and db_title and db_content:
            class_title = db_title
            class_description = db_content
            
            st.success(
                f"✅ Loaded: **{class_title[:60]}...** "
                f"({len(class_description.split())} words)"
            )
        else:
            article_id = None
            
            st.markdown("### 📌 News Title")
            class_title = st.text_input(
                "Enter news headline (Max 50 words)",
                placeholder="e.g., NASA Discovers Water on Mars",
                key="class_title"
            )
            
            if class_title:
                is_valid, message = validate_title(class_title)
                if is_valid:
                    st.markdown(
                        f"<div class='char-count'>✅ {message}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='char-count error'>❌ {message}</div>",
                        unsafe_allow_html=True
                    )
            
            st.markdown("### 📄 News Description")
            class_description = st.text_area(
                "Enter full news article",
                placeholder="Provide detailed information...",
                height=200,
                key="class_desc"
            )
        
        st.markdown("---")
        
        
        if st.button(
            "🔍 Analyze & Classify Article",
            key="classify_btn",
            use_container_width=True,
            type="primary"
        ):
            if not class_title or not class_description:
                st.error("⚠️ Please enter both title and description!")
            else:
                title_valid = True
                if not from_db:
                    title_valid, _ = validate_title(class_title)
                    if not title_valid:
                        st.error("⚠️ Title exceeds 50 words!")
                
                if title_valid:
                    full_text = class_title + " " + class_description
                    
                    with st.spinner('🔄 Analyzing...'):
                        progress_bar = st.progress(0)
                        
                        progress_bar.progress(20)
                        predicted_class, confidence_scores = classify_news(
                            class_title, class_description
                        )
                        
                        if predicted_class is not None:
                            progress_bar.progress(40)
                            keywords = extract_keywords(full_text, top_n=10)
                            
                            progress_bar.progress(60)
                            sentiment = analyze_sentiment(full_text)
                            
                            progress_bar.progress(70)
                            entities = extract_entities(full_text)
                            
                            progress_bar.progress(80)
                            readability = calculate_readability(full_text)
                            
                            progress_bar.progress(90)
                            influencing_words = get_top_influencing_words(
                                full_text, tfidf_vectorizer,
                                classifier_model, int(predicted_class), 15
                            )
                            
                            subcategories = suggest_subcategory(
                                CATEGORY_INFO[int(predicted_class)]['name'],
                                keywords, full_text
                            )
                            
                            progress_bar.progress(100)
                            time.sleep(0.3)
                            progress_bar.empty()
                            
                            category_name = CATEGORY_INFO[int(predicted_class)]['name']
                            max_confidence = confidence_scores[int(predicted_class)] * 100
                            
                            
                            
                            full_metadata = {
                                'all_scores': {
                                    CATEGORY_INFO[i]['name']: round(
                                        confidence_scores[i] * 100, 2
                                    ) for i in range(1, 5)
                                },
                                'keywords': keywords,
                                'title': class_title,
                                'sentiment': {
                                    'label': sentiment['label'],
                                    'score': sentiment['score'],
                                    'emotions': list(sentiment['emotions'].keys())
                                },
                                'readability': {
                                    'score': readability['flesch_score'],
                                    'level': readability['level']
                                },
                                'subcategories': [s[0] for s in subcategories],
                                'entities': {k: v[:5] for k, v in entities.items()}
                            }
                            
                            st.session_state.class_result = {
                                'title': class_title,
                                'description': class_description,
                                'category_name': category_name,
                                'category_color': CATEGORY_INFO[int(predicted_class)]['color'],
                                'category_emoji': CATEGORY_INFO[int(predicted_class)]['emoji'],
                                'max_confidence': max_confidence,
                                'confidence_scores': {
                                    i: confidence_scores[i] for i in range(1, 5)
                                },
                                'predicted_class': int(predicted_class),
                                'keywords': keywords,
                                'sentiment': sentiment,
                                'entities': entities,
                                'readability': readability,
                                'influencing_words': influencing_words,
                                'subcategories': subcategories,
                                'metadata': full_metadata,
                                'from_db': from_db,
                                'article_id': article_id
                            }
        
       
        
        if st.session_state.class_result:
            r = st.session_state.class_result
            
            st.markdown("---")
            
            
            st.markdown(f"""
                <div class='prediction-box' 
                style='background: linear-gradient(135deg, 
                {r['category_color']}33, {r['category_color']}66); 
                border: 3px solid {r['category_color']};'>
                    {r['category_emoji']} Predicted: 
                    <span style='color:{r["category_color"]};'>
                        {r['category_name'].upper()}
                    </span>
                    <br>
                    <span style='font-size:18px; color:#e0e0e0;'>
                        Confidence: {r['max_confidence']:.1f}%
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            
            if r['subcategories']:
                st.markdown("#### 🏷️ Suggested Sub-categories")
                subcat_html = ""
                for subcat, match_count in r['subcategories']:
                    subcat_html += (
                        f"<span style='background:{r['category_color']}; "
                        f"color:white; padding:5px 15px; border-radius:20px; "
                        f"margin:5px; display:inline-block; font-size:14px;'>"
                        f"{subcat} ({match_count} matches)</span>"
                    )
                st.markdown(
                    f"<div style='text-align:center; margin:10px 0;'>"
                    f"{subcat_html}</div>",
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            
            (a_tab1, a_tab2, a_tab3, 
             a_tab4, a_tab5) = st.tabs([
                "📊 Confidence",
                "💭 Sentiment",
                "🔍 Entities",
                "📖 Readability",
                "⚡ Why This Category?"
            ])
            
            with a_tab1:
                viz_col1, viz_col2 = st.columns([3, 2])
                
                with viz_col1:
                    cats = [CATEGORY_INFO[i]['name'] for i in range(1, 5)]
                    scores = [r['confidence_scores'][i] * 100 for i in range(1, 5)]
                    clrs = [CATEGORY_INFO[i]['color'] for i in range(1, 5)]
                    
                    fig = go.Figure(go.Bar(
                        x=scores, y=cats, orientation='h',
                        marker=dict(color=clrs, line=dict(color='white', width=2)),
                        text=[f"{s:.1f}%" for s in scores],
                        textposition='auto',
                    ))
                    fig.update_layout(
                        height=300,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e0e0e0', 'size': 14},
                        xaxis={'gridcolor': 'rgba(52,152,219,0.2)', 'title': 'Confidence (%)', 'range': [0, 100]},
                        yaxis={'gridcolor': 'rgba(52,152,219,0.2)'},
                        margin=dict(l=20, r=20, t=20, b=20),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with viz_col2:
                    for i in range(1, 5):
                        score = r['confidence_scores'][i] * 100
                        st.markdown(f"""
                            <div class='stats-box'>
                                <strong style='color:{CATEGORY_INFO[i]["color"]};'>
                                    {CATEGORY_INFO[i]["emoji"]} {CATEGORY_INFO[i]["name"]}
                                </strong><br>
                                <span style='font-size:24px; color:#e0e0e0;'>
                                    {score:.2f}%
                                </span>
                            </div>
                        """, unsafe_allow_html=True)
            
            with a_tab2:
                s = r['sentiment']
                sent_col1, sent_col2 = st.columns(2)
                with sent_col1:
                    st.markdown(f"""
                        <div style='text-align:center; padding:30px;
                        background:linear-gradient(135deg, {s['color']}22, {s['color']}44);
                        border-radius:20px; border:3px solid {s['color']};'>
                            <div style='font-size:60px;'>{s['emoji']}</div>
                            <h2 style='color:{s["color"]} !important;'>{s['label']}</h2>
                            <p style='color:#e0e0e0;'>
                                Score: {s['score']:.2f} | Intensity: {s['intensity']:.1f}%
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                with sent_col2:
                    if s['emotions']:
                        for emotion, count in s['emotions'].items():
                            st.markdown(f"<div class='stats-box'><strong>{emotion}</strong><br>{count} indicators</div>", unsafe_allow_html=True)
                    if s['positive_words']:
                        pw = "".join([f"<span style='background:#27ae60;color:white;padding:3px 8px;border-radius:8px;margin:2px;display:inline-block;font-size:12px;'>✅{w}</span>" for w in s['positive_words'][:6]])
                        st.markdown(f"**Positive:** {pw}", unsafe_allow_html=True)
                    if s['negative_words']:
                        nw = "".join([f"<span style='background:#e74c3c;color:white;padding:3px 8px;border-radius:8px;margin:2px;display:inline-block;font-size:12px;'>❌{w}</span>" for w in s['negative_words'][:6]])
                        st.markdown(f"**Negative:** {nw}", unsafe_allow_html=True)
            
            with a_tab3:
                if r['entities']:
                    for cat_name, items in r['entities'].items():
                        st.markdown(f"**{cat_name}**")
                        items_html = "".join([f"<span style='background:rgba(52,152,219,0.3);color:#e0e0e0;padding:4px 12px;border-radius:8px;margin:3px;display:inline-block;border:1px solid rgba(52,152,219,0.5);'>{item}</span>" for item in items[:10]])
                        st.markdown(items_html, unsafe_allow_html=True)
                else:
                    st.info("No named entities detected.")
            
            with a_tab4:
                rd = r['readability']
                rd_col1, rd_col2 = st.columns(2)
                with rd_col1:
                    st.markdown(f"""
                        <div style='text-align:center; padding:30px;
                        background:linear-gradient(135deg, {rd['color']}22, {rd['color']}44);
                        border-radius:20px; border:3px solid {rd['color']};'>
                            <div style='font-size:50px;'>{rd['emoji']}</div>
                            <h2 style='color:{rd["color"]} !important;'>{rd['level']}</h2>
                            <p style='color:#e0e0e0;'>Flesch: {rd['flesch_score']:.1f}/100 | Grade: {rd['grade']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                with rd_col2:
                    st.metric("📝 Words", rd['total_words'])
                    st.metric("📄 Sentences", rd['total_sentences'])
                    st.metric("📏 Avg Word Length", f"{rd['avg_word_length']:.1f}")
                    st.metric("📐 Avg Sentence Length", f"{rd['avg_sentence_length']:.1f}")
            
            with a_tab5:
                if r['influencing_words']:
                    inf_words = [w['word'] for w in r['influencing_words'][:12]]
                    inf_scores = [w['influence'] for w in r['influencing_words'][:12]]
                    fig_inf = go.Figure(go.Bar(
                        x=inf_scores, y=inf_words, orientation='h',
                        marker=dict(color=r['category_color']),
                        text=[f"{s:.4f}" for s in inf_scores],
                        textposition='auto',
                    ))
                    fig_inf.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font={'color': '#e0e0e0'},
                        xaxis={'title': 'Influence', 'gridcolor': 'rgba(52,152,219,0.2)'},
                        yaxis={'gridcolor': 'rgba(52,152,219,0.2)', 'autorange': 'reversed'},
                        margin=dict(l=20, r=20, t=10, b=40),
                    )
                    st.plotly_chart(fig_inf, use_container_width=True)
                else:
                    st.info("Could not extract influence data.")
            
            
            st.markdown("---")
            st.markdown("### 🔑 Keywords")
            kw_html = "".join([f"<span class='keyword-tag'>{kw}</span>" for kw in r['keywords']])
            st.markdown(f"<div style='text-align:center;margin:20px 0;'>{kw_html}</div>", unsafe_allow_html=True)
            
            # Downloads
            st.markdown("### 📥 Downloads")
            dl1, dl2 = st.columns(2)
            with dl1:
                pdf_buf = create_classification_pdf(r['title'], r['description'], r['category_name'], r['max_confidence'], r['keywords'])
                st.download_button("📄 PDF", pdf_buf, f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf", use_container_width=True, key="cls_pdf")
            with dl2:
                csv_d = pd.DataFrame([{'Title': r['title'], 'Category': r['category_name'], 'Confidence': f"{r['max_confidence']:.2f}%", 'Sentiment': r['sentiment']['label'], 'Keywords': ", ".join(r['keywords'])}])
                st.download_button("📊 CSV", csv_d.to_csv(index=False), f"classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True, key="cls_csv")
            
            
            st.markdown("---")
            st.markdown("### 💾 Save to Database")
            
            if r['from_db'] and r['article_id']:
                
                db.save_result(
                    article_id=r['article_id'],
                    task_type='classification',
                    result_data=r['category_name'],
                    confidence=r['max_confidence'],
                    metadata=r['metadata']
                )
                db.update_category(r['article_id'], r['category_name'])
                st.success("✅ Result saved & category updated!")
                
                st.session_state.class_result = None
            
            else:
                
                save_col1, save_col2 = st.columns([2, 1])
                
                with save_col1:
                    if st.button(
                        "💾 Save Article + Result to Database",
                        key="save_class_to_db",
                        use_container_width=True
                    ):
                        success, msg, new_id = db.add_article(
                            title=r['title'],
                            content=r['description'],
                            category=r['category_name'],
                            language="English",
                            tags=", ".join(r['keywords'][:5])
                        )
                        
                        if success and new_id:
                            db.save_result(
                                article_id=new_id,
                                task_type='classification',
                                result_data=r['category_name'],
                                confidence=r['max_confidence'],
                                metadata=r['metadata']
                            )
                            st.success(f"✅ {msg} + Result saved!")
                            st.balloons()
                           
                            st.session_state.class_result = None
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {msg}")
                
                with save_col2:
                    st.markdown(
                        f"<div class='stats-box'>"
                        f"<strong>📚 DB</strong><br>"
                        f"{db.get_count()}/100</div>",
                        unsafe_allow_html=True
                    )
            
            
            history_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'title': r['title'][:50] + "...",
                'category': r['category_name'],
                'confidence': r['max_confidence'],
                'sentiment': r['sentiment']['label'],
                'keywords': ", ".join(r['keywords'][:5]),
            }
            if history_record not in st.session_state.classification_history:
                st.session_state.classification_history.insert(0, history_record)
                if len(st.session_state.classification_history) > 5:
                    st.session_state.classification_history = st.session_state.classification_history[:5]
    


with tab2:
    st.markdown(
        '<div class="section-header">'
        '📝 Advanced News Summarization'
        '</div>',
        unsafe_allow_html=True
    )
    
    if not summarizer_loaded:
        st.error("❌ Summarizer not loaded.")
    else:
        st.markdown("""
            <div class='info-card'>
                <p style='color: #e0e0e0;'>
                    <strong>Features:</strong> 
                    Adjustable length | Bullet points | 
                    Classification | Summary translation
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.warning("⚠️ Article must contain at least **50 words**.")
        
       
        db_title, db_content, from_db, article_id = article_selector(
            "Summarization",
            english_content_only=True,
            key_prefix="summ"
        )
        
        if from_db and db_content:
            summary_text = db_content
            st.success(
                f"✅ Loaded: **{db_title[:60]}...** "
                f"({len(summary_text.split())} words)"
            )
        else:
            article_id = None
            summary_text = st.text_area(
                "📄 Enter article (minimum 50 words)",
                placeholder="Paste your article here...",
                height=300,
                key="summary_input"
            )
        
        if summary_text:
            stats = get_text_stats(summary_text)
            if stats['words'] < 50:
                st.markdown(
                    f"<div class='char-count error'>"
                    f"❌ {stats['words']}/50 words</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='char-count'>"
                    f"✅ {stats['words']} words</div>",
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
     
        
        st.markdown("### ⚙️ Summarization Settings")
        
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
        
        with ctrl_col1:
            
            summary_pct = st.slider(
                "📏 Summary Length",
                min_value=10,
                max_value=60,
                value=30,
                step=5,
                format="%d%%",
                help="Percentage of original article length",
                key="summ_slider"
            )
            
            summary_ratio = summary_pct / 100.0
            
         
            if summary_text:
                est_words = int(len(summary_text.split()) * summary_ratio)
                st.caption(
                    f"Estimated summary: ~{est_words} words "
                    f"from {len(summary_text.split())} words"
                )
        
        with ctrl_col2:
            
            translate_summary = st.checkbox(
                "🌐 Translate Summary",
                value=False,
                key="summ_translate_check"
            )
            
            if translate_summary:
                if translator_loaded:
                    target_lang_summ = st.selectbox(
                        "Target Language",
                        ["Hindi", "German", "Spanish", "French"],
                        key="summ_target_lang"
                    )
                else:
                    st.warning("Translator not loaded!")
                    translate_summary = False
        
        with ctrl_col3:
          
            classify_article = st.checkbox(
                "📊 Classify Article",
                value=True,
                key="summ_classify_check",
                disabled=not classifier_loaded
            )
            
        
            num_bullets = st.selectbox(
                "📋 Bullet Points",
                [3, 5, 7, 10],
                index=1,
                key="summ_bullets_count"
            )
        
        st.markdown("---")
        
        if st.button(
            "✨ Generate Complete Analysis",
            key="summarize_btn",
            use_container_width=True,
            type="primary"
        ):
            if not summary_text:
                st.error("⚠️ Please enter an article!")
            elif len(summary_text.split()) < 50:
                st.error(
                    f"⚠️ Minimum 50 words "
                    f"(current: {len(summary_text.split())})"
                )
            else:
                with st.spinner('🔄 Processing...'):
                    
                   
                    summary = summarize_text(
                        summary_text, summary_ratio
                    )
                    
                    if summary:
                        orig_stats = get_text_stats(summary_text)
                        summ_stats = get_text_stats(summary)
                        reduction = (
                            (orig_stats['words'] - summ_stats['words'])
                            / orig_stats['words']
                        ) * 100
                        keywords = extract_keywords(summary, top_n=8)
                        
                        bullet_points = extract_bullet_points(
                            summary_text, num_bullets
                        )
                        
                       
                        classification_result = None
                        if classify_article and classifier_loaded:
                           
                            auto_title = ' '.join(
                                summary_text.split()[:15]
                            )
                            pred_class, conf_scores = classify_news(
                                auto_title, summary_text
                            )
                            if pred_class is not None:
                                cat_name = CATEGORY_INFO[
                                    int(pred_class)
                                ]['name']
                                cat_emoji = CATEGORY_INFO[
                                    int(pred_class)
                                ]['emoji']
                                cat_color = CATEGORY_INFO[
                                    int(pred_class)
                                ]['color']
                                max_conf = (
                                    conf_scores[int(pred_class)] * 100
                                )
                                classification_result = {
                                    'category': cat_name,
                                    'emoji': cat_emoji,
                                    'color': cat_color,
                                    'confidence': max_conf,
                                    'all_scores': {
                                        CATEGORY_INFO[i]['name']: round(
                                            conf_scores[i] * 100, 2
                                        ) for i in range(1, 5)
                                    }
                                }
                        
                    
                        translated_summary = None
                        translated_lang = None
                        if (translate_summary and translator_loaded
                                and target_lang_summ):
                            target_code = LANG_CODES.get(
                                target_lang_summ, "hin_Deva"
                            )
                            st.info(
                                f"🌐 Translating summary to "
                                f"{target_lang_summ}..."
                            )
                            translated_summary = translate_bidirectional(
                                summary, "eng_Latn", target_code
                            )
                            translated_lang = target_lang_summ
                        
                        
                        st.session_state.summ_result = {
                            'original_text': summary_text,
                            'summary': summary,
                            'bullet_points': bullet_points,
                            'classification': classification_result,
                            'translated_summary': translated_summary,
                            'translated_lang': translated_lang,
                            'keywords': keywords,
                            'orig_stats': orig_stats,
                            'summ_stats': summ_stats,
                            'reduction': reduction,
                            'summary_ratio': summary_ratio,
                            'from_db': from_db,
                            'article_id': article_id,
                            'db_title': db_title
                        }
        
       
        if st.session_state.summ_result:
            r = st.session_state.summ_result
            
            st.markdown("---")
            
            
            if r['classification']:
                c = r['classification']
                st.markdown(f"""
                    <div style='text-align:center; padding:15px;
                    background:linear-gradient(135deg, 
                    {c['color']}22, {c['color']}44);
                    border-radius:15px; border:2px solid {c['color']};
                    margin-bottom:20px;'>
                        <span style='font-size:16px; color:#e0e0e0;'>
                            📊 Classification:
                        </span>
                        <span style='font-size:24px; font-weight:bold;
                        color:{c["color"]};'>
                            {c['emoji']} {c['category']}
                        </span>
                        <span style='font-size:16px; color:#e0e0e0;'>
                            ({c['confidence']:.1f}% confidence)
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
        -
            st.markdown("### 📊 Summary Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(
                    "<div class='result-box'>"
                    "<h4 style='color:#3498db;'>📄 Original</h4>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                preview = r['original_text'][:1000]
                if len(r['original_text']) > 1000:
                    preview += "..."
                st.markdown(
                    f"<div style='color:#e0e0e0; padding:10px;'>"
                    f"{preview}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                    <div class='stats-box'>
                        <strong>Words:</strong> {r['orig_stats']['words']}
                        <br>
                        <strong>Characters:</strong> 
                        {r['orig_stats']['characters']}
                    </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(
                    "<div class='result-box'>"
                    "<h4 style='color:#27ae60;'>✨ Summary</h4>"
                    "</div>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div style='color:#e0e0e0; padding:10px; "
                    f"background:rgba(39,174,96,0.1); "
                    f"border-radius:10px;'>{r['summary']}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                    <div class='stats-box'>
                        <strong>Words:</strong> {r['summ_stats']['words']}
                        | <strong style='color:#27ae60;'>
                        Reduction:</strong> {r['reduction']:.1f}%
                        | <strong>Target:</strong> 
                        {int(r['summary_ratio']*100)}%
                    </div>
                """, unsafe_allow_html=True)
            
            
            if r['bullet_points']:
                st.markdown("### 📋 Key Points")
                
                st.markdown(
                    "<div style='background:rgba(52,152,219,0.1); "
                    "padding:20px; border-radius:15px; "
                    "border-left:5px solid #3498db;'>",
                    unsafe_allow_html=True
                )
                
                for i, point in enumerate(r['bullet_points']):
                    st.markdown(
                        f"<div style='color:#e0e0e0; padding:8px 0; "
                        f"border-bottom:1px solid rgba(52,152,219,0.2);'>"
                        f"<strong style='color:#3498db;'>"
                        f"{'●'} Point {i+1}:</strong> {point}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
            
         
            if r['translated_summary']:
                t_lang = r['translated_lang']
                t_emoji = LANG_EMOJIS.get(t_lang, '🌐')
                
                st.markdown(
                    f"### {t_emoji} Summary in {t_lang}"
                )
                
                st.markdown(f"""
                    <div style='background:rgba(155,89,182,0.1);
                    padding:20px; border-radius:15px;
                    border-left:5px solid #9b59b6;
                    color:#e0e0e0;'>
                        {r['translated_summary']}
                    </div>
                """, unsafe_allow_html=True)
                
                st.code(r['translated_summary'], language=None)
            
           
            if r['classification']:
                c = r['classification']
                
                with st.expander(
                    f"📊 Classification Details — "
                    f"{c['emoji']} {c['category']}",
                    expanded=False
                ):
                    score_cols = st.columns(4)
                    for idx, (cat, score) in enumerate(
                        c['all_scores'].items()
                    ):
                        with score_cols[idx % 4]:
                            cat_color = {
                                "World": "#3498db",
                                "Sports": "#e74c3c",
                                "Business": "#f39c12",
                                "Sci-Fi": "#9b59b6"
                            }.get(cat, '#95a5a6')
                            
                            st.markdown(f"""
                                <div class='stats-box'>
                                    <strong style='color:{cat_color};'>
                                        {cat}
                                    </strong><br>
                                    <span style='font-size:20px;'>
                                        {score:.1f}%
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
            
            
            st.markdown("### 🔑 Key Topics")
            kw_html = "".join(
                [f"<span class='keyword-tag'>{kw}</span>"
                 for kw in r['keywords']]
            )
            st.markdown(
                f"<div style='text-align:center; margin:20px 0;'>"
                f"{kw_html}</div>",
                unsafe_allow_html=True
            )
            
            
            st.markdown("### 📋 Copy Summary")
            st.code(r['summary'], language=None)
            
       
            st.markdown("### 📥 Downloads")
            dl1, dl2 = st.columns(2)
            
            with dl1:
                pdf_buf = create_summary_pdf(
                    r['original_text'], r['summary'], r['reduction']
                )
                st.download_button(
                    "📄 PDF", pdf_buf,
                    f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "application/pdf",
                    use_container_width=True, key="smm_pdf"
                )
            
            with dl2:
                csv_dict = {
                    'Original Words': r['orig_stats']['words'],
                    'Summary Words': r['summ_stats']['words'],
                    'Reduction': f"{r['reduction']:.1f}%",
                    'Summary': r['summary'],
                    'Bullet Points': " | ".join(
                        r['bullet_points']
                    ) if r['bullet_points'] else "",
                    'Keywords': ", ".join(r['keywords']),
                }
                if r['classification']:
                    csv_dict['Category'] = r['classification']['category']
                    csv_dict['Confidence'] = (
                        f"{r['classification']['confidence']:.1f}%"
                    )
                if r['translated_summary']:
                    csv_dict[f'Summary ({r["translated_lang"]})'] = (
                        r['translated_summary']
                    )
                
                csv_d = pd.DataFrame([csv_dict])
                st.download_button(
                    "📊 CSV", csv_d.to_csv(index=False),
                    f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True, key="smm_csv"
                )
            
            
            st.markdown("---")
            st.markdown("### 💾 Save to Database")
            
            if r['from_db'] and r['article_id']:
                metadata = {
                    'original_words': r['orig_stats']['words'],
                    'summary_words': r['summ_stats']['words'],
                    'reduction_pct': round(r['reduction'], 1),
                    'keywords': r['keywords'],
                    'bullet_points': r['bullet_points'],
                    'summary_ratio': r['summary_ratio']
                }
                if r['classification']:
                    metadata['classification'] = r['classification']
                if r['translated_summary']:
                    metadata['translated_summary'] = {
                        'text': r['translated_summary'],
                        'language': r['translated_lang']
                    }
                
                db.save_result(
                    article_id=r['article_id'],
                    task_type='summarization',
                    result_data=r['summary'],
                    confidence=r['reduction'],
                    metadata=metadata
                )
                st.success("✅ Complete analysis saved to database!")
                st.session_state.summ_result = None
            
            else:
                save_col1, save_col2 = st.columns([2, 1])
                
                with save_col1:
                    if st.button(
                        "💾 Save Article + Analysis to Database",
                        key="save_summ_to_db",
                        use_container_width=True
                    ):
                        auto_title = r['original_text'][:80].strip()
                        if len(r['original_text']) > 80:
                            auto_title += "..."
                        
                        cat = "Uncategorized"
                        if r['classification']:
                            cat = r['classification']['category']
                        
                        success, msg, new_id = db.add_article(
                            title=auto_title,
                            content=r['original_text'],
                            category=cat,
                            language="English",
                            tags=", ".join(r['keywords'][:5])
                        )
                        
                        if success and new_id:
                            metadata = {
                                'original_words': r['orig_stats']['words'],
                                'summary_words': r['summ_stats']['words'],
                                'reduction_pct': round(r['reduction'], 1),
                                'keywords': r['keywords'],
                                'bullet_points': r['bullet_points'],
                            }
                            if r['classification']:
                                metadata['classification'] = (
                                    r['classification']
                                )
                            if r['translated_summary']:
                                metadata['translated_summary'] = {
                                    'text': r['translated_summary'],
                                    'language': r['translated_lang']
                                }
                            
                            db.save_result(
                                article_id=new_id,
                                task_type='summarization',
                                result_data=r['summary'],
                                confidence=r['reduction'],
                                metadata=metadata
                            )
                            st.success(f"✅ {msg} + Analysis saved!")
                            st.balloons()
                            st.session_state.summ_result = None
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {msg}")
                
                with save_col2:
                    st.markdown(
                        f"<div class='stats-box'>"
                        f"<strong>📚 DB</strong><br>"
                        f"{db.get_count()}/100</div>",
                        unsafe_allow_html=True
                    )
            
            
            h = {
                'timestamp': datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'summary': r['summary'][:100] + "...",
                'original_words': r['orig_stats']['words'],
                'summary_words': r['summ_stats']['words'],
                'reduction': r['reduction'],
                'keywords': ", ".join(r['keywords'])
            }
            if h not in st.session_state.summarization_history:
                st.session_state.summarization_history.insert(0, h)
                if len(st.session_state.summarization_history) > 5:
                    st.session_state.summarization_history = (
                        st.session_state.summarization_history[:5]
                    )
with tab3:
    st.markdown(
        '<div class="section-header">'
        '🌐 Bidirectional Translation'
        '</div>',
        unsafe_allow_html=True
    )
    
    if not translator_loaded:
        st.error("❌ Translator not loaded.")
    else:
        st.markdown("""
            <div class='info-card'>
                <p style='color: #e0e0e0;'>
                    <strong>Bidirectional:</strong> 
                    Translate between English ↔ Hindi, German, 
                    Spanish, French. Optionally summarize and 
                    classify results.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
       
        
        direction = st.radio(
            "🔄 Translation Direction:",
            [
                "🌍 Other Language → 🇬🇧 English",
                "🇬🇧 English → 🌍 Other Language"
            ],
            horizontal=True,
            key="trans_direction"
        )
        
        is_to_english = direction.startswith("🌍")
        
        st.markdown("---")
        

        
        if is_to_english:
            
            other_languages = {
                "🇮🇳 Hindi": ("Hindi", "hin_Deva"),
                "🇩🇪 German": ("German", "deu_Latn"),
                "🇪🇸 Spanish": ("Spanish", "spa_Latn"),
                "🇫🇷 French": ("French", "fra_Latn")
            }
            
            selected = st.selectbox(
                "🌍 Source Language:",
                list(other_languages.keys()),
                key="trans_source_lang"
            )
            
            source_name, source_code = other_languages[selected]
            target_name = "English"
            target_code = "eng_Latn"
            source_emoji = selected.split()[0]
            target_emoji = "🇬🇧"
            
            
            db_filter_lang = source_name
        
        else:
          
            other_languages = {
                "🇮🇳 Hindi": ("Hindi", "hin_Deva"),
                "🇩🇪 German": ("German", "deu_Latn"),
                "🇪🇸 Spanish": ("Spanish", "spa_Latn"),
                "🇫🇷 French": ("French", "fra_Latn")
            }
            
            selected = st.selectbox(
                "🌍 Target Language:",
                list(other_languages.keys()),
                key="trans_target_lang"
            )
            
            source_name = "English"
            source_code = "eng_Latn"
            target_name, target_code = other_languages[selected]
            source_emoji = "🇬🇧"
            target_emoji = selected.split()[0]
            
         
            db_filter_lang = "English"
        
        st.markdown(
            f"<div style='text-align:center; padding:10px; "
            f"background:rgba(155,89,182,0.1); border-radius:10px; "
            f"margin:10px 0;'>"
            f"<span style='font-size:20px; color:#e0e0e0;'>"
            f"{source_emoji} {source_name} → "
            f"{target_emoji} {target_name}"
            f"</span></div>",
            unsafe_allow_html=True
        )
        
        
        db_title, db_content, from_db, article_id = article_selector(
            "Translation",
            filter_language=db_filter_lang,
            key_prefix="trans"
        )
        
        if from_db and db_content:
            translation_text = db_content
            st.success(
                f"✅ Loaded: **{db_title[:60]}...** "
                f"({len(translation_text.split())} words)"
            )
        else:
            article_id = None
            translation_text = st.text_area(
                f"📄 Enter {source_name} text:",
                placeholder=f"Paste your {source_name} text...",
                height=300,
                key="trans_text_input"
            )
        
        if translation_text:
            stats = get_text_stats(translation_text)
            st.markdown(
                f"<div class='char-count'>"
                f"📊 {stats['words']} words | "
                f"{stats['characters']} characters</div>",
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
       
        
        st.markdown("### ⚙️ Additional Options")
        
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            
            if is_to_english:
                summ_help = (
                    "Summarize the English translation"
                )
            else:
                summ_help = (
                    "Summarize the English source before translating"
                )
            
            do_summarize = st.checkbox(
                "📝 Summarize",
                value=False,
                key="trans_do_summarize",
                disabled=not summarizer_loaded,
                help=summ_help
            )
            
            if do_summarize and summarizer_loaded:
                trans_summ_ratio = st.slider(
                    "Summary Length",
                    min_value=15, max_value=50,
                    value=30, step=5,
                    format="%d%%",
                    key="trans_summ_slider"
                ) / 100.0
        
        with opt_col2:
            
            if is_to_english:
                cls_help = (
                    "Classify the English translation"
                )
            else:
                cls_help = (
                    "Classify the English source"
                )
            
            do_classify = st.checkbox(
                "📊 Classify",
                value=False,
                key="trans_do_classify",
                disabled=not classifier_loaded,
                help=cls_help
            )
        
        st.markdown("---")
        
     
        if st.button(
            f"🔄 Translate {source_name} → {target_name}",
            key="trans_btn_main",
            type="primary",
            use_container_width=True
        ):
            if not translation_text or not translation_text.strip():
                st.error("⚠️ Please enter text!")
            else:
                with st.spinner(
                    f'🔄 Translating {source_name} → {target_name}...'
                ):
                    # Step 1: Translate
                    translated = translate_bidirectional(
                        translation_text, source_code, target_code
                    )
                    
                    if translated:
                        src_stats = get_text_stats(translation_text)
                        tgt_stats = get_text_stats(translated)
                        trans_keywords = extract_keywords(
                            translated if is_to_english
                            else translation_text,
                            top_n=8
                        )
                        
                     
                        summary_result = None
                        if do_summarize and summarizer_loaded:
                          
                            if is_to_english:
                               
                                summ_text = translated
                            else:
                                
                                summ_text = translation_text
                            
                            if len(summ_text.split()) >= 50:
                                st.info("📝 Generating summary...")
                                summ = summarize_text(
                                    summ_text, trans_summ_ratio
                                )
                                if summ:
                                    summary_result = {
                                        'text': summ,
                                        'source': (
                                            'translation'
                                            if is_to_english
                                            else 'source'
                                        ),
                                        'words': len(summ.split())
                                    }
                            else:
                                st.warning(
                                    "⚠️ Text too short to summarize "
                                    "(need 50+ words)"
                                )
                        
                        
                        classification_result = None
                        if do_classify and classifier_loaded:
                            if is_to_english:
                                cls_text = translated
                            else:
                                cls_text = translation_text
                            
                            auto_title = ' '.join(
                                cls_text.split()[:15]
                            )
                            pred_class, conf_scores = classify_news(
                                auto_title, cls_text
                            )
                            
                            if pred_class is not None:
                                classification_result = {
                                    'category': CATEGORY_INFO[
                                        int(pred_class)
                                    ]['name'],
                                    'emoji': CATEGORY_INFO[
                                        int(pred_class)
                                    ]['emoji'],
                                    'color': CATEGORY_INFO[
                                        int(pred_class)
                                    ]['color'],
                                    'confidence': (
                                        conf_scores[int(pred_class)]
                                        * 100
                                    ),
                                    'all_scores': {
                                        CATEGORY_INFO[i]['name']: round(
                                            conf_scores[i] * 100, 2
                                        ) for i in range(1, 5)
                                    }
                                }
                     
                        
                        st.session_state.trans_result = {
                            'source_text': translation_text,
                            'translated': translated,
                            'source_name': source_name,
                            'target_name': target_name,
                            'source_code': source_code,
                            'target_code': target_code,
                            'source_emoji': source_emoji,
                            'target_emoji': target_emoji,
                            'is_to_english': is_to_english,
                            'src_stats': src_stats,
                            'tgt_stats': tgt_stats,
                            'keywords': trans_keywords,
                            'summary': summary_result,
                            'classification': classification_result,
                            'from_db': from_db,
                            'article_id': article_id,
                            'db_title': db_title
                        }
                        
                   
                        if from_db and article_id and is_to_english:
                            db.update_translated_content(
                                article_id, translated
                            )
                    else:
                        st.error("❌ Translation failed.")
        
       
        
        if st.session_state.trans_result:
            r = st.session_state.trans_result
            
            st.markdown("---")
            
          
            if r['classification']:
                c = r['classification']
                st.markdown(f"""
                    <div style='text-align:center; padding:15px;
                    background:linear-gradient(135deg,
                    {c['color']}22, {c['color']}44);
                    border-radius:15px; border:2px solid {c['color']};
                    margin-bottom:20px;'>
                        <span style='color:#e0e0e0;'>
                            📊 Classification:
                        </span>
                        <span style='font-size:24px; font-weight:bold;
                        color:{c["color"]};'>
                            {c['emoji']} {c['category']}
                        </span>
                        <span style='color:#e0e0e0;'>
                            ({c['confidence']:.1f}%)
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            
       
            st.markdown("### 🌐 Translation Results")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown(f"""
                    <div class='result-box'>
                        <h4 style='color:#e74c3c;'>
                            {r['source_emoji']} Original 
                            ({r['source_name']})
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(
                    f"<div style='color:#e0e0e0; padding:10px; "
                    f"background:rgba(231,76,60,0.1); "
                    f"border-radius:10px;'>"
                    f"{r['source_text']}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                    <div class='stats-box'>
                        <strong>Words:</strong> 
                        {r['src_stats']['words']}
                    </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(f"""
                    <div class='result-box'>
                        <h4 style='color:#27ae60;'>
                            {r['target_emoji']} Translated 
                            ({r['target_name']})
                        </h4>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(
                    f"<div style='color:#e0e0e0; padding:10px; "
                    f"background:rgba(39,174,96,0.1); "
                    f"border-radius:10px;'>"
                    f"{r['translated']}</div>",
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                    <div class='stats-box'>
                        <strong>Words:</strong> 
                        {r['tgt_stats']['words']}
                    </div>
                """, unsafe_allow_html=True)
            
        
            if r['summary']:
                s = r['summary']
                summ_label = (
                    "English Translation"
                    if s['source'] == 'translation'
                    else "English Source"
                )
                
                st.markdown(
                    f"### 📝 Summary of {summ_label}"
                )
                st.markdown(f"""
                    <div style='background:rgba(39,174,96,0.1);
                    padding:20px; border-radius:15px;
                    border-left:5px solid #27ae60;
                    color:#e0e0e0;'>
                        {s['text']}
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(
                    f"<div class='stats-box'>"
                    f"<strong>Summary Words:</strong> {s['words']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.code(s['text'], language=None)
            
            
            if r['classification']:
                c = r['classification']
                with st.expander(
                    f"📊 Classification — "
                    f"{c['emoji']} {c['category']}",
                    expanded=False
                ):
                    score_cols = st.columns(4)
                    for idx, (cat, score) in enumerate(
                        c['all_scores'].items()
                    ):
                        with score_cols[idx % 4]:
                            cat_color = {
                                "World": "#3498db",
                                "Sports": "#e74c3c",
                                "Business": "#f39c12",
                                "Sci-Fi": "#9b59b6"
                            }.get(cat, '#95a5a6')
                            st.markdown(f"""
                                <div class='stats-box'>
                                    <strong style='color:{cat_color};'>
                                        {cat}
                                    </strong><br>
                                    <span style='font-size:20px;'>
                                        {score:.1f}%
                                    </span>
                                </div>
                            """, unsafe_allow_html=True)
            
            
            if r['keywords']:
                st.markdown("### 🔑 Key Topics")
                kw_html = "".join(
                    [f"<span class='keyword-tag'>{kw}</span>"
                     for kw in r['keywords']]
                )
                st.markdown(
                    f"<div style='text-align:center; "
                    f"margin:20px 0;'>{kw_html}</div>",
                    unsafe_allow_html=True
                )
            
            
            st.markdown("### 📋 Copy Translation")
            st.code(r['translated'], language=None)
            
            
            st.markdown("### 📥 Downloads")
            dl1, dl2 = st.columns(2)
            
            with dl1:
                pdf_buf = create_translation_pdf(
                    r['source_text'], r['translated']
                )
                st.download_button(
                    "📄 PDF", pdf_buf,
                    f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "application/pdf",
                    use_container_width=True, key="trn_pdf"
                )
            
            with dl2:
                csv_dict = {
                    'Direction': (
                        f"{r['source_name']} → {r['target_name']}"
                    ),
                    'Source Text': r['source_text'],
                    'Translation': r['translated'],
                    'Source Words': r['src_stats']['words'],
                    'Target Words': r['tgt_stats']['words'],
                    'Keywords': ", ".join(r['keywords']),
                }
                if r['summary']:
                    csv_dict['Summary'] = r['summary']['text']
                if r['classification']:
                    csv_dict['Category'] = (
                        r['classification']['category']
                    )
                
                csv_d = pd.DataFrame([csv_dict])
                st.download_button(
                    "📊 CSV",
                    csv_d.to_csv(index=False),
                    f"translation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True, key="trn_csv"
                )
            
           
            st.markdown("---")
            st.markdown("### 💾 Save to Database")
            
            if r['from_db'] and r['article_id']:
                metadata = {
                    'source_language': r['source_name'],
                    'target_language': r['target_name'],
                    'direction': (
                        f"{r['source_name']}→{r['target_name']}"
                    ),
                    'source_words': r['src_stats']['words'],
                    'translated_words': r['tgt_stats']['words'],
                    'keywords': r['keywords']
                }
                if r['summary']:
                    metadata['summary'] = r['summary']['text']
                if r['classification']:
                    metadata['classification'] = {
                        'category': r['classification']['category'],
                        'confidence': r['classification']['confidence']
                    }
                
                db.save_result(
                    article_id=r['article_id'],
                    task_type='translation',
                    result_data=r['translated'],
                    confidence=0,
                    metadata=metadata
                )
                
                if r['is_to_english']:
                    st.success(
                        "✅ Translation saved! English version "
                        "available for Classification & Summarization."
                    )
                else:
                    st.success("✅ Translation saved to database!")
                
                st.session_state.trans_result = None
            
            else:
                
                save_col1, save_col2, save_col3 = st.columns(3)
                
                with save_col1:
                    if st.button(
                        f"💾 Save {r['source_name']} + "
                        f"{r['target_name']} to DB",
                        key="save_trans_to_db",
                        use_container_width=True
                    ):
                        auto_title = r['source_text'][:80].strip()
                        if len(r['source_text']) > 80:
                            auto_title += "..."
                        
                        cat = "Uncategorized"
                        if r['classification']:
                            cat = r['classification']['category']
                        
                  
                        if r['is_to_english']:
                            success, msg, new_id = db.add_article(
                                title=auto_title,
                                content=r['source_text'],
                                category=cat,
                                language=r['source_name'],
                                tags=", ".join(r['keywords'][:5]),
                                translated_content=r['translated']
                            )
                        else:
                            success, msg, new_id = db.add_article(
                                title=auto_title,
                                content=r['source_text'],
                                category=cat,
                                language="English",
                                tags=", ".join(r['keywords'][:5]),
                                translated_content=r['translated']
                            )
                        
                        if success and new_id:
                            metadata = {
                                'source_language': r['source_name'],
                                'target_language': r['target_name'],
                                'source_words': r['src_stats']['words'],
                                'translated_words': (
                                    r['tgt_stats']['words']
                                ),
                                'keywords': r['keywords']
                            }
                            if r['summary']:
                                metadata['summary'] = (
                                    r['summary']['text']
                                )
                            if r['classification']:
                                metadata['classification'] = {
                                    'category': (
                                        r['classification']['category']
                                    ),
                                    'confidence': (
                                        r['classification']['confidence']
                                    )
                                }
                            
                            db.save_result(
                                article_id=new_id,
                                task_type='translation',
                                result_data=r['translated'],
                                confidence=0,
                                metadata=metadata
                            )
                            
                            st.success(f"✅ {msg} + Result saved!")
                            st.balloons()
                            st.session_state.trans_result = None
                            st.rerun()
                        else:
                            st.warning(f"⚠️ {msg}")
                
                with save_col2:
                    if r['is_to_english']:
                        if st.button(
                            "💾 Save ONLY English as Article",
                            key="save_eng_only",
                            use_container_width=True
                        ):
                            eng_title = r['translated'][:80].strip()
                            if len(r['translated']) > 80:
                                eng_title += "..."
                            
                            success, msg, _ = db.add_article(
                                title=f"[Translated] {eng_title}",
                                content=r['translated'],
                                category="Uncategorized",
                                language="English",
                                tags=f"translated, {r['source_name']}"
                            )
                            if success:
                                st.success("✅ English article saved!")
                            else:
                                st.warning(f"⚠️ {msg}")
                    else:
                        if st.button(
                            f"💾 Save ONLY {r['target_name']} Version",
                            key="save_target_only",
                            use_container_width=True
                        ):
                            tgt_title = r['translated'][:80].strip()
                            if len(r['translated']) > 80:
                                tgt_title += "..."
                            
                            success, msg, _ = db.add_article(
                                title=(
                                    f"[{r['target_name']}] {tgt_title}"
                                ),
                                content=r['translated'],
                                category="Uncategorized",
                                language=r['target_name'],
                                tags=f"translated, {r['source_name']}"
                            )
                            if success:
                                st.success(
                                    f"✅ {r['target_name']} "
                                    f"article saved!"
                                )
                            else:
                                st.warning(f"⚠️ {msg}")
                
                with save_col3:
                    st.markdown(
                        f"<div class='stats-box'>"
                        f"<strong>📚 DB</strong><br>"
                        f"{db.get_count()}/100</div>",
                        unsafe_allow_html=True
                    )
            
            
            h = {
                'timestamp': datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'source_language': r['source_name'],
                'target_language': r['target_name'],
                'source_text': r['source_text'][:100] + "...",
                'english': (
                    r['translated'][:100] + "..."
                    if r['is_to_english']
                    else r['source_text'][:100] + "..."
                ),
                'source_words': r['src_stats']['words'],
                'english_words': r['tgt_stats']['words'],
                'keywords': ", ".join(r['keywords'])
            }
            if h not in st.session_state.translation_history:
                st.session_state.translation_history.insert(0, h)
                if len(st.session_state.translation_history) > 5:
                    st.session_state.translation_history = (
                        st.session_state.translation_history[:5]
                    )
with tab4:
    st.markdown(
        '<div class="section-header">📚 Article Database</div>',
        unsafe_allow_html=True
    )
    
    
    count = db.get_count()
    
    st.markdown(f"""
        <div class='info-card'>
            <p style='color: #e0e0e0;'>
                Store up to <strong>100</strong> news articles locally.
                Use them across Classification, Summarization & Translation 
                without pasting every time.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    
    capacity_pct = count / 100
    capacity_color = (
        "#27ae60" if capacity_pct < 0.7 
        else "#f39c12" if capacity_pct < 0.9 
        else "#e74c3c"
    )
    
    col_cap1, col_cap2 = st.columns([3, 1])
    with col_cap1:
        st.progress(capacity_pct)
    with col_cap2:
        st.markdown(
            f"<h3 style='color:{capacity_color}; text-align:center;'>"
            f"{count}/100</h3>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    

    db_tab1, db_tab2, db_tab3, db_tab4, db_tab5 = st.tabs([
        "➕ Add Article",
        "📋 Browse & Search",
        "📊 View Results",
        "✏️ Edit / Delete",
        "📤 Import / Export"
    ])
    
    
    with db_tab1:
        st.markdown("### ➕ Add New Article")
        
        with st.form("add_article_form", clear_on_submit=True):
            
            add_title = st.text_input(
                "📌 Title *",
                placeholder="Enter article headline",
                max_chars=200
            )
            
            add_content = st.text_area(
                "📄 Content *",
                placeholder="Paste full article text here...",
                height=250
            )
            
            add_col1, add_col2, add_col3 = st.columns(3)
            
            with add_col1:
                add_category = st.selectbox(
                    "🏷️ Category",
                    ["Uncategorized", "World", "Sports",
                     "Business", "Sci-Fi", "Technology",
                     "Health", "Entertainment", "Politics", "Other"]
                )
            
            with add_col2:
                add_language = st.selectbox(
                    "🌍 Language",
                    ["English", "Hindi", "German", 
                     "Spanish", "French", "Other"]
                )
            
            with add_col3:
                add_tags = st.text_input(
                    "🔖 Tags (comma-separated)",
                    placeholder="e.g. climate, policy, UN"
                )
            
            submitted = st.form_submit_button(
                "💾 Save Article",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                if not add_title or not add_title.strip():
                    st.error("⚠️ Title is required!")
                elif not add_content or not add_content.strip():
                    st.error("⚠️ Content is required!")
                elif len(add_content.split()) < 10:
                    st.error("⚠️ Content must be at least 10 words.")
                else:
                    success, msg = db.add_article(
                        add_title.strip(),
                        add_content.strip(),
                        add_category,
                        add_language,
                        add_tags.strip()
                    )
                    if success:
                        st.success(f"✅ {msg}")
                        st.balloons()
                    else:
                        st.error(f"❌ {msg}")
    
    
    with db_tab2:
        st.markdown("### 📋 Browse Articles")
        
        search_col1, search_col2, search_col3 = st.columns([2, 1, 1])
        
        with search_col1:
            search_query = st.text_input(
                "🔍 Search",
                placeholder="Search by title, content, or tags...",
                key="db_search"
            )
        
        with search_col2:
            filter_category = st.selectbox(
                "🏷️ Filter Category",
                ["All", "World", "Sports", "Business", "Sci-Fi",
                 "Technology", "Health", "Entertainment",
                 "Politics", "Other", "Uncategorized"],
                key="db_filter_cat"
            )
        
        with search_col3:
            filter_language = st.selectbox(
                "🌍 Filter Language",
                ["All", "English", "Hindi", "German",
                 "Spanish", "French", "Other"],
                key="db_filter_lang"
            )
        
        
        if search_query:
            articles = db.search(search_query)
        elif filter_category != "All":
            articles = db.get_by_category(filter_category)
        elif filter_language != "All":
            articles = db.get_by_language(filter_language)
        else:
            articles = db.get_all()
        
        
        if filter_category != "All" and not search_query:
            articles = [
                a for a in articles 
                if a['category'] == filter_category
            ]
        if filter_language != "All" and not search_query:
            articles = [
                a for a in articles 
                if a['language'] == filter_language
            ]
        
        st.markdown(f"**Showing {len(articles)} article(s)**")
        st.markdown("---")
        
        if not articles:
            st.info(
                "📭 No articles found. Add some in the "
                "'➕ Add Article' tab!"
            )
        else:
            for article in articles:
                cat_colors = {
                    "World": "#3498db", "Sports": "#e74c3c",
                    "Business": "#f39c12", "Sci-Fi": "#9b59b6",
                    "Technology": "#1abc9c", "Health": "#2ecc71",
                    "Entertainment": "#e67e22", "Politics": "#34495e",
                    "Other": "#7f8c8d", "Uncategorized": "#95a5a6"
                }
                color = cat_colors.get(article['category'], '#95a5a6')
                
                
                result_summary = db.get_article_result_summary(article['id'])
                result_icons = ""
                if 'classification' in result_summary:
                    result_icons += "📊"
                if 'summarization' in result_summary:
                    result_icons += "📝"
                if 'translation' in result_summary:
                    result_icons += "🌐"
                
                with st.expander(
                    f"📰 **{article['title'][:80]}** "
                    f"| {article['category']} "
                    f"| {article['word_count']}w "
                    f"| {article['language']} "
                    f"{result_icons}"
                ):
                    st.markdown(
                        f"<span style='background:{color}; color:white; "
                        f"padding:3px 12px; border-radius:12px; "
                        f"font-size:13px;'>{article['category']}</span>"
                        f"&nbsp;&nbsp;"
                        f"<span style='color:#95a5a6; font-size:13px;'>"
                        f"Added: {article['created_at']}</span>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(f"**Language:** {article['language']}")
                    
                    if article['tags']:
                        tags_html = "".join(
                            [f"<span class='keyword-tag'>{t.strip()}</span>"
                             for t in article['tags'].split(',') 
                             if t.strip()]
                        )
                        st.markdown(tags_html, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                   
                    preview = article['content'][:500]
                    if len(article['content']) > 500:
                        preview += "..."
                    st.markdown(
                        f"<div style='color:#e0e0e0;'>{preview}</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        f"**📊 Stats:** {article['word_count']} words | "
                        f"{len(article['content'])} characters | "
                        f"ID: {article['id']}"
                    )
                    
                   
                    if result_summary:
                        st.markdown("---")
                        st.markdown("**📊 Processed Results:**")
                        for task, info in result_summary.items():
                            task_emoji = {
                                'classification': '📊',
                                'summarization': '📝',
                                'translation': '🌐'
                            }.get(task, '📄')
                            st.markdown(
                                f"  {task_emoji} **{task.title()}** — "
                                f"Run {info['count']} time(s), "
                                f"latest: {info['latest']}"
                            )
    
   
    with db_tab3:
        st.markdown("### 📊 All Stored Results")
        
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        
        with stat_col1:
            total = db.get_result_count()
            st.metric("📊 Total Results", total)
        
        with stat_col2:
            cls_count = db.get_result_count_by_task('classification')
            st.metric("🏷️ Classifications", cls_count)
        
        with stat_col3:
            sum_count = db.get_result_count_by_task('summarization')
            st.metric("📝 Summaries", sum_count)
        
        with stat_col4:
            trans_count = db.get_result_count_by_task('translation')
            st.metric("🌐 Translations", trans_count)
        
        st.markdown("---")
        
        
        result_filter = st.selectbox(
            "🔍 Filter by Task:",
            ["All Results", "Classification", 
             "Summarization", "Translation"],
            key="result_filter"
        )
        
        
        if result_filter == "All Results":
            all_results = db.get_all_results()
        else:
            task_map = {
                "Classification": "classification",
                "Summarization": "summarization",
                "Translation": "translation"
            }
            all_results = db.get_results_by_task(
                task_map[result_filter]
            )
        
        if not all_results:
            st.info(
                "📭 No results yet. Run Classification, Summarization, "
                "or Translation on database articles to see results here!"
            )
        else:
            st.markdown(f"**Showing {len(all_results)} result(s)**")
            
            for result in all_results:
                task_config = {
                    'classification': {
                        'emoji': '📊',
                        'color': '#3498db',
                        'label': 'Classification'
                    },
                    'summarization': {
                        'emoji': '📝',
                        'color': '#27ae60',
                        'label': 'Summarization'
                    },
                    'translation': {
                        'emoji': '🌐',
                        'color': '#9b59b6',
                        'label': 'Translation'
                    }
                }
                
                config = task_config.get(
                    result['task_type'],
                    {'emoji': '📄', 'color': '#95a5a6', 'label': 'Unknown'}
                )
                
                header = (
                    f"{config['emoji']} {config['label']} | "
                    f"📰 {result['article_title'][:50]} | "
                    f"{result['created_at']}"
                )
                
                with st.expander(header):
                    st.markdown(
                        f"<span style='background:{config['color']}; "
                        f"color:white; padding:4px 15px; "
                        f"border-radius:15px; font-size:14px; "
                        f"font-weight:bold;'>"
                        f"{config['emoji']} {config['label']}</span>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown(
                        f"**📰 Source Article:** "
                        f"{result['article_title']}"
                    )
                    st.markdown(
                        f"**📅 Date:** {result['created_at']}"
                    )
                    
                    st.markdown("---")
                    
                    
                    if result['task_type'] == 'classification':
                        cat_colors = {
                            "World": "#3498db",
                            "Sports": "#e74c3c",
                            "Business": "#f39c12",
                            "Sci-Fi": "#9b59b6"
                        }
                        
                        cat_color = cat_colors.get(
                            result['result_data'], '#95a5a6'
                        )
                        
                        st.markdown(f"""
                            <div style='text-align:center; padding:20px; 
                            background:linear-gradient(135deg, 
                            {cat_color}33, {cat_color}66); 
                            border-radius:15px; 
                            border:2px solid {cat_color};'>
                                <h2 style='color:{cat_color} !important; 
                                margin:0;'>
                                    {result['result_data'].upper()}
                                </h2>
                                <p style='color:#e0e0e0; margin:5px 0 0 0;'>
                                    Confidence: 
                                    <strong>
                                        {result['confidence']:.2f}%
                                    </strong>
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        meta = result.get('metadata', {})
                        all_scores = meta.get('all_scores', {})
                        
                        if all_scores:
                            st.markdown("**All Category Scores:**")
                            score_cols = st.columns(4)
                            for idx, (cat, score) in enumerate(
                                all_scores.items()
                            ):
                                with score_cols[idx % 4]:
                                    c = cat_colors.get(cat, '#95a5a6')
                                    st.markdown(f"""
                                        <div class='stats-box'>
                                            <strong style='color:{c};'>
                                                {cat}
                                            </strong><br>
                                            <span style='font-size:20px;'>
                                                {score:.1f}%
                                            </span>
                                        </div>
                                    """, unsafe_allow_html=True)
                        
                        kw = meta.get('keywords', [])
                        if kw:
                            kw_html = "".join(
                                [f"<span class='keyword-tag'>{k}</span>"
                                 for k in kw]
                            )
                            st.markdown(
                                f"<div style='margin:10px 0;'>"
                                f"{kw_html}</div>",
                                unsafe_allow_html=True
                            )
                    
                 
                    elif result['task_type'] == 'summarization':
                        st.markdown("**Generated Summary:**")
                        st.markdown(f"""
                            <div style='background:rgba(39,174,96,0.1); 
                            padding:20px; border-radius:10px; 
                            border-left:4px solid #27ae60; 
                            color:#e0e0e0;'>
                                {result['result_data']}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        meta = result.get('metadata', {})
                        
                        m_col1, m_col2, m_col3 = st.columns(3)
                        with m_col1:
                            st.metric(
                                "Original Words",
                                meta.get('original_words', 'N/A')
                            )
                        with m_col2:
                            st.metric(
                                "Summary Words",
                                meta.get('summary_words', 'N/A')
                            )
                        with m_col3:
                            st.metric(
                                "Reduction",
                                f"{meta.get('reduction_pct', 0):.1f}%"
                            )
                        
                        st.code(result['result_data'], language=None)
                    
                    
                    elif result['task_type'] == 'translation':
                        meta = result.get('metadata', {})
                        src_lang = meta.get(
                            'source_language', 'Unknown'
                        )
                        
                        st.markdown(
                            f"**🌍 {src_lang} → 🇬🇧 English**"
                        )
                        
                        st.markdown("**Translation:**")
                        st.markdown(f"""
                            <div style='background:rgba(155,89,182,0.1); 
                            padding:20px; border-radius:10px; 
                            border-left:4px solid #9b59b6; 
                            color:#e0e0e0;'>
                                {result['result_data']}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        t_col1, t_col2 = st.columns(2)
                        with t_col1:
                            st.metric(
                                "Source Words",
                                meta.get('source_words', 'N/A')
                            )
                        with t_col2:
                            st.metric(
                                "Translated Words",
                                meta.get('translated_words', 'N/A')
                            )
                        
                        st.code(result['result_data'], language=None)
                    
                    
                    st.markdown("---")
                    
                    if st.button(
                        "🗑️ Delete This Result",
                        key=f"del_result_{result['id']}",
                        type="secondary"
                    ):
                        db.delete_result(result['id'])
                        st.success("✅ Result deleted!")
                        st.rerun()
            
            
            st.markdown("---")
            
            export_col, clear_col = st.columns(2)
            
            with export_col:
                st.download_button(
                    label="📊 Export All Results (CSV)",
                    data=db.export_results_csv(),
                    file_name=(
                        f"results_"
                        f"{datetime.now().strftime('%Y%m%d')}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                    key="export_results_csv"
                )
            
            with clear_col:
                confirm_clear = st.checkbox(
                    "I want to delete all results",
                    key="confirm_clear_results"
                )
                
                if st.button(
                    "🗑️ Clear All Results",
                    disabled=not confirm_clear,
                    use_container_width=True,
                    key="clear_all_results"
                ):
                    db.clear_results()
                    st.success("✅ All results cleared!")
                    st.rerun()
    
    
    with db_tab4:
        st.markdown("### ✏️ Edit or Delete Articles")
        
        all_articles = db.get_all()
        
        if not all_articles:
            st.info("📭 No articles to edit.")
        else:
            article_options = {
                f"[ID {a['id']}] {a['title'][:70]}": a['id']
                for a in all_articles
            }
            
            selected_label = st.selectbox(
                "Select article to edit/delete:",
                list(article_options.keys()),
                key="edit_select"
            )
            
            selected_id = article_options[selected_label]
            article = db.get_by_id(selected_id)
            
            if article:
                st.markdown("---")
                
                edit_col1, edit_col2 = st.columns([3, 1])
                
                with edit_col1:
                    with st.form("edit_article_form"):
                        
                        edit_title = st.text_input(
                            "📌 Title",
                            value=article['title'],
                            key="edit_title"
                        )
                        
                        edit_content = st.text_area(
                            "📄 Content",
                            value=article['content'],
                            height=250,
                            key="edit_content"
                        )
                        
                        e_col1, e_col2, e_col3 = st.columns(3)
                        
                        categories = [
                            "Uncategorized", "World", "Sports",
                            "Business", "Sci-Fi", "Technology",
                            "Health", "Entertainment", 
                            "Politics", "Other"
                        ]
                        
                        with e_col1:
                            cat_idx = (
                                categories.index(article['category'])
                                if article['category'] in categories
                                else 0
                            )
                            edit_category = st.selectbox(
                                "🏷️ Category",
                                categories,
                                index=cat_idx,
                                key="edit_cat"
                            )
                        
                        languages = [
                            "English", "Hindi", "German",
                            "Spanish", "French", "Other"
                        ]
                        
                        with e_col2:
                            lang_idx = (
                                languages.index(article['language'])
                                if article['language'] in languages
                                else 0
                            )
                            edit_language = st.selectbox(
                                "🌍 Language",
                                languages,
                                index=lang_idx,
                                key="edit_lang"
                            )
                        
                        with e_col3:
                            edit_tags = st.text_input(
                                "🔖 Tags",
                                value=article['tags'],
                                key="edit_tags"
                            )
                        
                        if st.form_submit_button(
                            "💾 Save Changes",
                            use_container_width=True,
                            type="primary"
                        ):
                            if (edit_title.strip() 
                                    and edit_content.strip()):
                                db.update_article(
                                    selected_id,
                                    edit_title.strip(),
                                    edit_content.strip(),
                                    edit_category,
                                    edit_language,
                                    edit_tags.strip()
                                )
                                st.success("✅ Article updated!")
                                st.rerun()
                            else:
                                st.error(
                                    "⚠️ Title and content required!"
                                )
                
                with edit_col2:
                    st.markdown("### ⚠️ Danger Zone")
                    
                    
                    result_summary = db.get_article_result_summary(
                        selected_id
                    )
                    result_count = sum(
                        v['count'] for v in result_summary.values()
                    )
                    
                    st.markdown(
                        f"<div class='stats-box'>"
                        f"<strong>ID:</strong> {article['id']}<br>"
                        f"<strong>Words:</strong> "
                        f"{article['word_count']}<br>"
                        f"<strong>Results:</strong> "
                        f"{result_count}<br>"
                        f"<strong>Added:</strong><br>"
                        f"{article['created_at']}"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("---")
                    
                    confirm = st.checkbox(
                        "I confirm deletion",
                        key="confirm_delete"
                    )
                    
                    if st.button(
                        "🗑️ Delete Article",
                        use_container_width=True,
                        disabled=not confirm,
                        key="delete_btn"
                    ):
                        db.delete_article(selected_id)
                        st.success(
                            "✅ Article & all its results deleted!"
                        )
                        st.rerun()
                    
                    st.markdown("---")
                    
                    confirm_all = st.checkbox(
                        "I want to delete ALL",
                        key="confirm_clear_all"
                    )
                    
                    if st.button(
                        "💣 Clear Entire Database",
                        use_container_width=True,
                        disabled=not confirm_all,
                        key="clear_all_btn"
                    ):
                        db.clear_all()
                        st.success("✅ All articles & results deleted!")
                        st.rerun()
    
   
    with db_tab5:
        st.markdown("### 📤 Import & Export")
        
        imp_col, exp_col = st.columns(2)
        
        with exp_col:
            st.markdown("#### 📥 Export Database")
            
            if db.get_count() > 0:
                csv_export = db.export_csv()
                
                st.download_button(
                    label=(
                        f"📊 Download Articles CSV "
                        f"({db.get_count()} articles)"
                    ),
                    data=csv_export,
                    file_name=(
                        f"news_db_"
                        f"{datetime.now().strftime('%Y%m%d')}.csv"
                    ),
                    mime="text/csv",
                    use_container_width=True,
                    key="export_articles_csv"
                )
                
              
                result_count = db.get_result_count()
                if result_count > 0:
                    st.download_button(
                        label=(
                            f"📊 Download Results CSV "
                            f"({result_count} results)"
                        ),
                        data=db.export_results_csv(),
                        file_name=(
                            f"news_results_"
                            f"{datetime.now().strftime('%Y%m%d')}.csv"
                        ),
                        mime="text/csv",
                        use_container_width=True,
                        key="export_results_csv_tab"
                    )
                
                st.info(
                    "CSV includes: title, content, category, "
                    "language, tags, word_count, timestamps"
                )
            else:
                st.info("📭 No articles to export.")
        
        with imp_col:
            st.markdown("#### 📤 Import from CSV")
            
            st.info(
                "CSV must have columns: **title**, **content**\n\n"
                "Optional: category, language, tags"
            )
            
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=['csv'],
                key="csv_upload"
            )
            
            if uploaded_file:
                if st.button(
                    "📥 Import Articles",
                    use_container_width=True,
                    type="primary",
                    key="import_btn"
                ):
                    csv_text = uploaded_file.getvalue().decode('utf-8')
                    added = db.import_from_csv(csv_text)
                    
                    if added > 0:
                        st.success(f"✅ Imported {added} articles!")
                        st.rerun()
                    elif added == 0:
                        st.warning(
                            "⚠️ No new articles imported. "
                            "Possibly duplicates or invalid data."
                        )
                    else:
                        st.error(
                            "❌ Import failed. Check CSV format."
                        )

with tab5:
    st.markdown('<div class="section-header">📜 History & Downloads</div>', unsafe_allow_html=True)
    
    history_tab1, history_tab2, history_tab3 = st.tabs([
        "📊 Classification",
        "📝 Summarization",
        "🌐 Translation"
    ])
    
  
    with history_tab1:
        st.markdown("### Recent Classifications (Last 5)")
        
        if len(st.session_state.classification_history) == 0:
            st.info("📭 No history yet.")
        else:
            for idx, record in enumerate(st.session_state.classification_history):
                with st.expander(f"{record['category']} - {record['timestamp']}", expanded=(idx==0)):
                    st.markdown(f"**Title:** {record['title']}")
                    st.markdown(f"**Category:** {record['category']}")
                    st.markdown(f"**Confidence:** {record['confidence']:.2f}%")
                    st.markdown(f"**Keywords:** {record['keywords']}")
            
            st.markdown("---")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                history_df = pd.DataFrame(st.session_state.classification_history)
                st.download_button(
                    label="📊 Download All (CSV)",
                    data=history_df.to_csv(index=False),
                    file_name=f"classification_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_col2:
                if st.button("🗑️ Clear History", key="clear_class", use_container_width=True):
                    st.session_state.classification_history = []
                    st.rerun()
    
    
    with history_tab2:
        st.markdown("### Recent Summaries (Last 5)")
        
        if len(st.session_state.summarization_history) == 0:
            st.info("📭 No history yet.")
        else:
            for idx, record in enumerate(st.session_state.summarization_history):
                with st.expander(f"Summary {idx+1} - {record['timestamp']}", expanded=(idx==0)):
                    st.markdown(f"**Preview:** {record['summary']}")
                    st.markdown(f"**Original Words:** {record['original_words']}")
                    st.markdown(f"**Summary Words:** {record['summary_words']}")
                    st.markdown(f"**Reduction:** {record['reduction']:.1f}%")
                    st.markdown(f"**Keywords:** {record['keywords']}")
            
            st.markdown("---")
            
            download_col1, download_col2 = st.columns(2)
            
            with download_col1:
                history_df = pd.DataFrame(st.session_state.summarization_history)
                st.download_button(
                    label="📊 Download All (CSV)",
                    data=history_df.to_csv(index=False),
                    file_name=f"summarization_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with download_col2:
                if st.button("🗑️ Clear History", key="clear_summ", use_container_width=True):
                    st.session_state.summarization_history = []
                    st.rerun()
    
    
with history_tab3:
    st.markdown("### Recent Translations (Last 5)")
    
    if len(st.session_state.translation_history) == 0:
        st.info("📭 No history yet.")
    else:
        for idx, record in enumerate(st.session_state.translation_history):
           
            lang_emoji = {
                'Hindi': '🇮🇳',
                'German': '🇩🇪',
                'Spanish': '🇪🇸',
                'French': '🇫🇷'
            }.get(record.get('source_language', 'Unknown'), '🌍')
            
            with st.expander(f"{lang_emoji} {record.get('source_language', 'Unknown')} → 🇬🇧 English - {record['timestamp']}", expanded=(idx==0)):
                st.markdown(f"**Source Language:** {record.get('source_language', 'Unknown')}")
                st.markdown(f"**Source Text:** {record.get('source_text', record.get('hindi', 'N/A'))}")
                st.markdown(f"**English:** {record['english']}")
                st.markdown(f"**Source Words:** {record.get('source_words', record.get('hindi_words', 'N/A'))}")
                st.markdown(f"**English Words:** {record.get('english_words', 'N/A')}")
                st.markdown(f"**Keywords:** {record['keywords']}")
        
        st.markdown("---")
        
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            history_df = pd.DataFrame(st.session_state.translation_history)
            st.download_button(
                label="📊 Download All (CSV)",
                data=history_df.to_csv(index=False),
                file_name=f"translation_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with download_col2:
            if st.button("🗑️ Clear History", key="clear_trans", use_container_width=True):
                st.session_state.translation_history = []
                st.rerun()


with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='font-size: 80px;'>📰</div>
            <h2 style='color: #3498db;'>News AI System</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🎯 Features")
    st.markdown("""
        <div class='info-card'>
            <ul style='color: #e0e0e0; font-size: 14px;'>
                <li>📊 <strong>Classification</strong><br>4 categories + confidence</li>
                <li>📝 <strong>Summarization</strong><br>56-142 word summaries</li>
                <li>🌐 <strong>Translation</strong><br>Hindi → English</li>
                <li>🔑 <strong>Keywords</strong><br>Auto extraction</li>
                <li>📥 <strong>Downloads</strong><br>PDF & CSV</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    st.metric("Classifications", len(st.session_state.classification_history))
    st.metric("Summaries", len(st.session_state.summarization_history))
    st.metric("Translations", len(st.session_state.translation_history))
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips")
    st.info("""
        ✓ Clear, well-written text
        
        ✓ Check word requirements
        
        ✓ Review confidence scores
        
        ✓ Download results
    """)
    
    st.markdown("---")
    
    st.markdown("""
        <div style='text-align: center; color: #e0e0e0;'>
            <p>📰 News AI System</p>
            <p>Multi-Task NLP</p>
            <p>© 2024</p>
        </div>
    """, unsafe_allow_html=True)



st.markdown("---")
st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
    border-radius: 15px; color: white; margin-top: 30px; box-shadow: 0 4px 15px rgba(52, 152, 219, 0.4);'>
        <h3 style='color: white !important;'>🌟 Advanced NLP for News Processing</h3>
        <p style='color: white !important;'>Classification • Summarization • Translation</p>
        <p style='color: white !important;'>Made with ❤️ using Streamlit & Transformers</p>
    </div>
""", unsafe_allow_html=True)