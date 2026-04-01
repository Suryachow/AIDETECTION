"""
Lexical Agent Module
Processes sentences and extracts lexical information for each word.
"""

import spacy
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Any, Set
import re
import string
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexicalAgent:
    CEFR_THRESHOLDS = {
        'A1': 0.85, 'A2': 0.70, 'B1': 0.50, 'B2': 0.30, 'C1': 0.15, 'C2': 0.0
    }
    
    HIGH_FREQ_WORDS = {
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
        'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
        'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
        'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
        'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
        'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
    }
    
    POS_MAP = {
        'NOUN': 'noun', 'VERB': 'verb', 'ADJ': 'adjective', 'ADV': 'adverb',
        'PRON': 'pronoun', 'DET': 'determiner', 'ADP': 'preposition', 'NUM': 'number',
        'CONJ': 'conjunction', 'CCONJ': 'conjunction', 'SCONJ': 'conjunction',
        'PART': 'particle', 'INTJ': 'interjection', 'PROPN': 'proper noun', 'AUX': 'auxiliary'
    }
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self._download_nltk_data()
        self.WORDNET_POS_MAP = {
            'NOUN': wordnet.NOUN, 'VERB': wordnet.VERB, 'ADJ': wordnet.ADJ, 'ADV': wordnet.ADV
        }
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            import spacy.cli
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)
    
    def _download_nltk_data(self):
        required_data = ['wordnet', 'omw-1.4', 'punkt']
        for data in required_data:
            try:
                nltk.data.find(f'corpora/{data}')
            except LookupError:
                nltk.download(data, quiet=True)
    
    def _clean_word(self, word: str) -> str:
        cleaned = word.lower().strip(string.punctuation)
        cleaned = re.sub(r'[^a-z\-]', '', cleaned)
        return cleaned
    
    def _is_valid_word(self, word: str) -> bool:
        if not word or len(word) < 2: return False
        if word.isdigit(): return False
        return bool(re.match(r'^[a-z\-]+$', word))
    
    def _get_wordnet_pos(self, spacy_pos: str) -> str:
        return self.WORDNET_POS_MAP.get(spacy_pos, wordnet.NOUN)
    
    def _get_synonyms_antonyms(self, word: str, pos: str) -> tuple:
        synonyms, antonyms = set(), set()
        wordnet_pos = self._get_wordnet_pos(pos)
        try:
            synsets = wordnet.synsets(word, pos=wordnet_pos)
            for synset in synsets[:3]:
                for lemma in synset.lemmas():
                    syn = lemma.name().replace('_', ' ')
                    if syn.lower() != word.lower(): synonyms.add(syn.lower())
                for lemma in synset.lemmas():
                    if lemma.antonyms():
                        for ant in lemma.antonyms(): antonyms.add(ant.name().replace('_', ' ').lower())
        except: pass
        return sorted(list(synonyms))[:10], sorted(list(antonyms))[:5]

    def _calculate_frequency(self, word: str) -> float:
        score = 0.5
        if word in self.HIGH_FREQ_WORDS: score = random.uniform(0.80, 0.95)
        else:
            length = len(word)
            if length <= 3: score = random.uniform(0.70, 0.85)
            elif length <= 5: score = random.uniform(0.55, 0.75)
            elif length <= 7: score = random.uniform(0.40, 0.60)
            else: score = random.uniform(0.10, 0.40)
        return round(min(1.0, max(0.0, score)), 2)
    
    def _assign_cefr_level(self, frequency: float) -> str:
        if frequency >= 0.85: return 'A1'
        elif frequency >= 0.70: return 'A2'
        elif frequency >= 0.50: return 'B1'
        elif frequency >= 0.30: return 'B2'
        elif frequency >= 0.15: return 'C1'
        return 'C2'
    
    def process_word(self, word: str, pos: str) -> Dict[str, Any]:
        simple_pos = self.POS_MAP.get(pos, 'other')
        synonyms, antonyms = self._get_synonyms_antonyms(word, pos)
        frequency = self._calculate_frequency(word)
        level = self._assign_cefr_level(frequency)
        return {
            "word": word, "pos": [simple_pos], "synonyms": synonyms,
            "antonyms": antonyms, "level": level, "frequency": frequency
        }
    
    def process_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        doc = self.nlp(sentence)
        processed_words = set()
        word_entries = []
        for token in doc:
            word = self._clean_word(token.text)
            if not self._is_valid_word(word) or word in processed_words: continue
            processed_words.add(word)
            try:
                entry = self.process_word(word, token.pos_)
                word_entries.append(entry)
            except: continue
        return word_entries
