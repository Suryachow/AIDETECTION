"""
Core agent logic for the AI Text Humanizer.
Enhanced version with robust bypass techniques and grammar correction.
"""
import time
import logging
import random
import re
import string
import math
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import unicodedata
import html
import nltk
from collections import Counter

# Import local modules
from app.core.rag import RAGSystem
from app.core.reward import RewardModel
from app.utils import preprocess_text, postprocess_text, calculate_humanness_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import advanced NLP libraries
try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    logger.warning("SpellChecker not available. Install with: pip install pyspellchecker")

try:
    import language_tool_python
    LANGUAGE_TOOL_AVAILABLE = True
except ImportError:
    LANGUAGE_TOOL_AVAILABLE = False
    logger.warning("LanguageTool not available. Install with: pip install language-tool-python")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    logger.warning("TextBlob not available. Install with: pip install textblob")

try:
    import gensim
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    logger.warning("Gensim not available. Install with: pip install gensim")

try:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    NLTK_AVAILABLE = True
    # Download necessary NLTK data
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available or data missing. Install with: pip install nltk")

# Define constants for regex patterns
ENDING_PUNCTUATION_REGEX = r'[.!?]$'
SENTENCE_SPLIT_REGEX = r'(?<=[.!?])\s+'

# Define a constant for sentence splitting regex
SENTENCE_SPLIT_PATTERN = SENTENCE_SPLIT_REGEX

class HumanizerAgent:
    """
    Agent that coordinates the text humanization process.
    Enhanced version with robust bypass techniques and grammar correction.
    """
    
    # Define a constant for the regex pattern
    ENDING_PUNCTUATION_REGEX = r'[.!?]$'
    
    def __init__(self, models: Dict[str, Any], rag_system: RAGSystem):
        """
        Initialize the humanizer agent.
        
        Args:
            models: Dictionary of loaded models
            rag_system: RAG system for retrieving relevant examples
        """
        self.models = models
        self.rag_system = rag_system
        self.reward_model = RewardModel()
        self.llm_type = "enhanced-bypass-humanizer"
        
        # Load language patterns, templates, and human writing samples
        self._load_language_patterns()
        self._load_human_writing_patterns()
        self._load_bypass_techniques()
        
        # Initialize NLP tools if available
        if SPELLCHECKER_AVAILABLE:
            self.spell = SpellChecker()
        else:
            self.spell = None
            
        if LANGUAGE_TOOL_AVAILABLE:
            self.language_tool = language_tool_python.LanguageTool('en-US')
        else:
            self.language_tool = None
            
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            self.stopwords = set(stopwords.words('english'))
        else:
            self.lemmatizer = None
            self.stopwords = set()
            
        logger.info(f"Humanizer agent initialized with type: {self.llm_type}")
    
    def _load_language_patterns(self):
        """Load language patterns and templates for text transformation."""
        # Load synonyms for common words (extensive dictionary)
        self._load_synonyms()
        
        # Load sentence structure transformations
        self._load_sentence_transformations()
        
        # Load paragraph structure transformations
        self._load_paragraph_transformations()
        
        # Load transition phrases
        self._load_transitions()
        
        # Load filler words and phrases by style
        self._load_fillers()
        
        # Load human-specific writing quirks
        self._load_writing_quirks()
    
    def _load_bypass_techniques(self):
        """Load techniques for bypassing AI detection."""
        # Character substitution mapping (visually similar but different Unicode)
        self.char_substitutions = {
            'a': ['а', 'ɑ', 'α'],  # Cyrillic 'а', Latin 'ɑ', Greek 'α'
            'e': ['е', 'ε', 'ℯ'],  # Cyrillic 'е', Greek 'ε', script 'ℯ'
            'o': ['о', 'ο', 'ⲟ'],  # Cyrillic 'о', Greek 'ο', Coptic 'ⲟ'
            'i': ['і', 'ι', 'ⅰ'],  # Ukrainian 'і', Greek 'ι', Roman numeral 'ⅰ'
            'c': ['с', 'ϲ', 'ⅽ'],  # Cyrillic 'с', Greek lunate sigma 'ϲ', Roman numeral 'ⅽ'
            'p': ['р', 'ρ', 'ⲣ'],  # Cyrillic 'р', Greek 'ρ', Coptic 'ⲣ'
            'y': ['у', 'γ', 'ỿ'],  # Cyrillic 'у', Greek 'γ', Latin 'ỿ'
            'x': ['х', 'χ', 'ⅹ'],  # Cyrillic 'х', Greek 'χ', Roman numeral 'ⅹ'
            'j': ['ј', 'ϳ'],       # Cyrillic 'ј', Greek 'ϳ'
            'h': ['һ', 'ℎ'],       # Cyrillic 'һ', script 'ℎ'
            'n': ['ո', 'ⲛ'],       # Armenian 'ո', Coptic 'ⲛ'
            's': ['ѕ', 'ꜱ'],       # Cyrillic 'ѕ', Latin small capital 'ꜱ'
        }
        
        # Homoglyphs (characters that look similar)
        self.homoglyphs = {
            'a': ['а', 'ɑ', 'α'],
            'b': ['Ь', 'ƅ', 'Ꮟ'],
            'c': ['с', 'ϲ', 'ⅽ'],
            'd': ['ԁ', 'ⅾ', 'ꓒ'],
            'e': ['е', 'ε', 'ℯ'],
            'f': ['ſ', 'ք', 'ẝ'],
            'g': ['ɡ', 'ց', 'ǵ'],
            'h': ['һ', 'հ', 'ℎ'],
            'i': ['і', 'ⅰ', 'ℹ'],
            'j': ['ϳ', 'ј', 'ⅉ'],
            'k': ['κ', 'ⲕ', 'ḳ'],
            'l': ['І', 'ⅼ', 'ℓ'],
            'm': ['ⅿ', 'ṃ', 'ṁ'],
            'n': ['ո', 'ռ', 'ⁿ'],
            'o': ['о', 'ο', 'ℴ'],
            'p': ['р', 'ρ', 'ṗ'],
            'q': ['ԛ', 'զ', 'ɋ'],
            'r': ['г', 'ⲅ', 'ṛ'],
            's': ['ѕ', 'ꜱ', 'ṣ'],
            't': ['τ', 'ṭ', 'ṫ'],
            'u': ['ս', 'υ', 'ṳ'],
            'v': ['ѵ', 'ν', 'ṿ'],
            'w': ['ԝ', 'ѡ', 'ẇ'],
            'x': ['х', 'ⅹ', 'ẋ'],
            'y': ['у', 'γ', 'ẏ'],
            'z': ['ᴢ', 'ꮓ', 'ẓ'],
        }
        
        # Subtle spacing variations
        self.spacing_variations = [
            '\u200A',  # Hair space
            '\u2009',  # Thin space
            '\u202F',  # Narrow no-break space
            '\u2006',  # Six-per-em space
            '\u2005',  # Four-per-em space
            '\u2004',  # Three-per-em space
        ]
        
        # Punctuation variations
        self.punctuation_variations = {
            '.': ['．', '｡', '․'],
            ',': ['，', '､', '‚'],
            '!': ['！', '﹗', '‼'],
            '?': ['？', '﹖', '⁇'],
            '-': ['－', '‐', '‑'],
            ':': ['：', '︰', '꞉'],
            ';': ['；', '︔', '⁏'],
            '"': ['＂', '"', '"'],
            "'": ['＇', ''', '''],
            '(': ['（', '❨', '﹙'],
            ')': ['）', '❩', '﹚'],
            '[': ['［', '❲', '﹂'],
            ']': ['］', '❳', '﹃'],
            '{': ['｛', '❴', '﹛'],
            '}': ['｝', '❵', '﹜'],
            '/': ['／', '⁄', '∕'],
            '\\': ['＼', '⧵', '⧹'],
            '|': ['｜', '❘', '∣'],
            '*': ['＊', '⋆', '∗'],
            '&': ['＆', '﹠', '﹅'],
            '#': ['＃', '﹟', '♯'],
            '%': ['％', '﹪', '⁒'],
            '@': ['＠', '﹫', '⧄'],
            '$': ['＄', '﹩', '﹊'],
            '^': ['＾', '﹀', '˄'],
            '+': ['＋', '﹢', '➕'],
            '=': ['＝', '﹦', '⩵'],
            '<': ['＜', '﹤', '❮'],
            '>': ['＞', '﹥', '❯'],
            '~': ['～', '⁓', '∼'],
            '`': ['｀', '❛', '‵'],
        }
        
        # Word structure variations
        self.word_structure_variations = [
            self._add_suffix,
            self._change_tense,
            self._pluralize,
            self._singularize,
            self._add_prefix,
            self._change_comparative,
            self._change_superlative,
        ]
        
        # Sentence structure variations
        self.sentence_structure_variations = [
            self._reorder_clauses,
            self._change_voice,
            self._split_sentence,
            self._combine_sentences,
            self._add_subordinate_clause,
            self._change_question_to_statement,
            self._change_statement_to_question,
        ]
    
    def _load_synonyms(self):
        """Load extensive synonym dictionary."""
        # This is a much more extensive synonym dictionary than before
        self.synonyms = {
            # Nouns
            "person": ["individual", "human", "being", "soul", "character", "figure", "mortal", "somebody", "fellow", "citizen"],
            "time": ["period", "era", "age", "epoch", "moment", "instance", "occasion", "duration", "stretch", "span", "interval"],
            "year": ["annum", "twelve months", "calendar year", "fiscal period", "solar cycle", "season", "term", "stint", "spell"],
            "way": ["method", "manner", "approach", "technique", "strategy", "procedure", "path", "route", "course", "direction", "means"],
            "day": ["date", "period", "time", "occasion", "moment", "juncture", "interval", "24 hours", "daytime", "daylight hours"],
            "thing": ["item", "object", "article", "entity", "material", "substance", "element", "matter", "stuff", "piece", "bit"],
            "man": ["gentleman", "fellow", "male", "guy", "chap", "individual", "person", "human", "dude", "bloke", "gent"],
            "world": ["earth", "globe", "planet", "realm", "domain", "sphere", "universe", "cosmos", "creation", "society", "civilization"],
            "life": ["existence", "being", "living", "animation", "vitality", "lifetime", "biography", "experience", "journey", "adventure"],
            "hand": ["palm", "grip", "grasp", "extremity", "appendage", "digits", "fingers", "mitt", "paw", "fist", "claw"],
            "part": ["portion", "section", "segment", "component", "element", "piece", "fraction", "division", "chunk", "slice", "bit"],
            "child": ["youngster", "minor", "juvenile", "youth", "adolescent", "teenager", "kid", "offspring", "young person", "little one"],
            "eye": ["optic", "vision", "sight", "view", "gaze", "glance", "peeper", "ocular", "orb", "visual organ"],
            "woman": ["lady", "female", "girl", "gal", "madam", "matron", "lass", "dame", "gentlewoman", "miss", "mrs"],
            "place": ["location", "spot", "site", "position", "point", "locale", "venue", "area", "region", "zone", "locality"],
            "work_task": ["labor", "toil", "employment", "job", "occupation", "profession", "vocation", "career", "business", "trade", "craft"],
            "week": ["seven days", "workweek", "period", "stretch", "interval", "stint", "spell", "cycle", "sequence", "series"],
            "case": ["instance", "example", "occurrence", "situation", "circumstance", "scenario", "event", "affair", "matter", "issue"],
            "point": ["spot", "location", "position", "place", "dot", "mark", "speck", "detail", "aspect", "feature", "characteristic"],
            "government": ["administration", "authority", "regime", "management", "leadership", "control", "command", "rule", "jurisdiction"],
            "company": ["business", "firm", "corporation", "enterprise", "establishment", "organization", "outfit", "concern", "operation"],
            "number": ["figure", "digit", "numeral", "quantity", "amount", "sum", "total", "count", "tally", "statistic", "value"],
            "group": ["collection", "assembly", "gathering", "crowd", "throng", "cluster", "bunch", "set", "array", "assortment", "batch"],
            "problem": ["issue", "matter", "difficulty", "trouble", "complication", "obstacle", "hurdle", "dilemma", "predicament", "challenge"],
            "fact": ["reality", "truth", "actuality", "certainty", "verity", "detail", "information", "particular", "point", "datum"],
            
            # Verbs
            "be": ["exist", "live", "remain", "stay", "continue", "persist", "endure", "prevail", "subsist", "abide", "dwell", "reside"],
            "have": ["possess", "own", "hold", "maintain", "keep", "retain", "contain", "include", "bear", "carry", "harbor", "entertain"],
            "do": ["perform", "execute", "accomplish", "achieve", "complete", "finish", "conclude", "carry out", "fulfill", "discharge", "conduct"],
            "say": ["state", "declare", "announce", "express", "articulate", "voice", "utter", "pronounce", "mention", "remark", "observe"],
            "get": ["obtain", "acquire", "gain", "procure", "secure", "attain", "collect", "gather", "receive", "earn", "win", "achieve"],
            "make": ["create", "produce", "generate", "form", "construct", "build", "develop", "establish", "fashion", "manufacture", "fabricate"],
            "go": ["move", "proceed", "advance", "progress", "travel", "journey", "venture", "head", "depart", "leave", "exit", "withdraw"],
            "know": ["understand", "comprehend", "grasp", "recognize", "perceive", "apprehend", "discern", "realize", "appreciate", "fathom"],
            "take": ["grab", "seize", "grasp", "clutch", "grip", "snatch", "catch", "capture", "acquire", "obtain", "secure", "procure"],
            "see": ["observe", "view", "witness", "behold", "notice", "spot", "perceive", "discern", "distinguish", "make out", "glimpse"],
            "come": ["approach", "advance", "near", "arrive", "reach", "appear", "materialize", "emerge", "loom", "surface", "manifest"],
            "think": ["believe", "consider", "contemplate", "reflect", "ponder", "meditate", "muse", "ruminate", "cogitate", "deliberate"],
            "look": ["gaze", "stare", "glance", "peek", "peer", "glimpse", "view", "observe", "regard", "contemplate", "survey", "scan"],
            "want": ["desire", "wish", "crave", "covet", "yearn", "long", "hanker", "aspire", "hope", "fancy", "prefer", "choose"],
            "give": ["provide", "supply", "furnish", "deliver", "contribute", "donate", "present", "offer", "bestow", "grant", "confer"],
            "use": ["employ", "utilize", "apply", "exercise", "implement", "practice", "operate", "handle", "wield", "manipulate", "work"],
            "find": ["discover", "locate", "uncover", "detect", "spot", "identify", "unearth", "come across", "encounter", "stumble upon"],
            "tell": ["inform", "notify", "advise", "relate", "recount", "narrate", "describe", "explain", "communicate", "report", "reveal"],
            "ask": ["inquire", "query", "question", "interrogate", "probe", "request", "solicit", "seek", "demand", "require", "entreat"],
            "work": ["labor", "toil", "function", "operate", "perform", "serve", "act", "behave", "proceed", "go", "run", "move"],
            "seem": ["appear", "look", "sound", "feel", "give the impression", "come across", "strike one as", "have the appearance of"],
            "feel": ["sense", "experience", "undergo", "perceive", "discern", "be aware of", "be conscious of", "notice", "observe", "detect"],
            "try": ["attempt", "endeavor", "strive", "seek", "undertake", "venture", "essay", "aim", "aspire", "struggle", "labor", "toil"],
            "leave": ["depart", "exit", "withdraw", "retire", "retreat", "quit", "vacate", "abandon", "desert", "forsake", "relinquish"],
            "call": ["name", "term", "designate", "label", "dub", "title", "style", "christen", "denominate", "identify", "specify"],
            
            # Adjectives
            "good": ["excellent", "fine", "superior", "quality", "satisfactory", "valuable", "worthy", "admirable", "commendable", "praiseworthy"],
            "new": ["recent", "fresh", "novel", "modern", "current", "contemporary", "latest", "innovative", "original", "cutting-edge", "state-of-the-art"],
            "first": ["initial", "primary", "earliest", "original", "foremost", "premier", "leading", "principal", "chief", "main", "paramount"],
            "last": ["final", "ultimate", "concluding", "terminal", "closing", "end", "hindmost", "rearmost", "latest", "most recent", "newest"],
            "long": ["extended", "lengthy", "prolonged", "protracted", "extensive", "elongated", "stretched", "expanded", "spread out", "drawn out"],
            "great": ["excellent", "wonderful", "marvelous", "superb", "outstanding", "magnificent", "splendid", "grand", "remarkable", "exceptional"],
            "little": ["small", "tiny", "miniature", "diminutive", "minute", "petite", "slight", "modest", "minor", "insignificant", "negligible"],
            "own": ["personal", "individual", "private", "particular", "special", "distinct", "separate", "unique", "exclusive", "proprietary"],
            "other": ["different", "alternative", "additional", "further", "extra", "supplementary", "another", "second", "distinct", "disparate"],
            "old": ["aged", "elderly", "senior", "ancient", "antique", "vintage", "mature", "experienced", "venerable", "time-worn", "seasoned"],
            "right": ["correct", "accurate", "exact", "precise", "proper", "appropriate", "suitable", "fitting", "apt", "pertinent", "relevant"],
            "big": ["large", "sizable", "substantial", "considerable", "great", "huge", "enormous", "immense", "vast", "gigantic", "colossal"],
            "high": ["tall", "lofty", "elevated", "towering", "soaring", "raised", "steep", "prominent", "eminent", "distinguished", "exalted"],
            "different": ["diverse", "varied", "various", "distinct", "separate", "dissimilar", "unlike", "contrasting", "divergent", "disparate"],
            "small": ["little", "tiny", "miniature", "diminutive", "minute", "petite", "slight", "modest", "minor", "insignificant", "negligible"],
            "large": ["big", "sizable", "substantial", "considerable", "great", "huge", "enormous", "immense", "vast", "gigantic", "colossal"],
            "next": ["subsequent", "following", "succeeding", "ensuing", "coming", "approaching", "impending", "forthcoming", "future", "prospective"],
            "early": ["initial", "first", "primary", "beginning", "preliminary", "opening", "starting", "introductory", "incipient", "nascent"],
            "young": ["youthful", "juvenile", "adolescent", "immature", "budding", "developing", "growing", "emerging", "fledgling", "inexperienced"],
            "important": ["significant", "crucial", "vital", "essential", "critical", "key", "central", "fundamental", "principal", "major", "paramount"],
            "few": ["scant", "limited", "meager", "sparse", "scarce", "scanty", "negligible", "insufficient", "inadequate", "deficient", "minimal"],
            "public": ["communal", "community", "common", "shared", "collective", "general", "universal", "open", "accessible", "available", "unrestricted"],
            "bad": ["poor", "inferior", "substandard", "deficient", "inadequate", "unsatisfactory", "unacceptable", "disappointing", "disagreeable"],
            "same": ["identical", "equivalent", "equal", "matching", "corresponding", "comparable", "similar", "alike", "uniform", "consistent"],
            "able": ["capable", "competent", "qualified", "skilled", "proficient", "adept", "accomplished", "talented", "gifted", "apt", "adroit"],
            
            # Adverbs
            "up": ["upward", "aloft", "overhead", "above", "skyward", "upwards", "upwardly", "upstairs", "uphill", "upland", "uptown"],
            "out": ["outside", "outdoors", "outward", "outwards", "outwardly", "externally", "abroad", "away", "forth", "beyond", "afar"],
            "then": ["at that time", "at that point", "next", "subsequently", "afterward", "thereafter", "later", "following that", "after that"],
            "only": ["merely", "simply", "just", "solely", "exclusively", "purely", "entirely", "wholly", "completely", "totally", "utterly"],
            "now": ["currently", "presently", "at present", "at this moment", "at this time", "nowadays", "today", "these days", "in this day and age"],
            "very": ["extremely", "exceedingly", "exceptionally", "especially", "particularly", "remarkably", "notably", "decidedly", "emphatically"],
            "even": ["still", "yet", "nevertheless", "nonetheless", "however", "though", "although", "despite", "in spite of", "regardless"],
            "also": ["too", "as well", "in addition", "besides", "furthermore", "moreover", "likewise", "similarly", "correspondingly", "equally"],
            "back": ["backward", "behind", "rearward", "in reverse", "in return", "in response", "again", "once more", "anew", "afresh"],
            "there": ["in that place", "at that point", "in that location", "at that location", "yonder", "over there", "in that direction"],
            "down": ["downward", "below", "beneath", "underneath", "under", "downstairs", "downwards", "downwardly", "downhill", "downrange"],
            "still": ["yet", "nevertheless", "nonetheless", "however", "even so", "all the same", "just the same", "notwithstanding", "despite this"],
            "here": ["at this place", "at this point", "at this location", "in this place", "nearby", "close by", "in this vicinity", "in this area", "right here", "at hand"],
            "well": ["skillfully", "proficiently", "competently", "adeptly", "expertly", "masterfully", "adroitly", "dexterously", "capably"],
            "just": ["only", "merely", "simply", "solely", "exclusively", "purely", "entirely", "wholly", "completely", "totally", "utterly"],
            "how": ["in what way", "by what means", "in what manner", "to what extent", "to what degree", "by what method", "in which way"],
            "too": ["also", "as well", "in addition", "besides", "furthermore", "moreover", "likewise", "similarly", "correspondingly", "equally"],
            "more": ["additional", "extra", "further", "added", "supplementary", "increased", "greater", "larger", "bigger", "higher", "enhanced"],
            "never": ["not ever", "at no time", "on no occasion", "under no circumstances", "not at all", "not once", "not in any way", "by no means"],
            "most": ["nearly all", "almost all", "the majority", "the bulk", "the greater part", "predominantly", "mainly", "chiefly", "principally"],
            "again": ["once more", "anew", "afresh", "another time", "yet again", "repeatedly", "over again", "one more time", "a second time"],
            "always": ["constantly", "continually", "continuously", "perpetually", "eternally", "forever", "evermore", "at all times", "without fail"],
            "however": ["nevertheless", "nonetheless", "even so", "still", "yet", "but", "though", "although", "despite this", "in spite of this"],
            "why": ["for what reason", "for what purpose", "for what cause", "on what grounds", "how come", "what for", "wherefore", "to what end"],
            "off": ["away", "aside", "apart", "to one side", "at a distance", "remote", "separate", "detached", "removed", "disconnected"],
        }
    
    def _load_sentence_transformations(self):
        """Load sentence structure transformations."""
        self.sentence_transformations = [
            self._reorder_clauses,
            self._change_voice,
            self._split_sentence,
            self._combine_sentences,
            self._add_subordinate_clause,
            self._change_question_to_statement,
            self._change_statement_to_question,
        ]
    
    def _transform_beginning(self, sentence: str, style: str) -> str:
        """
        Transform the beginning of a sentence to make it more engaging or varied.
        
            self._transform_beginning,  # Ensure this method is defined below
            sentence: The sentence to transform
            style: The style to apply
            
        Returns:
            Transformed sentence with a modified beginning
        """
        beginnings = [
            "Interestingly,", "Surprisingly,", "In fact,", "As it turns out,", 
            "To begin with,", "First of all,", "On a related note,", "In essence,"
        ]
        if sentence:
            return f"{random.choice(beginnings)} {sentence[0].lower()}{sentence[1:]}"
        return sentence

    
    def _load_paragraph_transformations(self):
        """Load paragraph structure transformations."""
        self.paragraph_transformations = [
            self._transform_paragraph_order,
            self._transform_add_transition,
            self._transform_add_example,
            self._transform_add_elaboration,
            self._transform_change_perspective,
            self._transform_add_rhetorical_question,
            self._transform_add_personal_reflection,
            self._transform_add_counterargument,
            self._transform_add_real_world_connection,
            self._transform_add_real_world_connection,
            self._transform_add_rhetorical_question,
            self._transform_add_historical_context,  # Ensure this method is defined below
        ]

    def _transform_add_personal_reflection(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Add a personal reflection to the paragraph to make it more relatable.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with an added personal reflection
        """
        reflections = [
            "This reminds me of a time when I experienced something similar.",
            "I can't help but think about how this relates to my own life.",
            "Reflecting on this, I realize how it connects to personal experiences.",
            "It makes me wonder about the times I've faced similar challenges."
        ]
        reflection = random.choice(reflections)
        return f"{paragraph} {reflection}"


    def _transform_paragraph_order(self, paragraphs: List[str]) -> List[str]:
        """
        Reorder paragraphs to improve flow or emphasize certain points.
        
        Args:
            paragraphs: List of paragraphs to reorder
            
        Returns:
            Reordered list of paragraphs
        """
        if len(paragraphs) <= 1:
            return paragraphs
        
        # Shuffle the middle paragraphs while keeping the first and last intact
        first = paragraphs[0]
        last = paragraphs[-1]
        middle = paragraphs[1:-1]
        random.shuffle(middle)
        
        return [first] + middle + [last]
    
    def _load_transitions(self):
        """Load transition phrases."""
        self.transitions = {
            "addition": [
                "Furthermore", "Moreover", "In addition", "Additionally", "Also", "Besides", 
                "What's more", "On top of that", "Not only that, but", "As well as", 
                "Another key point", "Similarly", "Likewise", "Along with this", "To add to this",
                "Equally important", "Further", "Then again", "In the same way", "Just as important"
            ],
            "contrast": [
                "However", "Nevertheless", "On the other hand", "Conversely", "In contrast", 
                "Yet", "Still", "Despite this", "Notwithstanding", "Although", 
                "Even though", "While this is true", "Whereas", "Unlike", "But", "Instead",
                "On the contrary", "Alternatively", "Otherwise", "Rather", "In spite of this"
            ],
            "cause_effect": [
                "Therefore", "Thus", "Consequently", "As a result", "Hence", 
                "For this reason", "Because of this", "This leads to", "Accordingly", 
                "Due to this", "This results in", "This causes", "This means that",
                "So", "So that", "With this in mind", "As a consequence", "Thereby"
            ],
            "example": [
                "For example", "For instance", "To illustrate", "As an example", 
                "Specifically", "To demonstrate", "In particular", "Namely", 
                "Such as", "Including", "Consider the case of", "Take the case of",
                "As shown by", "To be specific", "Particularly", "Especially", "Notably"
            ],
            "emphasis": [
                "Indeed", "Certainly", "Undoubtedly", "Without question", "Obviously", 
                "Clearly", "Notably", "Importantly", "Significantly", "Particularly", 
                "Especially", "Above all", "Most importantly", "Remarkably", "Unquestionably",
                "Without a doubt", "Definitely", "Absolutely", "Emphatically", "Truly"
            ],
            "conclusion": [
                "In conclusion", "To summarize", "In summary", "To conclude", 
                "Finally", "Lastly", "In the end", "To wrap up", "All things considered", 
                "In brief", "On the whole", "Ultimately", "To sum up", "In closing",
                "In a nutshell", "To put it briefly", "In the final analysis", "As I've shown"
            ],
            "time": [
                "Meanwhile", "Subsequently", "Previously", "Initially", "Eventually", 
                "Afterward", "Later", "Earlier", "In the meantime", "Simultaneously", 
                "At the same time", "Following this", "Beforehand", "Soon after",
                "Presently", "Formerly", "Lately", "Recently", "During", "Thereafter"
            ],
            "personal_opinion": [
                "In my opinion", "From my perspective","It seems to me",
                "As far as I'm concerned", "To my mind", "I would say", "Personally",
                "In my view", "From where I stand", "I feel that", "My sense is that"
            ],
            "concession": [
                "Admittedly", "Granted", "Of course", "Naturally", "I concede that",
                "It's true that", "To be fair", "I admit that", "Certainly", "No doubt",
                "Understandably", "One could argue that", "It's understandable that"
            ],
            "clarification": [
                "In other words", "That is to say", "To clarify", "To put it another way",
                "Simply put", "To be clear", "Let me explain", "What I mean is",
                "To rephrase that", "More specifically", "Put differently", "By this I mean"
            ]
        }
    
    def _load_fillers(self):
        """Load filler words and phrases by style."""
        self.fillers = {
            "casual": [
                "you know", "like", "I mean", "basically", "actually", 
                "honestly", "pretty much", "kind of", "sort of", "anyway",
                "well", "so", "right", "I guess", "literally", "seriously",
                "totally", "absolutely", "for real", "no joke", "to be honest",
                "as a matter of fact", "believe it or not", "frankly speaking",
                "truth be told", "between you and me", "if you ask me"
            ],
            "professional": [
                "in essence", "fundamentally", "in principle", "in fact",
                "to clarify", "specifically", "to elaborate", "in particular",
                "to be precise", "in this context", "with that in mind",
                "to that end", "consequently", "as a result", "in summary",
                "with respect to", "regarding", "concerning", "pertaining to",
                "in relation to", "in terms of", "from this perspective",
                "in this regard", "in light of this", "given these points"
            ],
            "creative": [
                "imagine", "picture this", "envision", "consider",
                "remarkably", "fascinatingly", "curiously", "strikingly",
                "vividly", "brilliantly", "magnificently", "exquisitely",
                "profoundly", "intensely", "dramatically", "wondrously",
                "mesmerizingly", "enchantingly", "captivatingly", "breathtakingly",
                "astonishingly", "extraordinarily", "spectacularly", "phenomenally"
            ]
        }
    
    def _load_writing_quirks(self):
        """Load human-specific writing quirks."""
        self.writing_quirks = {
            "repetition": [
                "word_repetition",  # Occasionally repeat words
                "idea_repetition",  # Occasionally repeat ideas in different words
                "favorite_phrases"  # Use certain phrases multiple times
            ],
            "inconsistency": [
                "punctuation_variation",  # Inconsistent punctuation
                "spelling_variation",     # Occasional spelling variations
                "formality_shifts"        # Shifts in formality level
            ],
            "personal_touches": [
                "self_reference",         # References to self
                "reader_address",         # Direct address to reader
                "personal_anecdotes",     # Brief personal stories
                "opinion_insertion",      # Insertion of opinions
                "hedging_language"        # Use of hedging language
            ],
            "structural_quirks": [
                "sentence_fragments",     # Occasional sentence fragments
                "run_on_sentences",       # Occasional run-on sentences
                "parenthetical_asides",   # Use of parenthetical asides
                "em_dash_breaks",         # Em dash interruptions
                "varied_sentence_length"  # Highly varied sentence lengths
            ],
            "thought_process": [
                "self_correction",        # Correcting oneself
                "thought_evolution",      # Evolving thoughts within paragraph
                "rhetorical_questions",   # Asking questions to reader
                "thinking_aloud",         # Phrases that mimic thinking aloud
                "uncertainty_markers"     # Expressions of uncertainty
            ]
        }
        
        # Specific examples of human writing quirks
        self.quirk_examples = {
            "self_correction": [
                "Actually, no, that's not quite right.",
                "Or rather,",
                "Let me rephrase that.",
                "What I meant to say was",
                "No, scratch that.",
                "Actually, I think what I'm trying to say is"
            ],
            "hedging_language": [
                "I think", "I believe", "perhaps", "maybe", "possibly",
                "it seems", "it appears", "from what I understand",
                "as far as I know", "in my experience", "I'm not entirely sure, but",
                "I could be wrong, but", "correct me if I'm wrong"
            ],
            "thought_evolution": [
                "At first I thought... but now I realize",
                "Initially it seemed... however, upon reflection",
                "I used to believe... but I've come to see",
                "The more I think about it,",
                "On second thought,"
            ],
            "uncertainty_markers": [
                "I'm not sure if", "I wonder if", "I'm still figuring out",
                "It's hard to say", "Who knows?", "It's difficult to determine",
                "I haven't quite decided", "I go back and forth on this"
            ],
            "favorite_phrases": [
                "at the end of the day", "when all is said and done",
                "for what it's worth", "needless to say", "truth be told",
                "as a matter of fact", "to be perfectly honest"
            ]
        }
    
    def _load_human_writing_patterns(self):
        """Load patterns that mimic human writing styles."""
        # Patterns of human inconsistency
        self.human_inconsistency = {
            "punctuation": {
                "comma_splice": 0.05,      # Probability of comma splice error
                "missing_comma": 0.04,     # Probability of missing a needed comma
                "extra_comma": 0.03,       # Probability of adding an unnecessary comma
                "semicolon_misuse": 0.02,  # Probability of semicolon misuse
                "apostrophe_error": 0.02   # Probability of apostrophe error
            },
            "capitalization": {
                "inconsistent_title_case": 0.03,  # Inconsistent capitalization in titles/headings
                "missed_proper_noun": 0.02,      # Missing capitalization of proper noun
                "over_capitalization": 0.02      # Capitalizing non-proper nouns
            },
            "spelling": {
                "common_misspellings": 0.02,     # Common misspellings
                "typos": 0.01,                   # Simple typos
                "homophone_confusion": 0.01      # Confusing homophones
            },
            "grammar": {
                "subject_verb_disagreement": 0.01,  # Subject-verb disagreement
                "pronoun_antecedent": 0.02,        # Unclear pronoun antecedent
                "dangling_modifier": 0.01          # Dangling modifier
            }
        }
        
        # Sentence length variation patterns
        self.sentence_length_patterns = {
            "short_sentence_probability": 0.25,  # Probability of a short sentence (5-10 words)
            "medium_sentence_probability": 0.55, # Probability of a medium sentence (11-20 words)
            "long_sentence_probability": 0.20,   # Probability of a long sentence (21+ words)
            "min_words_short": 5,
            "max_words_short": 10,
            "min_words_medium": 11,
            "max_words_medium": 20,
            "min_words_long": 21,
            "max_words_long": 35
        }
        
        # Paragraph structure patterns
        self.paragraph_patterns = {
            "short_paragraph_probability": 0.30,  # 1-3 sentences
            "medium_paragraph_probability": 0.60, # 4-6 sentences
            "long_paragraph_probability": 0.10,   # 7+ sentences
            "min_sentences_short": 1,
            "max_sentences_short": 3,
            "min_sentences_medium": 4,
            "max_sentences_medium": 6,
            "min_sentences_long": 7,
            "max_sentences_long": 10
        }
    
    def humanize(
        self,
        text: str,
        style: str = "casual",
        preserve_meaning: bool = True,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Humanize AI-generated text with robust bypass techniques and grammar correction.
        
        Args:
            text: The text to humanize
            style: The style to apply (casual, professional, creative)
            preserve_meaning: Whether to strictly preserve the original meaning
            temperature: Temperature for generation (higher = more creative)
            
        Returns:
            Dictionary with humanized text and metadata
        """
        start_time = time.time()
        
        # Step 1: Preprocess the text
        preprocessed_text = preprocess_text(text)
        
        # Step 2: Get relevant examples from RAG system
        examples = self.rag_system.retrieve(
            query=preprocessed_text,
            style=style,
            k=3
        )
        
        # Step 3: Parse and analyze the text to understand its structure and meaning
        analyzed_text = self._analyze_text(preprocessed_text)
        
        # Step 4: Generate humanized text using robust bypass techniques
        humanized_text = self._robust_bypass_transform(
            text=preprocessed_text,
            analyzed_text=analyzed_text,
            style=style,
            temperature=temperature,
            preserve_meaning=preserve_meaning,
            examples=examples
        )
        
        # Step 5: Apply grammar and spelling corrections while preserving bypass techniques
        humanized_text = self._correct_grammar_preserve_bypass(humanized_text)
        
        # Step 6: Fix repetitive words and phrases
        humanized_text = self._fix_repetitive_words(humanized_text)
        
        # Step 7: Ensure proper sentence structure and punctuation
        humanized_text = self._fix_sentence_structure(humanized_text)
        
        # Step 8: Postprocess the text
        humanized_text = postprocess_text(humanized_text)
        
        # Step 9: Calculate humanness score
        humanness_score = calculate_humanness_score(humanized_text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return {
            "humanized_text": humanized_text,
            "humanness_score": humanness_score,
            "processing_time": processing_time,
            "method": self.llm_type
        }
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text to understand its structure, meaning, and key components.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        analysis = self._initialize_analysis()
        paragraphs = self._split_paragraphs(text)
        analysis["paragraph_count"] = len(paragraphs)
        analysis["paragraphs"] = paragraphs

        all_sentences = self._process_sentences(paragraphs)
        analysis["sentence_count"] = len(all_sentences)
        analysis["sentences"] = all_sentences

        all_words = self._count_words(all_sentences)
        analysis["word_count"] = len(all_words)

        self._calculate_averages(analysis, all_words)

        analysis["key_terms"] = self._extract_key_terms(all_words)

        self._determine_complexity(analysis)
        self._determine_sentiment(analysis, text)

        return analysis

    def _initialize_analysis(self) -> Dict[str, Any]:
        """Initialize the analysis dictionary."""
        return {
            "sentences": [],
            "paragraphs": [],
            "key_terms": [],
            "sentiment": "neutral",
            "complexity": "medium",
            "formality": "medium",
            "topics": [],
            "entities": [],
            "word_count": 0,
            "sentence_count": 0,
            "paragraph_count": 0,
            "avg_sentence_length": 0,
            "avg_word_length": 0,
        }

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = text.split("\n\n")
        if len(paragraphs) == 1 and "\n" in text:
            paragraphs = text.split("\n")
        return paragraphs

    def _process_sentences(self, paragraphs: List[str]) -> List[str]:
        """Process sentences from paragraphs."""
        all_sentences = []
        for paragraph in paragraphs:
                    sentences = re.split(SENTENCE_SPLIT_PATTERN, paragraph)
        if NLTK_AVAILABLE:
                # Removed unused assignment
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        all_sentences.extend(sentences)
        all_sentences.extend(sentences)
        return all_sentences

    def _count_words(self, sentences: List[str]) -> List[str]:
        """Count words in sentences."""
        all_words = []
        for sentence in sentences:
            if NLTK_AVAILABLE:
                words = word_tokenize(sentence)
            else:
                words = sentence.split()
            all_words.extend(words)
        return all_words

    def _calculate_averages(self, analysis: Dict[str, Any], words: List[str]):
        """Calculate average sentence length and word length."""
        if analysis["sentence_count"] > 0:
            analysis["avg_sentence_length"] = analysis["word_count"] / analysis["sentence_count"]
        if analysis["word_count"] > 0:
            total_char_count = sum(len(word) for word in words)
            analysis["avg_word_length"] = total_char_count / analysis["word_count"]

    def _extract_key_terms(self, words: List[str]) -> List[str]:
        """Extract key terms from words."""
        if NLTK_AVAILABLE:
            filtered_words = [word.lower() for word in words if word.lower() not in self.stopwords and word.isalpha()]
            word_freq = Counter(filtered_words)
            return [term for term, _ in word_freq.most_common(10)]
        else:
            word_freq = Counter([word.lower() for word in words if len(word) > 3 and word.isalpha()])
            return [term for term, _ in word_freq.most_common(10)]

    def _determine_complexity(self, analysis: Dict[str, Any]):
        """Determine text complexity."""
        if analysis["avg_sentence_length"] > 20 or analysis["avg_word_length"] > 6:
            analysis["complexity"] = "high"
        elif analysis["avg_sentence_length"] < 10 or analysis["avg_word_length"] < 4:
            analysis["complexity"] = "low"

    def _determine_sentiment(self, analysis: Dict[str, Any], text: str):
        """Determine sentiment of the text."""
        if TEXTBLOB_AVAILABLE:
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            if sentiment > 0.2:
                analysis["sentiment"] = "positive"
            elif sentiment < -0.2:
                analysis["sentiment"] = "negative"
    
    def _robust_bypass_transform(
        self,
        text: str,
        analyzed_text: Dict[str, Any],
        style: str,
        temperature: float,
        preserve_meaning: bool,
        examples: List[Dict[str, Any]]
    ) -> str:
        """
        Apply robust bypass transformations to text that preserve meaning and readability.
        
        Args:
            text: The preprocessed text
            analyzed_text: Analysis of the text
            style: The style to apply
            temperature: Temperature for generation (higher = more creative)
            preserve_meaning: Whether to strictly preserve meaning
            examples: Retrieved examples from RAG
            
        Returns:
            Transformed text with bypass techniques
        """
        logger.info(f"Applying robust bypass transformation with style: {style}")
        
        paragraphs = analyzed_text["paragraphs"]
        transformed_paragraphs = [
            self._process_paragraph(paragraph, paragraphs, style, temperature, preserve_meaning, examples)
            for paragraph in paragraphs if paragraph.strip()
        ]
        
        if len(transformed_paragraphs) > 2 and random.random() < 0.3 * temperature and not preserve_meaning:
            transformed_paragraphs = self._transform_paragraph_order(transformed_paragraphs)
        
        if len(transformed_paragraphs) > 1:
            transformed_paragraphs = self._add_transitions_between_paragraphs(
                transformed_paragraphs, style
            )
        
        humanized_text = "\n\n".join(transformed_paragraphs)
        
        if not preserve_meaning and random.random() < 0.3 * temperature:
            humanized_text = self._apply_global_transformations(humanized_text, style, temperature)
        
        return humanized_text

    def _process_paragraph(
        self,
        paragraph: str,
        paragraphs: List[str],
        style: str,
        temperature: float,
        preserve_meaning: bool,
        examples: List[Dict[str, Any]]
    ) -> str:
        """Process a single paragraph with transformation layers."""
        transformed_paragraph = self._transform_words(paragraph, style, temperature)
        
        if random.random() < 0.7 * temperature:
            transformed_paragraph = self._apply_subtle_char_substitutions(transformed_paragraph, temperature)
        
        if random.random() < 0.6 * temperature:
            transformed_paragraph = self._transform_sentences(transformed_paragraph, style, temperature)
        
        if random.random() < 0.6 * temperature:
            transformed_paragraph = self._apply_subtle_spacing_variations(transformed_paragraph, temperature)
        
        if len(paragraphs) > 1 and random.random() < 0.5 * temperature:
            transformed_paragraph = self._apply_random_paragraph_transformation(
                transformed_paragraph, style, temperature
            )
        
        if examples and random.random() < 0.4 * temperature:
            transformed_paragraph = self._apply_example_based_patterns(
                transformed_paragraph, examples, style, temperature
            )
        
        if random.random() < 0.5 * temperature:
            transformed_paragraph = self._apply_subtle_punctuation_variations(transformed_paragraph, temperature)
        
        if not preserve_meaning and random.random() < 0.3 * temperature:
            transformed_paragraph = self._add_personal_touches(
                transformed_paragraph, style, temperature
            )
        
        return transformed_paragraph
    
    def _apply_subtle_char_substitutions(self, text: str, temperature: float) -> str:
        """
        Apply subtle character substitutions to bypass detection.
        
        Args:
            text: The text to transform
            temperature: Temperature for generation (higher = more substitutions)
            
        Returns:
            Text with subtle character substitutions
        """
        if not text:
            return text
        
        chars = list(text)
        
        # Determine number of substitutions based on text length and temperature
        # Higher temperature = more substitutions
        num_substitutions = max(1, min(len(chars) // 20, int(len(chars) * 0.05 * temperature)))
        
        # Randomly select positions for substitution
        positions = random.sample(range(len(chars)), num_substitutions)
        
        for pos in positions:
            char = chars[pos].lower()
            
            # Only substitute letters that have similar-looking alternatives
            if char in self.char_substitutions and random.random() < 0.8 * temperature:
                # Preserve case
                is_upper = chars[pos].isupper()
                
                # Choose a random substitution
                substitution = random.choice(self.char_substitutions[char])
                
                # Apply case if needed
                if is_upper and substitution.upper() != substitution:
                    # If we can't uppercase the substitution, skip it
                    continue
                
                chars[pos] = substitution.upper() if is_upper else substitution
        
        return ''.join(chars)
    
    def _apply_subtle_spacing_variations(self, text: str, temperature: float) -> str:
        """
        Apply subtle spacing variations to bypass detection.
        
        Args:
            text: The text to transform
            temperature: Temperature for generation (higher = more variations)
            
        Returns:
            Text with subtle spacing variations
        """
        if not text:
            return text
        
        words = text.split()
        result = []
        
        for word in words:
            result.append(word)
            
            # 10% chance to add a special space instead of a regular space
            if random.random() < 0.1 * temperature:
                result.append(random.choice(self.spacing_variations))
            else:
                result.append(' ')
        
        # Remove the last space
        if result:
            result.pop()
        
        return ''.join(result)
    
    def _apply_subtle_punctuation_variations(self, text: str, temperature: float) -> str:
        """
        Apply subtle punctuation variations to bypass detection.
        
        Args:
            text: The text to transform
            temperature: Temperature for generation (higher = more variations)
            
        Returns:
            Text with subtle punctuation variations
        """
        if not text:
            return text
        
        chars = list(text)
        
        # Determine number of substitutions based on text length and temperature
        num_substitutions = max(1, min(len(chars) // 30, int(len(chars) * 0.03 * temperature)))
        
        # Count punctuation characters
        punctuation_positions = [i for i, char in enumerate(chars) if char in self.punctuation_variations]
        
        if not punctuation_positions:
            return text
        
        # Randomly select positions for substitution
        positions = random.sample(punctuation_positions, min(num_substitutions, len(punctuation_positions)))
        
        for pos in positions:
            char = chars[pos]
            
            if char in self.punctuation_variations and random.random() < 0.7 * temperature:
                # Choose a random substitution
                substitution = random.choice(self.punctuation_variations[char])
                chars[pos] = substitution
        
        return ''.join(chars)
    
    def _correct_grammar_preserve_bypass(self, text: str) -> str:
        """
        Correct grammar and spelling while preserving bypass techniques.
        
        Args:
            text: The text to correct
            
        Returns:
            Corrected text with bypass techniques preserved
        """
        bypass_chars = self._identify_bypass_characters(text)
        clean_text = self._replace_bypass_characters(text, bypass_chars)
        clean_text = self._apply_corrections(clean_text)
        corrected_text = self._reinsert_bypass_characters(clean_text, text, bypass_chars)
        return corrected_text

    def _identify_bypass_characters(self, text: str) -> Dict[int, str]:
        """Identify and preserve bypass characters."""
        bypass_chars = {}
        for i, char in enumerate(text):
            if self._is_bypass_character(char):
                bypass_chars[i] = char
        return bypass_chars

    def _is_bypass_character(self, char: str) -> bool:
        """Check if a character is a bypass character."""
        if any(char in substitutions for substitutions in self.char_substitutions.values()):
            return True
        if char in self.spacing_variations:
            return True
        if any(char in substitutions for substitutions in self.punctuation_variations.values()):
            return True
        return False

    def _replace_bypass_characters(self, text: str, bypass_chars: Dict[int, str]) -> str:
        """Replace bypass characters with their standard equivalents."""
        clean_text = list(text)
        for i, char in bypass_chars.items():
            clean_text[i] = self._get_standard_equivalent(char)
        return ''.join(clean_text)

    def _get_standard_equivalent(self, char: str) -> str:
        """Get the standard equivalent of a bypass character."""
        for original, substitutions in self.char_substitutions.items():
            if char in substitutions:
                return original
        if char in self.spacing_variations:
            return ' '
        for original, substitutions in self.punctuation_variations.items():
            if char in substitutions:
                return original
        return char

    def _apply_corrections(self, text: str) -> str:
        """Apply grammar and spelling corrections."""
        if self.language_tool:
            text = self._fix_grammar(text)
        if self.spell:
            text = self._fix_spelling(text)
        return text

    def _reinsert_bypass_characters(self, clean_text: str, original_text: str, bypass_chars: Dict[int, str]) -> str:
        """Reinsert bypass characters into the corrected text."""
        corrected_text = list(clean_text)
        if len(corrected_text) == len(original_text):
            for i, char in bypass_chars.items():
                corrected_text[i] = char
        return ''.join(corrected_text)
    
    def _fix_repetitive_words(self, text: str) -> str:
        """
        Fix repetitive words and phrases in text.
        
        Args:
            text: The text to fix
            
        Returns:
            Text with reduced repetition
        """
        if not text:
            return text
        
        sentences = self._split_into_sentences(text)
        all_words = self._tokenize_and_filter_words(sentences)
        repetitive_words = self._identify_repetitive_words(all_words)
        text = self._replace_repetitive_words(text, repetitive_words)
        
        return text

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if NLTK_AVAILABLE:
            return re.split(SENTENCE_SPLIT_PATTERN, text)
        else:
            # Fallback: split on period, exclamation, or question mark followed by whitespace
            return re.split(r'(?<=[.!?])\s+', text)

    def _tokenize_and_filter_words(self, sentences: List[str]) -> List[str]:
        """Tokenize sentences and filter words."""
        all_words = []
        for sentence in sentences:
            if NLTK_AVAILABLE:
                words = word_tokenize(sentence.lower())
                words = [word for word in words if word.isalpha() and word not in self.stopwords]
            else:
                words = [word.lower() for word in sentence.split() if word.isalpha() and len(word) > 3]
            all_words.extend(words)
        return all_words

    def _identify_repetitive_words(self, words: List[str]) -> Dict[str, int]:
        """Identify repetitive words."""
        word_freq = Counter(words)
        return {word: count for word, count in word_freq.items() if count > 3}

    def _replace_repetitive_words(self, text: str, repetitive_words: Dict[str, int]) -> str:
        """Replace repetitive words with synonyms."""
        for word, count in repetitive_words.items():
            synonyms = self._find_synonyms(word)
            if not synonyms:
                continue
            
            replace_count = max(1, count // 2)
            pattern = r'\b' + re.escape(word) + r'\b'
            replacements = 0

            def replace_with_synonym(match, replace_count=replace_count, synonyms=synonyms):
                nonlocal replacements
                if replacements < replace_count and random.random() < 0.7:
                    replacements += 1
                    return random.choice(synonyms)
                return match.group(0)
            
            text = re.sub(pattern, replace_with_synonym, text, flags=re.IGNORECASE)
        return text
    
    def _find_synonyms(self, word: str) -> List[str]:
        """
        Find synonyms for a word.
        
        Args:
            word: The word to find synonyms for
            
        Returns:
            List of synonyms
        """
        # Check our synonym dictionary first
        if word in self.synonyms:
            return self.synonyms[word]
        
        synonyms = []
        
        # Use WordNet if available
        if NLTK_AVAILABLE:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != word and synonym not in synonyms:
                        synonyms.append(synonym)
        
        return synonyms[:5] if synonyms else []  # Limit to 5 synonyms
    
    def _fix_sentence_structure(self, text: str) -> str:
        """
        Fix sentence structure and punctuation issues.
        
        Args:
            text: The text to fix
            
        Returns:
            Text with improved sentence structure
        """
        if not text:
            return text
        
        # Split into sentences
        if NLTK_AVAILABLE:
            # Removed unused assignment to 'sentences'
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        fixed_sentences = []
        
        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue
            
            # Fix capitalization at the beginning of sentences
            if sentence and sentence[0].islower():
                sentence = sentence[0].upper() + sentence[1:]
            
            # Ensure sentences end with proper punctuation
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            # Fix spacing after punctuation
            sentence = re.sub(r'([.!?,;:])([A-Za-z])', r'\1 \2', sentence)
            
            # Fix multiple spaces
            sentence = re.sub(r' +', ' ', sentence)
            
            fixed_sentences.append(sentence)
        
        # Join sentences with proper spacing
        fixed_text = ' '.join(fixed_sentences)
        
        return fixed_text
    
    def _fix_grammar(self, text: str) -> str:
        """
        Fix grammar errors in text.

        Args:
            text: The text to correct

        Returns:
            Text with grammar errors corrected
        """
        # Some versions of LanguageTool server or API may not support certain characters or actions.
        # To avoid "Unsupported action: ':'", sanitize the text before sending.
        safe_text = text.replace('\u2028', ' ').replace('\u2029', ' ')
        # Optionally, you can remove or replace problematic colons if needed:
        safe_text = safe_text.replace(':', ';')

        try:
            matches = self.language_tool.check(safe_text)
        except Exception as e:
            logger.warning(f"LanguageTool error: {e}")
            return text  # Return original text if LanguageTool fails

        corrected_text = safe_text
        offset = 0

        for match in sorted(matches, key=lambda m: m.offset):
            # Skip certain categories of errors
            if match.ruleId in ['WHITESPACE_RULE', 'EN_QUOTES']:
                continue

            # Apply the correction
            if match.replacements:
                from_pos = match.offset + offset
                to_pos = match.offset + match.errorLength + offset
                replacement = match.replacements[0]

                corrected_text = corrected_text[:from_pos] + replacement + corrected_text[to_pos:]
                offset += len(replacement) - match.errorLength

        return corrected_text
    
    def _fix_spelling(self, text: str) -> str:
        """
        Fix spelling errors in text.
        
        Args:
            text: The text to correct
            
        Returns:
            Text with spelling errors corrected
        """
        words = text.split()
        corrected_words = [self._correct_word(word) for word in words]
        return ' '.join(corrected_words)

    def _correct_word(self, word: str) -> str:
        """
        Correct a single word for spelling errors.
        
        Args:
            word: The word to correct
            
        Returns:
            Corrected word with preserved punctuation.
        """
        prefix, core_word, suffix = self._extract_word_parts(word)
        
        if not core_word or any(c.isdigit() for c in core_word):
            return prefix + core_word + suffix
        
        corrected_word = self._apply_spelling_correction(core_word)
        return prefix + corrected_word + suffix

    def _extract_word_parts(self, word: str) -> tuple:
        """
        Extract the prefix, core word, and suffix from a word.
        
        Args:
            word: The word to extract parts from
            
        Returns:
            A tuple of (prefix, core_word, suffix).
        """
        prefix = ''
        suffix = ''
        while word and not word[0].isalnum():
            prefix += word[0]
            word = word[1:]
        while word and not word[-1].isalnum():
            suffix = word[-1] + suffix
            word = word[:-1]
        return prefix, word, suffix

    def _apply_spelling_correction(self, word: str) -> str:
        """
        Apply spelling correction to a word.
        
        Args:
            word: The word to correct
            
        Returns:
            Corrected word.
        """
        if word.lower() not in self.spell and not word.isupper():
            correction = self.spell.correction(word)
            if correction and correction != word:
                if word[0].isupper():
                    correction = correction[0].upper() + correction[1:]
                return correction
        return word
    
    def _transform_words(self, text: str, style: str, temperature: float) -> str:
        """Transform words using synonyms and style-specific patterns."""
        words = self._replace_with_synonyms(text, temperature)
        transformed_text = " ".join(words)
        transformed_text = self._apply_style_specific_transformations(transformed_text, style, temperature)
        return transformed_text

    def _replace_with_synonyms(self, text: str, temperature: float) -> List[str]:
        """Replace words with synonyms based on temperature."""
        words = text.split()
        for i, word in enumerate(words):
            clean_word = word.lower().strip(string.punctuation)
            if len(clean_word) <= 3 or not clean_word.isalpha():
                continue
            if clean_word in self.synonyms and random.random() < 0.3 * temperature:
                synonym_options = self.synonyms[clean_word]
                new_word = random.choice(synonym_options)
                if word[0].isupper():
                    new_word = new_word.capitalize()
                if not word[-1].isalpha():
                    new_word += word[word.rindex(clean_word[-1]) + 1:]
                words[i] = new_word
        return words

    def _apply_style_specific_transformations(self, text: str, style: str, temperature: float) -> str:
        """Apply style-specific word transformations."""
        if style == "casual":
            text = self._apply_casual_transformations(text, temperature)
        elif style == "professional":
            text = self._apply_professional_transformations(text, temperature)
        return text

    def _apply_casual_transformations(self, text: str, temperature: float) -> str:
        """Apply casual style transformations."""
        contractions = {
            r'\b(can)not\b': r'\1\'t',
            r'\b(do|does|did|would|should|could|will|is|are|have|has|had) not\b': r'\1n\'t',
            r'\bI am\b': r'I\'m',
            r'\byou are\b': r'you\'re',
            r'\bthey are\b': r'they\'re',
            r'\bwe are\b': r'we\'re',
            r'\bhe is\b': r'he\'s',
            r'\bshe is\b': r'she\'s',
            r'\bit is\b': r'it\'s',
            r'\bthat is\b': r'that\'s',
            r'\bwhat is\b': r'what\'s',
            r'\bwho is\b': r'who\'s',
            r'\bwhere is\b': r'where\'s',
            r'\bhow is\b': r'how\'s',
            r'\bI will\b': r'I\'ll',
            r'\byou will\b': r'you\'ll',
            r'\bhe will\b': r'he\'ll',
            r'\bshe will\b': r'she\'ll',
            r'\bit will\b': r'it\'ll',
            r'\bthey will\b': r'they\'ll',
            r'\bwe will\b': r'we\'ll',
        }
        for pattern, replacement in contractions.items():
            if random.random() < 0.6 * temperature:
                text = re.sub(pattern, replacement, text)
        return text

    def _apply_professional_transformations(self, text: str, temperature: float) -> str:
        """Apply professional style transformations."""
        expansions = {
            r'\bcan\'t\b': r'cannot',
            r'\bwon\'t\b': r'will not',
            r'\bdon\'t\b': r'do not',
            r'\bdoesn\'t\b': r'does not',
            r'\bdidn\'t\b': r'did not',
            r'\bI\'m\b': r'I am',
            r'\byou\'re\b': r'you are',
            r'\bthey\'re\b': r'they are',
            r'\bwe\'re\b': r'we are',
            r'\bhe\'s\b': r'he is',
            r'\bshe\'s\b': r'she is',
            r'\bit\'s\b': r'it is',
            r'\bthat\'s\b': r'that is',
            r'\bwhat\'s\b': r'what is',
            r'\bI\'ll\b': r'I will',
            r'\byou\'ll\b': r'you will',
            r'\bhe\'ll\b': r'he will',
            r'\bshe\'ll\b': r'she will',
            r'\bit\'ll\b': r'it will',
            r'\bthey\'ll\b': r'they will',
            r'\bwe\'ll\b': r'we will',
        }
        for pattern, replacement in expansions.items():
            if random.random() < 0.6 * temperature:
                text = re.sub(pattern, replacement, text)
        return text
    
    # Word structure variation methods
    def _add_suffix(self, word: str) -> str:
        """Add a suffix to a word."""
        suffixes = ['ly', 'ish', 'ness', 'ful', 'less', 'able', 'ment', 'tion', 'ity', 'ize']
        return word + random.choice(suffixes)
    
    def _change_tense(self, word: str) -> str:
        """Change the tense of a verb."""
        if word.endswith('e'):
            return word + 'd'
        elif word.endswith('y'):
            return word[:-1] + 'ied'
        else:
            return word + 'ed'
    
    def _pluralize(self, word: str) -> str:
        """Pluralize a noun."""
        if word.endswith('y'):
            return word[:-3] + 'ies'
        elif word.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return word + 'es'
        else:
            return word + 's'
    
    def _singularize(self, word: str) -> str:
        """Singularize a noun."""
        if word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('es'):
            return word[:-2]
        elif word.endswith('s'):
            return word[:-1]
        return word
    
    def _add_prefix(self, word: str) -> str:
        """Add a prefix to a word."""
        prefixes = ['un', 're', 'in', 'dis', 'pre', 'post', 'anti', 'non', 'over', 'under']
        return random.choice(prefixes) + word
    
    def _change_comparative(self, word: str) -> str:
        """Change an adjective to its comparative form."""
        if len(word) <= 2:
            return word
        elif word.endswith('e'):
            return word + 'r'
        elif word.endswith('y'):
            return word[:-1] + 'ier'
        else:
            return word + 'er'
    
    def _change_superlative(self, word: str) -> str:
        """Change an adjective to its superlative form."""
        if len(word) <= 2:
            return word
        elif word.endswith('e'):
            return word + 'st'
        elif word.endswith('y'):
            return word[:-1] + 'iest'
        else:
            return word + 'est'
    
    # Sentence structure variation methods
    def _reorder_clauses(self, sentence: str, style: str, temperature: float) -> str:
        """
        Reorder clauses in a sentence.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Sentence with reordered clauses
        """
        # Split by commas
        clauses = sentence.split(', ')
        if len(clauses) <= 1:
            return sentence
        
        # Only reorder if temperature is high enough
        if random.random() < 0.7 * temperature:
            # Shuffle the middle clauses
            first = clauses[0]
            last = clauses[-1]
            middle = clauses[1:-1]
            random.shuffle(middle)
            
            return ', '.join([first] + middle + [last])
            
        return sentence
    
    def _change_voice(self, sentence: str, style: str, temperature: float) -> str:
        """Change between active and passive voice."""
        if 'by' in sentence and random.random() < 0.5:
            return self._transform_passive_to_active(sentence)
        else:
            return self._transform_active_to_passive(sentence, style)

    def _transform_passive_to_active(self, sentence: str) -> str:
        """
        Transform a sentence from passive to active voice.
        
        Args:
            sentence: The sentence to transform
            
        Returns:
            Transformed sentence in active voice
        """
        # Simple implementation for transforming passive to active
        if "was" in sentence and "by" in sentence:
            parts = sentence.split(" by ")
            if len(parts) == 2:
                subject = parts[1].strip(".!?")
                predicate = parts[0].replace("was", "").strip()
                return f"{subject} {predicate}."
        return sentence
    
    def _split_sentence(self, sentence: str, style: str, temperature: float) -> str:
        """Split a sentence into two.
        
        Args:
            sentence: The sentence to split
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Split sentence
        """
        words = sentence.split()
        if len(words) < 8:  # Don't split short sentences
            return sentence
            
        # Find a suitable split point
        split_point = len(words) // 2
        for i in range(max(3, split_point - 2), min(len(words) - 3, split_point + 2)):
            if words[i].lower() in ['and', 'but', 'or', 'while', 'although', 'however']:
                split_point = i
                break
                
        # Create two sentences
        first_part = ' '.join(words[:split_point])
        second_part = ' '.join(words[split_point:])
        
        # Ensure proper capitalization and punctuation
        if not first_part.endswith(('.', '!', '?')):
            first_part += '.'
        if second_part[0].islower():
            second_part = second_part[0].upper() + second_part[1:]
            
        return f"{first_part} {second_part}"
    
    def _transform_active_to_passive(self, sentence: str, style: str) -> str:
        """
        Transform a sentence from active to passive voice.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed sentence in passive voice
        """
        # Simple implementation for transforming active to passive
        words = sentence.split()
        if len(words) < 3:
            return sentence  # Return as is if too short to transform
        
        # Example transformation: "The cat chased the mouse." -> "The mouse was chased by the cat."
        if "by" not in words:
            try:
                subject = words[0]
                verb = words[1]
                rest = " ".join(words[2:])
                return f"{rest} was {verb} by {subject}"
            except IndexError:
                return sentence
        return sentence

    def _combine_sentences(self, sentence: str, style: str, temperature: float) -> str:
        """Simulate combining with another sentence.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Combined sentences
        """
        return self._transform_combine_sentences(sentence, style, temperature)
    
    def _add_subordinate_clause(self, sentence: str, style: str, temperature: float) -> str:
        """Add a subordinate clause to a sentence.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Sentence with an added subordinate clause
        """
        subordinate_clauses = [
            "while considering the implications",
            "although it may seem complex",
            "when examining the details",
            "since that's how it works",
            "because of its nature"
        ]
        clause = random.choice(subordinate_clauses)
        return f"{sentence[:-1]}, {clause}{sentence[-1]}"
    
    def _change_question_to_statement(self, sentence: str, style: str, temperature: float) -> str:
        """
        Change a question to a statement.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Statement version of the question
        """
        if sentence.endswith('?'):
            # Remove the question mark
            sentence = sentence[:-1] + '.'
            
            # Replace question words
            question_words = {
                'What': 'That',
                'Who': 'Someone',
                'Where': 'Somewhere',
                'When': 'Sometime',
                'Why': 'Because',
                'How': 'Somehow'
            }
            
            for q_word, replacement in question_words.items():
                if sentence.startswith(q_word):
                    sentence = sentence.replace(q_word, replacement, 1)
                    break
            
            # Reorder subject-verb if needed
            sentence = re.sub(r'(Do|Does|Did|Is|Are|Was|Were|Have|Has|Had|Can|Could|Will|Would|Should|Might|Must) ([A-Za-z]+)', r'\2 \1', sentence)
        
        return sentence
    
    def _change_statement_to_question(self, sentence: str, style: str, temperature: float) -> str:
        """
        Change a statement to a question.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed sentence as a question
        """
        return self._transform_declarative_to_question(sentence, style)
    
    def _transform_declarative_to_question(self, sentence: str, style: str) -> str:
        """
        Transform a declarative sentence into a question.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed sentence as a question
        """
        if not sentence.endswith('.'):
            return sentence  # Return as is if not a declarative sentence
        
        # Basic transformation: Add a question word and a question mark
        question_words = ["Why", "How", "What", "When", "Where"]
        question_word = random.choice(question_words)
        return f"{question_word} {sentence[:-1]}?"

    def _transform_simple_to_complex(self, sentence: str, style: str) -> str:
        """
        Transform a simple sentence into a complex sentence.
        
        Args:
            sentence: The sentence to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed sentence as a complex sentence
        """
        conjunctions = ["because", "although", "since", "while", "even though"]
        conjunction = random.choice(conjunctions)
        additional_clause = "it provides more context or detail."
        return f"{sentence[:-1]} {conjunction} {additional_clause}"
    
    def _transform_sentences(self, text: str, style: str, temperature: float) -> str:
        """Apply sentence-level transformations."""
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        transformed_sentences = []
        
        # Process each sentence
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Apply random sentence transformations based on temperature
            if random.random() < 0.4 * temperature:
                # Choose a random transformation
                transformation = random.choice(self.sentence_transformations)
                transformed_sentence = transformation(sentence, style, temperature)
                
                # If transformation failed, use original sentence
                if not transformed_sentence or transformed_sentence == sentence:
                    transformed_sentence = sentence
            else:
                transformed_sentence = sentence
            
            # Add style-specific fillers
            if random.random() < 0.2 * temperature:
                words = transformed_sentence.split()
                if len(words) > 4:
                    insert_idx = random.randint(2, min(len(words) - 2, 5))
                    filler = random.choice(self.fillers[style])
                    words.insert(insert_idx, filler)
                    transformed_sentence = " ".join(words)
            
            transformed_sentences.append(transformed_sentence)
        
        # Vary sentence lengths according to human patterns
        transformed_sentences = self._vary_sentence_lengths(transformed_sentences, temperature)
        
        # Join sentences
        transformed_text = " ".join(transformed_sentences)
        
        # Ensure proper spacing after punctuation
        transformed_text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', transformed_text)
        
        return transformed_text
    
    def _vary_sentence_lengths(self, sentences: List[str], temperature: float) -> List[str]:
        """Vary sentence lengths according to human patterns."""
        if len(sentences) <= 2:
            return sentences

        result = []
        last_length_type = None

        for sentence in sentences:
            current_length_type = self._determine_sentence_length_type(sentence)
            if self._should_combine_or_split(last_length_type, current_length_type, temperature):
                sentence = self._combine_or_split_sentence(result, sentence, current_length_type)
            result.append(sentence)
            last_length_type = current_length_type

        return result

    def _determine_sentence_length_type(self, sentence: str) -> str:
        """Determine the length type of a sentence."""
        word_count = len(sentence.split())
        if word_count <= self.sentence_length_patterns["max_words_short"]:
            return "short"
        elif word_count <= self.sentence_length_patterns["max_words_medium"]:
            return "medium"
        return "long"

    def _should_combine_or_split(self, last_length_type: str, current_length_type: str, temperature: float) -> bool:
        """Check if sentences should be combined or split."""
        return last_length_type == current_length_type and random.random() < 0.2 * temperature

    def _combine_or_split_sentence(self, result: List[str], sentence: str, current_length_type: str) -> str:
        """Combine or split sentences based on length type."""
        if current_length_type == "short" and len(result) > 0:
            return self._combine_short_sentences(result, sentence)
        elif current_length_type == "long":
            return self._split_long_sentence(sentence)
        return sentence

    def _combine_short_sentences(self, result: List[str], sentence: str) -> str:
        """Combine short sentences."""
        last_sentence = result.pop()
        return last_sentence.rstrip(".!?") + ", " + sentence[0].lower() + sentence[1:]

    def _split_long_sentence(self, sentence: str) -> str:
        """Split a long sentence into two."""
        words = sentence.split()
        if len(words) <= 15:
            return sentence

        split_point = self._find_split_point(words)
        first_part = " ".join(words[:split_point])
        second_part = " ".join(words[split_point:])

        if second_part and second_part[0].islower():
            second_part = second_part[0].upper() + second_part[1:]
        if not first_part.endswith((".", "!", "?")):
            first_part += "."

        return f"{first_part} {second_part}"

    @staticmethod
    def _find_split_point(words: List[str]) -> int:
        """Find the best point to split a sentence."""
        split_point = len(words) // 2
        for i in range(max(3, split_point - 3), min(len(words) - 3, split_point + 3)):
            if words[i].lower() in ["and", "but", "or", "so", "yet", "because", "since", "although"]:
                return i
        return split_point
    
    def _transform_add_counterargument(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Add a counterargument to the paragraph to provide balance or contrast.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with an added counterargument
        """
        counterarguments = [
            "On the other hand, some might argue that...",
            "However, it's worth considering the opposing viewpoint that...",
            "Conversely, one could say that...",
            "That said, there are those who believe that..."
        ]
        counterargument = random.choice(counterarguments)
        return f"{paragraph} {counterargument}"

    def _apply_random_paragraph_transformation(self, paragraph: str, style: str, temperature: float) -> str:
        """Apply a random paragraph transformation."""
        transformation = random.choice([
            self._transform_add_example,
            self._transform_add_elaboration,
            self._transform_add_transition,
            self._transform_change_perspective,
            self._transform_add_rhetorical_question,
            self._transform_add_personal_reflection,
            self._transform_add_counterargument,
            self._transform_add_real_world_connection,
        ])
        
        return transformation(paragraph, style, temperature)

    def _transform_change_perspective(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Change the perspective of the paragraph (e.g., first-person to third-person).
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with a changed perspective
        """
        perspectives = [
            "From a broader perspective, ",
            "Looking at this from another angle, ",
            "Considering this from a different viewpoint, ",
            "If we shift our perspective, "
        ]
        perspective = random.choice(perspectives)
        return f"{perspective}{paragraph}"

    def _transform_add_elaboration(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Add elaboration to the paragraph to provide more detail or context.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with added elaboration
        """
        elaborations = [
            "To elaborate further,",
            "More specifically,",
            "To put it in more detail,",
            "To expand on this point,"
        ]
        elaboration = random.choice(elaborations)
        return f"{paragraph} {elaboration}"

    def _transform_combine_sentences(self, sentence1: str, sentence2: str, style: str = None) -> str:
        """
        Combine two sentences into one.
        
        Args:
            sentence1: First sentence
            sentence2: Second sentence
            style: The style to apply (optional)
            
        Returns:
            Combined sentence
        """
        conjunctions = ["and", "while", "whereas", "although", "however"]
        conjunction = random.choice(conjunctions)
        
        # Remove ending punctuation from first sentence
        sentence1 = sentence1.rstrip('.!?')
        
        # Ensure proper capitalization of second sentence
        if sentence2[0].isupper():
            sentence2 = sentence2[0].lower() + sentence2[1:]
            
        return f"{sentence1} {conjunction} {sentence2}"
    def _transform_add_real_world_connection(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Add a real-world connection to the paragraph to make it more relatable.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with a real-world connection
        """
        connections = [
            "This is similar to what we see in real-life scenarios.",
            "A real-world example of this can be observed in everyday life.",
            "This reminds us of situations we encounter in the real world.",
            "In practical terms, this is often seen in real-world applications."
        ]
        connection = random.choice(connections)
        return f"{paragraph} {connection}"

    def _transform_add_historical_context(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Add historical context to the paragraph to provide depth and perspective.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with added historical context
        """
        historical_contexts = [
            "Historically, this has been a significant factor in...",
            "Looking back, we can see how this played a role in...",
            "In the past, this was often associated with...",
            "Throughout history, this has been evident in..."
        ]
        context = random.choice(historical_contexts)
        return f"{paragraph} {context}"

    def _transform_add_rhetorical_question(self, paragraph: str) -> str:
        elaboration = "This provides additional context or detail."
        return f"{paragraph} {elaboration}"
    def _transform_add_rhetorical_question(self, paragraph: str, style: str) -> str:
        """
        Add a rhetorical question to the paragraph to engage the reader.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with an added rhetorical question
        """
        rhetorical_questions = [
            "Isn't it fascinating how this works?",
            "Have you ever wondered why this happens?",
            "What could be more intriguing than this?",
            "Doesn't this make you think?"
        ]
        question = random.choice(rhetorical_questions)
        return f"{paragraph} {question}"

    def _transform_add_example(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Add an example to the paragraph to make it more illustrative.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with an added example
        """
        examples = [
            "For instance, consider the case of...",
            "As an example, think about...",
            "To illustrate, imagine a scenario where...",
            "Take, for example, the situation in which..."
        ]
        example = random.choice(examples)
        return f"{paragraph} {example}"

    def _transform_add_transition(self, paragraph: str, style: str, temperature: float) -> str:
        """
        Add a transition phrase to the beginning of a paragraph.
        
        Args:
            paragraph: The paragraph to transform
            style: The style to apply
            temperature: Temperature for generation
            
        Returns:
            Transformed paragraph with a transition phrase
        """
        if not paragraph.strip():
            return paragraph
        
        transition_phrases = self.transitions.get("addition", ["Furthermore", "Moreover", "In addition"])
        transition = random.choice(transition_phrases)
        
        if paragraph[0].isupper():
            return f"{transition}, {paragraph[0].lower()}{paragraph[1:]}"
        else:
            return f"{transition}, {paragraph}"
    
    def _apply_example_based_patterns(self, paragraph: str, examples: List[Dict[str, Any]], style: str, temperature: float) -> str:
        """Apply patterns based on retrieved examples."""
        if not examples:
            return paragraph

        example = random.choice(examples)
        humanized_example = example.get("humanized", "")
        if not humanized_example:
            return paragraph

        patterns = self._extract_patterns_from_example(humanized_example)
        transformed_paragraph = paragraph

        transformed_paragraph = self._apply_transitions(transformed_paragraph, patterns, temperature)
        transformed_paragraph = self._apply_sentence_structures(transformed_paragraph, patterns, style, temperature)
        transformed_paragraph = self._apply_fillers(transformed_paragraph, patterns, temperature)

        return transformed_paragraph

    def _apply_transitions(self, paragraph: str, patterns: Dict[str, List[str]], temperature: float) -> str:
        """Apply transition phrases to the paragraph."""
        if patterns.get("transitions") and random.random() < 0.3 * temperature:
            transition = random.choice(patterns["transitions"])
            if paragraph[0].isupper():
                return f"{transition}, {paragraph[0].lower()}{paragraph[1:]}"
            return f"{transition}, {paragraph}"
        return paragraph

    def _apply_sentence_structures(self, paragraph: str, patterns: Dict[str, List[str]], style: str, temperature: float) -> str:
        """Apply sentence structure patterns to the paragraph."""
        if patterns.get("sentence_structures") and random.random() < 0.3 * temperature:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            if len(sentences) > 1:
                idx = random.randint(0, len(sentences) - 1)
                structure = random.choice(patterns["sentence_structures"])
                if structure == "question":
                    sentences[idx] = self._transform_declarative_to_question(sentences[idx], style)
                elif structure == "complex":
                    sentences[idx] = self._transform_simple_to_complex(sentences[idx], style)
                return " ".join(sentences)
        return paragraph

    def _apply_fillers(self, paragraph: str, patterns: Dict[str, List[str]], temperature: float) -> str:
        """Apply fillers to the paragraph."""
        if patterns.get("fillers") and random.random() < 0.2 * temperature:
            filler = random.choice(patterns["fillers"])
            words = paragraph.split()
            if len(words) > 5:
                insert_idx = random.randint(2, min(len(words) - 2, 6))
                words.insert(insert_idx, filler)
                return " ".join(words)
        return paragraph
    
    def _extract_patterns_from_example(self, example: str) -> Dict[str, List[str]]:
        """Extract patterns from an example text."""
        patterns = {
            "transitions": self._extract_transitions(example),
            "sentence_structures": self._identify_sentence_structures(example),
            "fillers": self._extract_fillers(example)
        }
        return patterns

    def _extract_transitions(self, example: str) -> List[str]:
        """Extract transition phrases from the example."""
        transitions_found = []
        for transition_type, transitions in self.transitions.items():
            for transition in transitions:
                if transition.lower() in example.lower():
                    transitions_found.append(transition)
        return transitions_found

    def _identify_sentence_structures(self, example: str) -> List[str]:
        """Identify sentence structures in the example."""
        # Removed unused assignment to 'sentences'
        sentences = re.split(r'(?<=[.!?])\s+', example)
        structures = []  # Initialize the list
        for sentence in sentences:
            if "?" in sentence:
                structures.append("question")
            elif "," in sentence and len(sentence.split()) > 15:
                structures.append("complex")
        return structures

    def _extract_fillers(self, example: str) -> List[str]:
        """Extract filler phrases from the example."""
        fillers_found = []
        for style, fillers in self.fillers.items():
            for filler in fillers:
                if f" {filler} " in f" {example} ":
                    fillers_found.append(filler)
        return fillers_found
    
    def _add_personal_touches(self, paragraph: str, style: str, temperature: float) -> str:
        """Add personal touches and quirks to make text more human-like."""
        transformed_paragraph = paragraph
        
        # Add personal touches based on style and temperature
        if style == "casual" and random.random() < 0.3 * temperature:
            # Add self-reference
            self_references = [
                "I think", "I believe", "in my experience", "from what I've seen",
                "personally,", "as far as I know", "I've found that", "I'd say"
            ]
            if transformed_paragraph[0].isupper():
                transformed_paragraph = f"{random.choice(self_references)} {transformed_paragraph[0].lower()}{transformed_paragraph[1:]}"
            else:
                transformed_paragraph = f"{random.choice(self_references)} {transformed_paragraph}"
        
        # Add hedging language
        if random.random() < 0.2 * temperature:
            hedging = random.choice(self.quirk_examples["hedging_language"])
            words = transformed_paragraph.split()
            if len(words) > 6:
                insert_idx = random.randint(3, min(len(words) - 3, 8))
                words.insert(insert_idx, hedging)
                transformed_paragraph = " ".join(words)
        
        return transformed_paragraph
    
    def _add_transitions_between_paragraphs(self, paragraphs: List[str], temperature: float) -> List[str]:
        """Add transitions between paragraphs."""
        if len(paragraphs) <= 1:
            return paragraphs
            
        result = [paragraphs[0]]
        
        for i in range(1, len(paragraphs)):
            paragraph = paragraphs[i]
            
            # Add transition at the beginning of paragraph with probability based on temperature
            if random.random() < 0.4 * temperature:
                # Choose a random transition type
                transition_type = random.choice(list(self.transitions.keys()))
                transition = random.choice(self.transitions[transition_type])
                
                # Add transition to beginning of paragraph
                if paragraph[0].isupper():
                    paragraph = f"{transition}, {paragraph[0].lower()}{paragraph[1:]}"
                else:
                    paragraph = f"{transition}, {paragraph}"
            
            result.append(paragraph)
        
        return result
    
    def _apply_global_transformations(self, text: str, style: str, temperature: float) -> str:
        """Apply global transformations to the entire text."""
        transformed_text = text
        
        # Add a personal introduction or conclusion
        if random.random() < 0.3 * temperature:
            if style == "casual":
                intros = [
                    "I've been thinking about this topic a lot lately. ",
                    "This is something I've had on my mind for a while. ",
                    "I wanted to share some thoughts on this. ",
                    "Here's my take on this issue. "
                ]
                transformed_text = random.choice(intros) + transformed_text
            elif style == "professional":
                intros = [
                    "Upon careful consideration of this matter, ",
                    "After analyzing the available information, ",
                    "Based on a thorough examination of the facts, ",
                    "Drawing from relevant expertise in this area, "
                ]
                transformed_text = random.choice(intros) + transformed_text
            elif style == "creative":
                intros = [
                    "Imagine a world where ",
                    "Picture this scenario: ",
                    "Let's explore this fascinating subject together. ",
                    "The beauty of this topic lies in its complexity. "
                ]
                transformed_text = random.choice(intros) + transformed_text
        
        # Add a conclusion
        if random.random() < 0.3 * temperature:
            if style == "casual":
                conclusions = [
                    " Anyway, that's just what I think about it.",
                    " That's my two cents on the matter.",
                    " Just some food for thought.",
                    " At least that's how I see it."
                ]
                transformed_text += random.choice(conclusions)
            elif style == "professional":
                conclusions = [
                    " In conclusion, these considerations merit further attention.",
                    " These points warrant careful deliberation moving forward.",
                    " This perspective offers valuable insights for future discussion.",
                    " This analysis provides a foundation for subsequent inquiry."
                ]
                transformed_text += random.choice(conclusions)
            elif style == "creative":
                conclusions = [
                    " And so, the story continues to unfold in unexpected ways.",
                    " Thus, we find ourselves at the intersection of possibility and reality.",
                    " The tapestry of this narrative weaves together countless threads of meaning.",
                    " Like stars in a constellation, these ideas form a pattern worth contemplating."
                ]
                transformed_text += random.choice(conclusions)
        
        return transformed_text
