import re
import random
from typing import List, Dict, Any, Tuple, Optional
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticRewriter:
    def __init__(self, nlp):
        self.nlp = nlp
        self.ai_patterns = {
            r'\bit is important to note that\b': ['notably,', 'importantly,', 'note that', ''],
            r'\bit is worth noting that\b': ['notably,', 'note that', ''],
            r'\bit should be noted that\b': ['note that', 'notably,', ''],
            r'\bit is interesting to note that\b': ['interestingly,', 'note that', ''],
            r'\bit is evident that\b': ['clearly,', 'evidently,', 'obviously,'],
            r'\bit is clear that\b': ['clearly,', 'obviously,'],
            r'\bit can be seen that\b': ['we can see', 'this shows', 'clearly,'],
            r'\bit is crucial to understand that\b': ['crucially,', 'importantly,', ''],
            r'\bit must be emphasized that\b': ['importantly,', 'crucially,', ''],
            r'\bimproves? the availability of\b': ['makes {} more accessible', 'enhances access to {}', 'improves access to {}'],
            r'\bincreases? the availability of\b': ['makes {} more available', 'provides greater access to {}', 'improves access to {}'],
            r'\bprovides? enhancement to\b': ['enhances {}', 'improves {}', 'makes {} better'],
            r'\boffers? improvement in\b': ['improves {}', 'makes {} better', 'enhances {}'],
            r'\bfacilitates? the optimization of\b': ['optimizes {}', 'improves {}', 'enhances {}'],
            r'\bhelps? to improve\b': ['improves', 'enhances', 'makes better'],
            r'\bhelps? to optimize\b': ['optimizes', 'improves', 'refines'],
            r'\bassists? in improving\b': ['helps improve', 'improves', 'enhances'],
            r'\bis able to provide\b': ['provides', 'offers', 'delivers', 'can provide'],
            r'\bhas the capability to\b': ['can', 'is able to', 'has the ability to'],
            r'\bhas the ability to\b': ['can', 'is able to'],
            r'\bmakes it possible to\b': ['enables', 'allows', 'lets us'],
            r'\bserves? to\b': ['helps', 'works to', 'aims to'],
            r'\btends? to\b': ['usually', 'often', 'typically'],
            r'\bthe utilization of\b': ['using', 'the use of', 'using'],
            r'\bthe implementation of\b': ['implementing', 'using', 'applying'],
            r'\bthe optimization of\b': ['optimizing', 'improving'],
            r'\bthe enhancement of\b': ['enhancing', 'improving'],
            r'\bthe availability of\b': ['access to', 'availability of', 'having'],
            r'\bthe application of\b': ['applying', 'using'],
            r'\bthe demonstration of\b': ['demonstrating', 'showing'],
            r'\bthe exploration of\b': ['exploring', 'examining'],
            r'\bthe examination of\b': ['examining', 'looking at', 'studying'],
            r'\bthe analysis of\b': ['analyzing', 'examining', 'studying'],
            r'\bthe evaluation of\b': ['evaluating', 'assessing'],
            r'\bthe identification of\b': ['identifying', 'finding', 'spotting'],
            r'\bthe comprehension of\b': ['understanding', 'grasping'],
            r'\bthe realization of\b': ['realizing', 'achieving', 'understanding'],
            r'\bthe performance of\b': ['performing', 'doing', 'carrying out'],
            r'\ba broad range of\b': ['various', 'many different', 'diverse', 'multiple'],
            r'\ba wide variety of\b': ['many', 'various', 'diverse', 'multiple'],
            r'\ba significant number of\b': ['many', 'numerous', 'several', 'a lot of'],
            r'\ba large number of\b': ['many', 'numerous', 'lots of'],
            r'\ba vast array of\b': ['many', 'numerous', 'various'],
            r'\bthe vast majority of\b': ['most', 'nearly all', 'the majority of'],
            r'\bthe majority of\b': ['most', 'many'],
            r'\bthe entirety of\b': ['all', 'the entire', 'the whole'],
            r'\bin order to\b': ['to'],
            r'\bfor the purpose of\b': ['to', 'for'],
            r'\bwith the objective of\b': ['to', 'aiming to'],
            r'\bwith the intention of\b': ['to', 'intending to'],
            r'\bin an effort to\b': ['to', 'trying to'],
            r'\bwith the goal of\b': ['to', 'aiming to'],
            r'\bfor the sake of\b': ['for'],
            r'\bdue to the fact that\b': ['because', 'since', 'as'],
            r'\bin light of the fact that\b': ['since', 'because', 'as'],
            r'\bin view of the fact that\b': ['since', 'because'],
            r'\bowing to the fact that\b': ['because', 'since'],
            r'\bfor the reason that\b': ['because', 'since'],
            r'\bin the event that\b': ['if', 'when'],
            r'\bunder circumstances in which\b': ['when', 'if'],
            r'\bin case of\b': ['if'],
            r'\bin the case of\b': ['for', 'with'],
            r'\bwith regard to\b': ['regarding', 'about', 'concerning'],
            r'\bwith respect to\b': ['regarding', 'about', 'concerning'],
            r'\bin relation to\b': ['regarding', 'about', 'concerning'],
            r'\bin reference to\b': ['regarding', 'about'],
            r'\bat this point in time\b': ['now', 'currently', 'at present'],
            r'\bat the present time\b': ['now', 'currently'],
            r'\bat the current time\b': ['now', 'currently'],
            r'\bat this moment\b': ['now', 'currently'],
            r'\bin the near future\b': ['soon', 'shortly', 'before long'],
            r'\bat a later point\b': ['later', 'afterwards'],
            r'\bat a later date\b': ['later', 'eventually'],
            r'\bat some point in the future\b': ['eventually', 'later', 'someday'],
            r'\bprior to\b': ['before'],
            r'\bsubsequent to\b': ['after', 'following'],
            r'\bduring the course of\b': ['during', 'while'],
            r'\bfor the duration of\b': ['during', 'throughout'],
            r'\bin the process of\b': ['while', 'during'],
            r'\bin the midst of\b': ['during', 'amid', 'while'],
            r'\bon a daily basis\b': ['daily', 'every day'],
            r'\bon a regular basis\b': ['regularly', 'often'],
            r'\bon an annual basis\b': ['annually', 'yearly', 'each year'],
            r'\bfrom time to time\b': ['sometimes', 'occasionally'],
            r'\bin the majority of cases\b': ['usually', 'often', 'typically'],
            r'\bin most cases\b': ['usually', 'typically', 'often'],
            r'\bin many instances\b': ['often', 'frequently'],
            r'\bas a matter of fact\b': ['actually', 'in fact'],
            r'\bas a general rule\b': ['usually', 'generally', 'typically'],
            r'\bby and large\b': ['generally', 'mostly', 'overall'],
            r'\bvery important\b': ['crucial', 'vital', 'critical', 'essential', 'key'],
            r'\bvery significant\b': ['highly significant', 'major', 'substantial', 'considerable'],
            r'\bvery useful\b': ['highly useful', 'valuable', 'beneficial', 'helpful'],
            r'\bvery effective\b': ['highly effective', 'powerful', 'impactful'],
            r'\bvery difficult\b': ['challenging', 'tough', 'complex', 'demanding', 'hard'],
            r'\bvery easy\b': ['straightforward', 'simple', 'uncomplicated', 'easy'],
            r'\bvery interesting\b': ['fascinating', 'intriguing', 'compelling'],
            r'\bvery different\b': ['quite different', 'vastly different', 'completely different'],
            r'\bvery similar\b': ['quite similar', 'nearly identical', 'almost the same'],
            r'\bvery good\b': ['excellent', 'great', 'outstanding', 'superb'],
            r'\bvery bad\b': ['terrible', 'awful', 'poor', 'dreadful'],
            r'\bvery large\b': ['huge', 'massive', 'enormous', 'substantial'],
            r'\bvery small\b': ['tiny', 'minuscule', 'minimal', 'slight'],
            r'\bextremely important\b': ['critical', 'vital', 'essential', 'crucial'],
            r'\bextremely difficult\b': ['incredibly challenging', 'very tough', 'exceptionally hard'],
            r'\bhighly effective\b': ['remarkably successful', 'very effective', 'quite powerful'],
            r'\bquite significant\b': ['fairly significant', 'notable', 'substantial'],
        }
        self.phrase_rewrites = {
            'it is important to note': ['notably', 'importantly', 'note that', 'worth noting'],
            'it should be emphasized': ['importantly', 'notably', 'crucially'],
            'as previously mentioned': ['as noted', 'as discussed', 'as stated earlier'],
            'as mentioned earlier': ['as noted', 'earlier', 'previously'],
            'in conclusion': ['to sum up', 'in summary', 'overall', 'in the end'],
            'the results indicate': ['results show', 'findings reveal', 'data suggests'],
            'research has shown': ['studies show', 'research shows', 'evidence reveals'],
            'according to research': ['research shows', 'studies suggest', 'evidence indicates'],
            'furthermore': ['also', 'additionally', 'moreover', 'plus', 'besides'],
            'moreover': ['also', 'besides', 'additionally', 'plus'],
            'in addition': ['also', 'additionally', 'plus', 'besides'],
            'nevertheless': ['however', 'yet', 'still', 'even so', 'but'],
            'however': ['but', 'yet', 'though', 'still'],
            'consequently': ['so', 'therefore', 'as a result', 'thus', 'hence'],
            'therefore': ['so', 'thus', 'hence', 'as a result'],
            'it is possible to': ['you can', 'we can', 'it\'s possible to', 'can'],
            'it is necessary to': ['need to', 'must', 'should', 'have to'],
            'market leader': ['industry leader', 'top company', 'leading player'],
            'innovative solutions': ['new solutions', 'fresh approaches', 'creative solutions'],
            'positioned to': ['ready to', 'able to', 'set to', 'prepared to'],
            'committed to': ['dedicated to', 'focused on', 'devoted to'],
        }

    def rewrite_sentence(self, sentence: str) -> str:
        result = sentence
        for pattern, replacements in self.ai_patterns.items():
            if re.search(pattern, result, re.IGNORECASE):
                replacement = random.choice(replacements)
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE, count=1)
        
        for phrase, replacements in self.phrase_rewrites.items():
            if phrase.lower() in result.lower():
                replacement = random.choice(replacements)
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                def replace_preserve_case(match):
                    original = match.group(0)
                    if original[0].isupper():
                        return replacement.capitalize()
                    return replacement
                result = pattern.sub(replace_preserve_case, result, count=1)
        return result

    def inject_burstiness(self, sentences: List[str]) -> List[str]:
        if len(sentences) <= 1:
            return sentences
        result = []
        i = 0
        while i < len(sentences):
            current = sentences[i]
            current_words = len(current.split())
            if (i + 1 < len(sentences) and 
                current_words < 8 and 
                len(sentences[i + 1].split()) < 10 and 
                random.random() < 0.3):
                combined = current.rstrip('.!?') + ', and ' + sentences[i + 1][0].lower() + sentences[i + 1][1:]
                result.append(combined)
                i += 2
            elif current_words > 20 and random.random() < 0.25:
                if ' and ' in current:
                    parts = current.split(' and ', 1)
                    result.append(parts[0] + '.')
                    result.append(parts[1].strip().capitalize())
                else:
                    result.append(current)
                i += 1
            else:
                result.append(current)
                i += 1
        return result

class HumanizerAgent:
    def __init__(self, db=None):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import spacy.cli
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.rewriter = SemanticRewriter(self.nlp)
        self.db = db

    def humanize_text(self, text: str, replacement_rate: float = 0.75, mode: str = "Standard") -> str:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        rewritten = [self.rewriter.rewrite_sentence(s) for s in sentences]
        bursted = self.rewriter.inject_burstiness(rewritten)
        return " ".join(bursted)
