import os
import re
import random
import collections
import numpy as np
import json
import nltk
from typing import List, Optional
from nltk.corpus import wordnet
from groq import Groq
from core.config import settings
from agent.lexical_agent import LexicalAgent
from agent.humanizer_agent import HumanizerAgent
from database.mongo_client import LexicalDatabase

# Configuration
LOCAL_MODEL_PATH = "models/roberta_detector"
REMOTE_MODEL_NAME = "roberta-base-openai-detector"

class AIService:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.detector_model = None
        self.model_type = "Unknown"
        self.humanizer_model = None
        self.humanizer_tokenizer = None
        
        self.bert_model = None
        self.bert_tokenizer = None
        self.redis_client = None
        
        # Paraphraser & Synonyms
        self.groq_client = None
        self.paraphrase_tokenizer = None
        self.paraphrase_model = None
        self.similarity_model = None
        self.nlp = None
        
        # New Advanced Agents
        self.lexical_agent = None
        self.humanizer_agent = None
        self.lexical_db = None
        self._suggestions_cache = collections.OrderedDict()
        
        self._init_nltk()

    def _init_nltk(self):
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('omw-1.4')

    def _get_groq_client(self):
        if self.groq_client is None:
            self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        return self.groq_client

    def _get_similarity_model(self):
        if self.similarity_model is None:
            from sentence_transformers import SentenceTransformer
            self.similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return self.similarity_model

    def _get_spacy(self):
        if self.nlp is None:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Downloading spacy model...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        return self.nlp

    def _load_paraphrase_model(self):
        if self.paraphrase_model is None:
            import torch
            from transformers import PegasusForConditionalGeneration, PegasusTokenizer
            model_name = "tuner007/pegasus_paraphrase"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Loading local model {model_name}...")
            self.paraphrase_tokenizer = PegasusTokenizer.from_pretrained(model_name)
            self.paraphrase_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

    def _get_lexical_agent(self):
        if self.lexical_agent is None:
            self.lexical_agent = LexicalAgent()
        return self.lexical_agent

    def _get_humanizer_agent(self):
        if self.humanizer_agent is None:
            db = self._get_lexical_db()
            self.humanizer_agent = HumanizerAgent(db=db)
        return self.humanizer_agent

    def _get_lexical_db(self):
        if self.lexical_db is None:
            try:
                self.lexical_db = LexicalDatabase()
            except:
                self.lexical_db = None
        return self.lexical_db

    async def _post_process_lexical(self, text: str):
        """Analyze text and store in DB in background."""
        if not text: return
        try:
            # Strip HTML for analysis
            clean_text = re.sub(r'<[^>]*>', ' ', text)
            agent = self._get_lexical_agent()
            db = self._get_lexical_db()
            if agent and db:
                entries = agent.process_sentence(clean_text)
                db.insert_multiple_entries(entries)
        except Exception as e:
            print(f"Lexical post-process failed: {e}")

    def _get_llm(self):
        if self.llm is None:
             from langchain_groq import ChatGroq
             self.llm = ChatGroq(
                temperature=0.7,
                model_name=settings.GROQ_MODEL_NAME,
                groq_api_key=settings.GROQ_API_KEY
            )
        return self.llm

    def load_model(self):
        """Lazy load the AI detector model."""
        if self.detector_model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import gc
        
        search_candidates = [
            ("Hello-SimpleAI/chatgpt-detector-roberta", "Modern HC3 ChatGPT Detector (Highly Accurate)"),
            ("openai-community/roberta-base-openai-detector", "Standard OpenAI Detector (Stable Base)"),
            (LOCAL_MODEL_PATH, "Local Fine-Tuned Model")
        ]

        print("[START] Initializing AI Detection Engine...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for model_name, type_desc in search_candidates:
            if model_name == LOCAL_MODEL_PATH and not os.path.exists(LOCAL_MODEL_PATH):
                continue
                
            try:
                print(f"[FETCH] Loading AI Detector: {model_name}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.detector_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.detector_model.eval()
                if torch.cuda.is_available():
                    self.detector_model.to('cuda')
                self.model_type = type_desc
                print(f"[OK] Loaded: {type_desc}")
                return
            except Exception as e:
                print(f"[FAIL] {model_name}: {e}")
                continue

    def apply_heuristics(self, text, ai_score, human_score):
        reasons = []
        lower_text = text.lower()
        
        # 1. Technical Structure
        code_symbols = ["```", "def ", "import ", "python ", "npm ", "{", "}"]
        if any(s in text for s in code_symbols):
            ai_score = min(100, ai_score + 40)
            reasons.append("Technical structure detected")

        # 2. Typos (Human trait)
        common_typos = [r"\biam\b", r"\bdont\b", r"\bcant\b", r"\bu\b", r"\bir\b"]
        typo_count = sum(1 for p in common_typos if re.search(p, lower_text))
        if typo_count > 0:
            ai_score = max(0, ai_score - (8 * typo_count))
            reasons.append("Human-like informal patterns/typos")

        # 3. Transitions
        ai_transitions = ["however,", "moreover,", "furthermore,", "consequently,", "initially,", "additionally,"]
        t_count = sum(1 for t in ai_transitions if t in lower_text)
        if t_count > 2:
            ai_score = min(100, ai_score + (10 * t_count))
            reasons.append("AI-style transitional flow")

        # 4. Human-like conversational fillers
        human_markers = [r"\bactually\b", r"\bwell\b", r"\byou see\b", r"\bi mean\b", r"\bbasically\b", r"\bjust\b", r"\bi think\b", r"\bi feel\b", r"\bto be honest\b", r"\bhonestly\b", r"\bbasically\b", r"\bi guess\b"]
        hm_count = sum(1 for p in human_markers if re.search(p, lower_text))
        if hm_count > 0:
            ai_score = max(0, ai_score - (6 * hm_count))
            reasons.append(f"Human conversational flow ({hm_count} markers)")
            
        # 5. Length Variation & Punctuation (Human markers)
        if "!" in text or "?" in text:
            ai_score = max(0, ai_score - 20)
            reasons.append("Human punctuation patterns")

        # 6. Fragment Detection (Human trait)
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if len(s.strip()) > 1]
        if any(len(s.split()) < 5 for s in sentences):
            ai_score = max(0, ai_score - 25)
            reasons.append("Natural sentence fragments")

        # 7. Sentence-length variance (higher variance is more human-like)
        sentence_lengths = [len(s.split()) for s in sentences if s]
        if len(sentence_lengths) >= 3:
            avg_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - avg_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            if variance >= 30:
                ai_score = max(0, ai_score - 12)
                reasons.append("Natural sentence-length variation")

        # 8. Lexical diversity (very low diversity can be machine-like)
        tokens = re.findall(r"[a-zA-Z']+", lower_text)
        if len(tokens) >= 60:
            unique_ratio = len(set(tokens)) / len(tokens)
            if unique_ratio >= 0.55:
                ai_score = max(0, ai_score - 8)
                reasons.append("Healthy lexical diversity")

        # 9. Keep outputs away from extreme certainty in UI scoring.
        if ai_score > 94:
            ai_score = 94.0
            reasons.append("Confidence capped to reduce false positives")
        if ai_score < 5:
            ai_score = 5.0

        return round(ai_score, 1), round(100 - ai_score, 1), reasons

    def _resolve_ai_class_index(self):
        """Infer which classifier label corresponds to AI-generated text."""
        labels = getattr(getattr(self, 'detector_model', None), 'config', None)
        labels = getattr(labels, 'id2label', {}) if labels else {}
        if not labels:
            return 1

        scored = {}
        for idx, label in labels.items():
            text = str(label).lower()
            score = 0
            if any(k in text for k in ["ai", "fake", "generated", "bot", "machine"]):
                score += 3
            if any(k in text for k in ["human", "real", "non-ai", "organic"]):
                score -= 3
            scored[int(idx)] = score

        best_idx = max(scored, key=scored.get)
        return best_idx if scored[best_idx] > 0 else 1

    def _detect_sync(self, text: str) -> dict:
        import torch
        if not self.detector_model: self.load_model()
        if not self.detector_model: return {"score": 0, "error": "Model load fail"}

        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        if len(clean_text) < 50: return {"score": 0, "explanation": "Text too short."}

        inputs = self.tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available(): inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            ai_idx = self._resolve_ai_class_index()
            ai_idx = max(0, min(ai_idx, probs.shape[-1] - 1))
            ai_score = probs[0][ai_idx].item() * 100

        # Mild score calibration to avoid extreme confidence spikes on neutral prose.
        ai_score = 50 + ((ai_score - 50) * 0.75)

        ai_score, human_score, reasons = self.apply_heuristics(clean_text, ai_score, 100 - ai_score)
        
        explanation = "Human-written" if ai_score < 50 else "AI-generated"
        if reasons:
            explanation += "<ul>" + "".join([f"<li>{r}</li>" for r in reasons]) + "</ul>"

        return {
            "score": ai_score,
            "explanation": explanation,
            "highlighted_content": text
        }

    async def detect_ai_content(self, text: str) -> dict:
        import asyncio
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_sync, text)

    # --- WRITING SUITE ---

    async def generate_text(self, prompt: str, context: str = "", length: str = "Standard", format: str = "Paragraph") -> str:
        length_map = {
            "Quick": "about 60-90 words",
            "Standard": "about 120-180 words",
            "Comprehensive": "about 220-320 words",
        }
        format_map = {
            "Paragraph": "Return 1-2 cohesive paragraphs.",
            "Bullet Points": "Return concise bullet points only.",
            "LinkedIn Post": "Return a polished LinkedIn-style post with a strong hook and concise body.",
            "Email": "Return a professional email body only (no subject line unless asked).",
        }

        target_length = length_map.get(length or "Standard", length_map["Standard"])
        target_format = format_map.get(format or "Paragraph", format_map["Paragraph"])

        system_instruction = (
            "You are a professional writing engine, not a chatbot. "
            "Return clean publish-ready prose only."
            "\n\nHard rules:"
            "\n1. Never use chat phrasing such as 'Sure', 'Here is', 'I can help', 'Let me know'."
            "\n2. Do not mention being an AI or assistant."
            "\n3. Do not include prefaces, disclaimers, or conclusions unless explicitly asked."
            "\n4. Write directly in the requested format and topic."
            "\n5. Keep output concise, coherent, and natural."
            f"\n6. Target length: {target_length}."
            f"\n7. Output format rule: {target_format}"
        )

        user_prompt = (
            f"Task: {prompt}\n"
            f"Context: {context}\n"
            f"Length: {length}\n"
            f"Format: {format}\n"
            "Output: written content only."
        )

        messages = [
            ("system", system_instruction),
            ("human", user_prompt)
        ]
        try:
            res = self._get_llm().invoke(messages)
            text = (res.content or "").strip()

            # Strip common chatty prefixes if model still emits them.
            text = re.sub(r"^(sure|certainly|absolutely|here(?:'s| is)|of course)[\s,:-]+", "", text, flags=re.IGNORECASE)
            text = re.sub(r"^(i can help|let me|as an ai)[^\n]*\n?", "", text, flags=re.IGNORECASE)
            text = text.strip().strip('"')

            # Enforce format guards if the model ignores output-shape instructions.
            if format == "Bullet Points":
                lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
                has_bullets = any(ln.startswith('-') or ln.startswith('•') for ln in lines)
                if not has_bullets:
                    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                    if not sentences:
                        sentences = [text]
                    text = "\n".join([f"- {s}" for s in sentences[:8]])
            elif format == "Email":
                text = re.sub(r"^\s*subject\s*:\s*.*\n?", "", text, flags=re.IGNORECASE).strip()

            return text
        except Exception as e: return f"Error: {e}"

    async def paraphrase(self, text: str, mode: str = "Standard", num_results: int = 1, use_groq: bool = True) -> str:
        res_text = ""
        # Paraphrase should stay independent from the humanize pipeline.
        if use_groq:
            try:
                variations = await self.paraphrase_groq(text, mode, num_results=max(1, num_results))
                res_text = variations[0] if num_results == 1 else variations
            except Exception as e:
                print(f"Groq paraphrase failed: {e}")
                pass
        
        if not res_text:
            res = await self.paraphrase_local(text, mode, num_results)
            res_text = res[0] if num_results == 1 else res
            
        # Post-process to populate DB
        import asyncio
        if isinstance(res_text, str):
            asyncio.create_task(self._post_process_lexical(res_text))
        elif isinstance(res_text, list):
            for t in res_text:
                asyncio.create_task(self._post_process_lexical(t))
            
        return res_text

    async def get_rephrase_recommendations(self, text: str, mode: str = "Standard") -> List[str]:
        """Core method for Smart Rewrite recommendations."""
        variations = await self.paraphrase_groq(text, mode, num_results=3)

        return variations

    async def paraphrase_groq(self, text: str, mode: str = "Standard", num_results: int = 3) -> List[str]:
        mode_style = {
            "Standard":  "Clear, natural, and conversational. Write like a thoughtful human blogger.",
            "Humanize":  "Ultra-Humanized. Shatter every robotic pattern. Highest possible bypass probability.",
            "Creative":  "Engaging, vivid, and creative. Tell it like a story with personality.",
            "Formal":    "Professional and formal. Precise language, but still warm and human – not stiff.",
            "Simple":    "Plain and simple. Short words. Short sentences. Easy to read.",
            "Shorten":   "Concise. Every word must earn its place. Remove all filler.",
            "Expand":    "Detailed and rich. Add context, examples, and natural asides.",
        }
        style = mode_style.get(mode, mode_style["Standard"])
        client = self._get_groq_client()
        
        instruction = (
            f"You are a skilled human writer. Rewrite the input EXACTLY in this style: '{style}'.\n\n"
            "=== MANDATORY ANTI-AI BYPASS RULES (NEVER SKIP THESE) ===\n"
            "1. BANNED AI WORDS: Never use — 'delve', 'ensure', 'comprehensive', 'synergy', 'transform', "
            "'unlock', 'enhance', 'meticulous', 'vital', 'in conclusion', 'to summarize', 'it's worth noting'.\n"
            "2. BURSTINESS PATTERN: Vary sentence length constantly. Short. Then a longer one that adds detail. Then medium. Never two of the same length in a row.\n"
            "3. HUMAN MARKERS: You MUST include at least 2 of these words naturally: "
            "'actually', 'well', 'basically', 'just', 'honestly', 'I think', 'you know', 'sort of'.\n"
            "4. BAN ROBOT TRANSITIONS: Never start a sentence with 'Furthermore', 'Moreover', 'Additionally', "
            "'However', 'In conclusion'. Use human transitions: 'So', 'Plus', 'But', 'And', 'Then'.\n"
            "5. IMPERFECT HUMANS: Occasionally use contractions (it's, don't, can't, they're). "
            "Put the occasional exclamation or question to show personality.\n"
            "6. VOCAB DIVERSITY: Never use the same word more than twice. Rephrase to use synonyms.\n\n"
            "=== CONTENT INTEGRITY RULES (NEVER SKIP THESE) ===\n"
            "1. NEVER invent new facts, data, or details not in the original.\n"
            "2. NEVER change proper nouns, brand names, numbers, or technical terms.\n"
            "3. MIRROR the input's HTML structure EXACTLY (paragraphs, lists, tags, bold). "
            "If input has 3 bullet points, you MUST output 3 rephrased bullet points.\n"
            "4. Rephrase EVERY sentence. Do not skip or truncate any part.\n\n"
            f"=== OUTPUT FORMAT ===\n"
            f"Return ONLY a valid JSON object like: {{\"variations\": [\"...variation 1...\", \"...variation 2...\", \"...variation 3...\"]}}\n"
            f"You MUST return exactly {num_results} variation(s). Each must be a complete, full-length rewrite."
        )
        
        try:
            print(f"[PARAPHRASE] Mode={mode} | {len(text)} chars | Preview: {text[:80]}...")
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                max_tokens=4096,
                temperature=0.85,
                top_p=0.9,
                presence_penalty=1.0,
                frequency_penalty=0.3,
            )
            content = completion.choices[0].message.content
            print(f"[PARAPHRASE] Response length: {len(content)}")
            data = json.loads(content)
            variations = data.get("variations", [])
            # Robustly handle LLMs that return [{"text": "..."}] instead of ["..."]
            results = [v.get("text", str(v)) if isinstance(v, dict) else str(v) for v in variations]
            # Fallback: if we got empty results, return the original
            return results if results else [text]
        except Exception as e:
            print(f"[PARAPHRASE] Groq error: {e}")
            return [text]


    async def paraphrase_local(self, text: str, mode: str = "Standard", num_results: int = 1) -> List[str]:
        self._load_paraphrase_model()
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.paraphrase_tokenizer(text, truncation=True, padding='longest', return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.paraphrase_model.generate(**inputs, num_beams=5, num_return_sequences=num_results)
        return self.paraphrase_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    async def get_word_suggestions(self, sentence: str, word: str) -> List[str]:
        """Ultra-fast synonym lookup with memory caching, DB priority and WordNet fallback."""
        cleaned_word = word.lower().strip()
        
        # 0. Check cache first (Immediate response)
        if cleaned_word in self._suggestions_cache:
            # Move to end (LRU)
            self._suggestions_cache.move_to_end(cleaned_word)
            return self._suggestions_cache[cleaned_word]

        # Immediate return for stop words or numbers
        if len(cleaned_word) <= 2 or cleaned_word in ["and", "the", "for", "with", "from", "that", "this"]:
            return []
        if re.match(r'^[\d,.\-/]+$', cleaned_word):
            return []

        print(f"DEBUG: Fast Lookup for '{cleaned_word}'")
        suggestions = []
        
        # 1. DB Lookup
        db = self._get_lexical_db()
        if db:
            entry = db.find_word(cleaned_word)
            if entry and entry.get("synonyms"):
                suggestions.extend([s for s in entry["synonyms"] if s.lower() != cleaned_word])

        # 2. WordNet Fallback
        if len(suggestions) < 5:
            # Quick check: if word ends in 's', 'ed', 'ing', try lemmatization. 
            lemma = cleaned_word
            if any(cleaned_word.endswith(suffix) for suffix in ['s', 'ed', 'ing', 'ly']):
                try:
                    nlp = self._get_spacy()
                    doc = nlp(cleaned_word)
                    lemma = doc[0].lemma_ if len(doc) > 0 else cleaned_word
                except: pass
            
            for syn in wordnet.synsets(lemma):
                for l in syn.lemmas():
                    name = l.name().replace('_', ' ')
                    if name.lower() != lemma.lower() and name.lower() != cleaned_word and name not in suggestions:
                        suggestions.append(name)
                if len(suggestions) >= 12: break
        
        results = suggestions[:10]
        # Store in cache
        self._suggestions_cache[cleaned_word] = results
        if len(self._suggestions_cache) > 2000:
            self._suggestions_cache.popitem(last=False)
            
        return results

    def _get_bert_model(self):
        if self.bert_model is None:
            from transformers import BertTokenizer, BertForMaskedLM
            import torch
            print("[FETCH] Loading BERT...")
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            if torch.cuda.is_available(): self.bert_model.to('cuda')
            self.bert_model.eval()
        return self.bert_model, self.bert_tokenizer

    def _get_redis_client(self):
        if self.redis_client is None:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=settings.REDIS_HOST, 
                    port=settings.REDIS_PORT, 
                    decode_responses=True,
                    socket_connect_timeout=0.5
                )
            except: self.redis_client = None
        return self.redis_client

    async def summarize(self, text: str, length: str = "Medium") -> str:
        prompt = f"Summarize this text ({length} length). Return HTML format."
        messages = [("system", "Summarizer AI"), ("human", f"{prompt}\n\n{text}")]
        try:
            res = self._get_llm().invoke(messages)
            res_text = res.content
            import asyncio
            asyncio.create_task(self._post_process_lexical(res_text))
            return res_text
        except: return text

    async def fix_grammar(self, text: str) -> str:
        messages = [("system", "Grammar Fixer. Maintain HTML tags."), ("human", text)]
        try:
            res = self._get_llm().invoke(messages)
            res_text = res.content
            import asyncio
            asyncio.create_task(self._post_process_lexical(res_text))
            return res_text
        except: return text

    async def humanize(
        self,
        text: str,
        passes: int = 1,
        mode: str = "universal",
        intensity: Optional[int] = None,
        debug: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = True,
    ) -> dict:
        import asyncio
        import httpx
        
        plain_text = re.sub(r'<[^>]+>', ' ', text)
        plain_text = re.sub(r'&nbsp;', ' ', plain_text)
        plain_text = re.sub(r'\s+', ' ', plain_text).strip()

        resolved_intensity = intensity if intensity is not None else 75
        
        h_text = plain_text
        diagnostics = None
        try:
            payload = {
                'text': plain_text,
                'intensity': resolved_intensity,
                'mode': mode,
                'debug': debug,
                'deterministic': deterministic,
            }
            if seed is not None:
                payload['seed'] = seed

            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    'http://localhost:3001/api/humanize',
                    json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                h_text = data.get('humanized', plain_text)
                diagnostics = data.get('diagnostics')
                print(f"[HUMANIZE] Node.js engine success | context={data.get('context', '?')} | {len(h_text)} chars")
        except Exception as e:
            print(f"[HUMANIZE] Falling back to local agent: {e}")
            try:
                agent = self._get_humanizer_agent()
                for _ in range(passes):
                    h_text = agent.humanize_text(h_text)
            except Exception as e3:
                h_text = plain_text
        
        paragraphs = [p.strip() for p in re.split(r'\n+', h_text) if p.strip()]
        if paragraphs:
            html_result = ''.join(f'<p>{p}</p>' for p in paragraphs)
        else:
            html_result = f'<p>{h_text}</p>'
        
        asyncio.create_task(self._post_process_lexical(html_result))
        
        return {
            "result": html_result,
            "diagnostics": diagnostics,
        }

    async def get_synonyms(self, word: str, context: Optional[str] = None):
        if not word or not word.strip():
            return []

        system_instruction = (
            "You are a thesaurus and contextual rephrasing assistant.\n"
            "Given a word/phrase and its context, provide 5-8 natural alternatives.\n"
            "Return ONLY a JSON list of strings."
        )
        
        user_prompt = f"Word/Phrase: '{word}'"
        if context:
            user_prompt += f"\nContext: '{context}'"

        client = self._get_groq_client()
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            content = json.loads(completion.choices[0].message.content)
            synonyms = []
            if isinstance(content, list): 
                synonyms = content
            elif isinstance(content, dict):
                for key in ["synonyms", "alternatives", "results", "words"]:
                    if key in content:
                        synonyms = content[key]
                        break
            
            synonyms = [s for s in synonyms if isinstance(s, str) and s.lower().strip() != word.lower().strip()]
            return synonyms[:10]
        except:
            from nltk.corpus import wordnet
            raw_synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    name = l.name().replace('_', ' ')
                    if name.lower() != word.lower():
                        raw_synonyms.append(name)
            return list(set(raw_synonyms))[:8]

    async def analyze_text_for_rephrasing(self, text: str) -> str:
        if not text.strip():
            return text

        system_instruction = (
            "You are an expert editor. Wrap robotic phrases in <span class='rephrased-word'> tags. Do not rewrite. Return HTML."
        )
        
        client = self._get_groq_client()
        try:
            completion = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": text}
                ],
                max_tokens=4096,
                temperature=0.8
            )
            return completion.choices[0].message.content.strip()
        except:
            return text

ai_service = AIService()
