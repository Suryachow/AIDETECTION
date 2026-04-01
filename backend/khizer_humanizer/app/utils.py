"""
Utility functions for the AI text humanizer.
"""
import re
import random
from typing import List, Dict, Any, Optional

def preprocess_text(text: str) -> str:
    """
    Preprocess text before sending it to the model.
    
    Args:
        text: The input text to preprocess
        
    Returns:
        Preprocessed text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add other preprocessing steps as needed
    return text

def postprocess_text(text: str) -> str:
    """
    Postprocess text after model generation.
    
    Args:
        text: The model output to postprocess
        
    Returns:
        Postprocessed text
    """
    # Fix common issues in generated text
    text = text.strip()
    
    # Ensure proper spacing after punctuation
    text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
    
    return text

def add_human_variations(text: str, style: str = "casual") -> str:
    """
    Add human-like variations to text based on the specified style.
    
    Args:
        text: The input text
        style: The style of variations to add (casual, professional, creative)
        
    Returns:
        Text with human-like variations
    """
    if style == "casual":
        # Add casual markers like contractions, filler words
        variations = [
            (r'\bcan not\b', "can't"),
            (r'\bwill not\b', "won't"),
            (r'\bdo not\b', "don't"),
            (r'\bI am\b', "I'm"),
            (r'\byou are\b', "you're"),
        ]
        
        # Apply random variations
        for pattern, replacement in variations:
            if random.random() > 0.3:  # 70% chance to apply each variation
                text = re.sub(pattern, replacement, text)
                
    elif style == "professional":
        # Make text more formal
        variations = [
            (r'\bdon\'t\b', "do not"),
            (r'\bcan\'t\b', "cannot"),
            (r'\bwon\'t\b', "will not"),
            (r'\bI think\b', "In my assessment"),
            (r'\bI believe\b', "I would posit that"),
        ]
        
        # Apply random variations
        for pattern, replacement in variations:
            if random.random() > 0.3:
                text = re.sub(pattern, replacement, text)
                
    elif style == "creative":
        # No specific replacements for creative style
        # This would be handled by the model itself
        pass
    
    return text

def calculate_humanness_score(text: str) -> float:
    """
    Calculate a humanness score for the given text.
    
    Args:
        text: The text to evaluate
        
    Returns:
        A score between 0.0 and 1.0 where higher is more human-like
    """
    # This is a placeholder implementation
    # A real implementation would use more sophisticated metrics
    
    # Simple heuristics for demonstration
    score = 0.5  # Start with neutral score
    
    # Check for contractions (more human-like)
    contraction_count = len(re.findall(r'\'[ts]|n\'t', text))
    score += min(contraction_count * 0.02, 0.1)
    
    # Check for varied sentence length (more human-like)
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 1:
        lengths = [len(s.split()) for s in sentences if s.strip()]
        variance = sum((l - sum(lengths)/len(lengths))**2 for l in lengths) / len(lengths)
        score += min(variance * 0.01, 0.1)
    
    # Check for repetition (less human-like)
    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words) if words else 0
    score += unique_ratio * 0.2
    
    # Ensure score is between 0 and 1
    return max(0.0, min(score, 1.0))
