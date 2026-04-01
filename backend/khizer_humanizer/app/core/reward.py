"""
Reward model for evaluating the quality of humanized text.
"""
import logging
import random
from typing import Dict, Any, List, Optional

# Set up logging
logger = logging.getLogger(__name__)

class RewardModel:
    """
    Model for evaluating the quality and humanness of generated text.
    """
    
    def __init__(self):
        """Initialize the reward model."""
        logger.info("Reward model initialized")
    
    def score(self, text: str) -> float:
        """
        Score the humanness of the given text.
        
        Args:
            text: The text to evaluate
            
        Returns:
            A score between 0.0 and 1.0 where higher is more human-like
        """
        # In a real implementation, this would use a trained model
        # For this example, we'll use a simple heuristic approach
        
        # Calculate basic metrics
        avg_word_length = sum(len(word) for word in text.split()) / max(1, len(text.split()))
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(text) / max(1, sentence_count)
        
        # Penalize very long or very short words/sentences
        word_length_score = max(0, 1 - abs(avg_word_length - 5) / 5)
        sentence_length_score = max(0, 1 - abs(avg_sentence_length - 80) / 80)
        
        # Check for human markers
        has_contractions = "'" in text
        has_fillers = any(filler in text.lower() for filler in ["well", "like", "you know", "actually", "basically"])
        
        # Calculate final score
        base_score = (word_length_score + sentence_length_score) / 2
        human_markers_score = 0.2 if has_contractions else 0 + 0.2 if has_fillers else 0
        
        final_score = min(1.0, base_score + human_markers_score)
        
        logger.info(f"Calculated humanness score: {final_score}")
        return final_score
    
    def compare(self, original_text: str, humanized_text: str) -> Dict[str, Any]:
        """
        Compare original and humanized text to evaluate improvement.
        
        Args:
            original_text: The original AI-generated text
            humanized_text: The humanized version
            
        Returns:
            Dictionary with comparison metrics
        """
        original_score = self.score(original_text)
        humanized_score = self.score(humanized_text)
        
        improvement = humanized_score - original_score
        
        return {
            "original_score": original_score,
            "humanized_score": humanized_score,
            "improvement": improvement,
            "percent_improvement": (improvement / max(0.01, original_score)) * 100
        }
    
    def batch_score(self, texts: List[str]) -> List[float]:
        """
        Score multiple texts in batch.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            List of scores
        """
        return [self.score(text) for text in texts]
