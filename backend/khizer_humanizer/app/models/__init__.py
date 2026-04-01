"""
Model loading and management for the AI Text Humanizer.
"""
import logging
from typing import Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

class Models:
    """Class to manage loaded models."""
    
    def __init__(self):
        """Initialize the models container."""
        self.models = {}
        self.loaded = False
    
    def load(self):
        """Load all required models."""
        if self.loaded:
            logger.info("Models already loaded")
            return self.models
        
        try:
            # In a real implementation, this would load actual models
            # For this example, we'll use placeholders
            
            # Load rewriter model
            logger.info("Loading rewriter model...")
            self.models["rewriter"] = self._load_mock_rewriter()
            
            # Load classifier model
            logger.info("Loading classifier model...")
            self.models["classifier"] = self._load_mock_classifier()
            
            self.loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
        
        return self.models
    
    def are_loaded(self) -> bool:
        """Check if models are loaded."""
        return self.loaded
    
    def get(self, model_name: str) -> Any:
        """Get a specific model by name."""
        if not self.loaded:
            self.load()
        
        return self.models.get(model_name)
    
    def _load_mock_rewriter(self):
        """Load a mock rewriter model."""
        # In a real implementation, this would load an actual model
        return {
            "name": "mock-rewriter",
            "type": "transformer",
            "function": lambda text, style: f"Humanized ({style}): {text}"
        }
    
    def _load_mock_classifier(self):
        """Load a mock classifier model."""
        # In a real implementation, this would load an actual model
        return {
            "name": "mock-classifier",
            "type": "classifier",
            "function": lambda text: {"ai_probability": 0.5}
        }

# Singleton instance
_models_instance = None

def load_models() -> Dict[str, Any]:
    """Load and return all models."""
    global _models_instance
    
    if _models_instance is None:
        _models_instance = Models()
    
    return _models_instance.load()

def get_models() -> Dict[str, Any]:
    """Get loaded models or load them if not loaded."""
    global _models_instance
    
    if _models_instance is None or not _models_instance.are_loaded():
        return load_models()
    
    return _models_instance.models
