"""
Base model interface

Abstract base class for receipt processing models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseModel(ABC):
    """Abstract base class for receipt processing models."""
    
    @abstractmethod
    def load(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def predict(
        self,
        token_ids: List[int],
        token_boxes: List[List[int]],
        image: Any
    ) -> Dict[str, Any]:
        """
        Run model prediction.
        
        Args:
            token_ids: List of token IDs
            token_boxes: List of normalized bounding boxes
            image: Image tensor or array
            
        Returns:
            Dictionary with predictions
        """
        pass
    
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        pass
