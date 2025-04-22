"""
Google Translate provider implementation.

This module provides the GoogleTranslateProvider class for interacting with Google's
Translation API v2.
"""

import os
import time
import logging
import requests
from typing import List, Dict, Optional

from .base import TranslationProvider, TranslationRequest, TranslationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleTranslateProvider(TranslationProvider):
    """
    Translation provider using Google Translate V2 REST API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Google Translate provider.
        
        Args:
            api_key: Optional API key (if not provided, will use environment variables)
        """
        super().__init__()
        self.api_key = api_key or os.getenv("GOOGLE_TRANSLATE_API_KEY")
        if not self.api_key:
            raise ValueError("Google Translate API key not provided and not found in environment variables")
        
        self.base_url = "https://translation.googleapis.com/language/translate/v2"
        self.session = requests.Session()
    
    def initialize(self):
        """Initialize the provider. No special initialization needed for REST API."""
        pass
    
    def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate a single text using Google Translate V2 REST API.
        
        Args:
            request: Translation request containing text and language info
            
        Returns:
            TranslationResponse with translated text
        """
        # Prepare API request parameters
        params = {
            "key": self.api_key,
            "q": request.text,
            "source": request.source_lang,
            "target": request.target_lang,
            "format": "text"
        }
        
        try:
            # Make API request
            response = self.session.post(self.base_url, params=params)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            if "data" not in data or "translations" not in data["data"]:
                raise ValueError(f"Unexpected API response format: {data}")
            
            translated_text = data["data"]["translations"][0]["translatedText"]
            
            return TranslationResponse(
                translated_text=translated_text,
                metadata=request.metadata
            )
            
        except requests.exceptions.RequestException as e:
            raise TranslationError(f"Google Translate API request failed: {str(e)}")
        except (KeyError, IndexError, ValueError) as e:
            raise TranslationError(f"Failed to parse Google Translate API response: {str(e)}")
    
    async def atranslate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Asynchronously translate a single text.
        Note: Currently just calls the synchronous version.
        """
        return self.translate(request)
    
    def batch_translate(self, requests: List[TranslationRequest]) -> List[TranslationResponse]:
        """
        Translate multiple texts in batch using Google Translate V2 REST API.
        
        Args:
            requests: List of translation requests
            
        Returns:
            List of translation responses
        """
        # Group requests by language pair for efficiency
        grouped_requests = {}
        for req in requests:
            lang_pair = (req.source_lang, req.target_lang)
            if lang_pair not in grouped_requests:
                grouped_requests[lang_pair] = []
            grouped_requests[lang_pair].append(req)
        
        responses = []
        
        # Process each language pair group
        for (source_lang, target_lang), group_requests in grouped_requests.items():
            # Prepare API request parameters
            params = {
                "key": self.api_key,
                "q": [req.text for req in group_requests],  # Send all texts for this language pair
                "source": source_lang,
                "target": target_lang,
                "format": "text"
            }
            
            try:
                # Make API request
                response = self.session.post(self.base_url, params=params)
                response.raise_for_status()
                
                # Parse response
                data = response.json()
                if "data" not in data or "translations" not in data["data"]:
                    raise ValueError(f"Unexpected API response format: {data}")
                
                translations = data["data"]["translations"]
                
                # Create response objects maintaining the original order
                for req, translation in zip(group_requests, translations):
                    responses.append(TranslationResponse(
                        translated_text=translation["translatedText"],
                        source_lang=source_lang,
                        target_lang=target_lang,
                        metadata=req.metadata
                    ))
                
            except requests.exceptions.RequestException as e:
                raise TranslationError(f"Google Translate API batch request failed: {str(e)}")
            except (KeyError, IndexError, ValueError) as e:
                raise TranslationError(f"Failed to parse Google Translate API batch response: {str(e)}")
        
        return responses
    
    async def abatch_translate(self, requests: List[TranslationRequest]) -> List[TranslationResponse]:
        """
        Asynchronously translate multiple texts in batch.
        Note: Currently just calls the synchronous version.
        """
        return self.batch_translate(requests)

    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes from Google Translate API.
        
        Returns:
            List of supported language codes (e.g., ['en', 'es', 'fr', ...])
        """
        # Google Translate supports a large number of languages
        # This is a subset of the most commonly used ones
        return [
            'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo',
            'es', 'et', 'eu', 'fa', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw', 'he', 'hi', 'hmn', 'hr', 'ht',
            'hu', 'hy', 'id', 'ig', 'is', 'it', 'iw', 'ja', 'jw', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb',
            'lo', 'lt', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'or', 'pa',
            'pl', 'ps', 'pt', 'ro', 'ru', 'rw', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw',
            'ta', 'te', 'tg', 'th', 'tk', 'tl', 'tr', 'tt', 'ug', 'uk', 'ur', 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zu'
        ]
