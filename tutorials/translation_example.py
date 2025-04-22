"""
Example usage of the APITranslationStep with Google Translate REST API.
"""

import os
from dotenv import load_dotenv
from marovipipelines.translation.steps import APITranslationStep
from marovipipelines.core.context import PipelineContext

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Create a pipeline context
    context = PipelineContext()
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY")
    if not api_key:
        print("Warning: GOOGLE_TRANSLATE_API_KEY not found in .env file")
        print("Please add it to your .env file: GOOGLE_TRANSLATE_API_KEY=your-api-key")
        return
    
    # Create a translation step
    step = APITranslationStep(
        source_lang="en",
        target_lang="es",
        provider="google",
        api_key=api_key,  # Pass the API key directly
        batch_size=2,  # Process 2 items at a time
        max_concurrency=5  # Maximum concurrent API requests
    )
    
    # Test texts to translate
    texts = [
        "Hello, how are you?",
        "The weather is nice today.",
        "I love programming in Python.",
        "This is a test of the translation pipeline."
    ]
    
    # Process the texts
    print("Translating texts...")
    translated_texts = step.process(texts, context)
    
    # Print results
    print("\nTranslation Results:")
    print("-" * 50)
    for original, translated in zip(texts, translated_texts):
        print(f"Original: {original}")
        print(f"Translated: {translated}")
        print("-" * 50)
    
    # Print some metrics from the context
    print("\nTranslation Metrics:")
    print(f"Total latency: {context.get_metric('translate_en_to_es_batch_translation_total_time'):.2f}s")
    print(f"Average time per text: {context.get_metric('translate_en_to_es_batch_translation_avg_time'):.2f}s")
    print(f"Total texts translated: {context.get_metric('translate_en_to_es_batch_translation_size')}")

if __name__ == "__main__":
    main() 