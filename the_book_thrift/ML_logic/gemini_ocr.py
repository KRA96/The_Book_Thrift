"""
Gemini Vision-based OCR Processor for Book Cover Text Extraction
Uses Google's Gemini AI to extract book information from images
"""

import google.generativeai as genai
import json
import os
from typing import Dict, List, Optional
from PIL import Image


class GeminiBookOCR:
    """
    Process book cover images using Gemini Vision API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini OCR
        
        Args:
            api_key: Gemini API key (if None, will try to get from environment)
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Please provide it as parameter "
                "or set GEMINI_API_KEY environment variable."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Use Gemini 2.5 Flash - fast and accurate
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        print("Gemini OCR initialized successfully!")
    
    def create_prompt(self, multiple_books: bool = False) -> str:
        """
        Create the prompt for Gemini based on use case
        
        Args:
            multiple_books: Whether to expect multiple books (bookshelf) or single book
            
        Returns:
            Prompt string
        """
        if multiple_books:
            return """
You are a book-cover OCR system.
Look at the image and detect every visible book.
Return ONLY valid JSON (no markdown, no comments, no code blocks) with this schema:
{
  "books": [
    {
      "title": "<best guess of the book title>",
      "author": "<best guess of the main author, can be null>",
      "confidence": <number between 0 and 1>
    }
  ]
}
- Use null for unknown fields.
- Include only books where you can see at least part of the title.
- Try to separate individual books even if titles are close together.
- For confidence: 1.0 = very confident, 0.5 = moderate, 0.0 = very unsure.
"""
        else:
            return """
You are a book-cover OCR system.
Look at this single book cover image and extract the book information.
Return ONLY valid JSON (no markdown, no comments, no code blocks) with this schema:
{
  "title": "<the book title>",
  "author": "<the author name, can be null>",
  "subtitle": "<subtitle if present, can be null>",
  "isbn": "<ISBN if visible, can be null>",
  "confidence": <number between 0 and 1>
}
- Use null for unknown fields.
- For confidence: 1.0 = very confident, 0.5 = moderate, 0.0 = very unsure.
"""
    
    def process_single_book(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Process a single book cover image
        
        Args:
            image_path: Path to the book cover image
            
        Returns:
            Dictionary with book information
        """
        print(f"Processing single book with Gemini: {image_path}")
        
        # Load image
        img = Image.open(image_path)
        
        # Create prompt
        prompt = self.create_prompt(multiple_books=False)
        
        # Generate response
        response = self.model.generate_content([prompt, img])
        
        # Parse JSON response
        try:
            # Clean response text (remove markdown if present)
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            result = json.loads(response_text)
            
            # Add detected text for compatibility
            result['all_detected_text'] = [result.get('title', ''), result.get('author', '')]
            result['raw_text'] = f"{result.get('title', '')} {result.get('author', '')}"
            
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Raw response: {response.text}")
            return {
                'title': None,
                'author': None,
                'isbn': None,
                'confidence': 0.0,
                'error': str(e),
                'raw_response': response.text
            }
    
    def process_bookshelf(self, image_path: str) -> Dict[str, List[Dict]]:
        """
        Process a bookshelf image with multiple books
        
        Args:
            image_path: Path to the bookshelf image
            
        Returns:
            Dictionary with list of detected books
        """
        print(f"Processing bookshelf with Gemini: {image_path}")
        
        # Load image
        img = Image.open(image_path)
        
        # Create prompt for multiple books
        prompt = self.create_prompt(multiple_books=True)
        
        # Generate response
        response = self.model.generate_content([prompt, img])
        
        # Parse JSON response
        try:
            # Clean response text
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing Gemini response: {e}")
            print(f"Raw response: {response.text}")
            return {
                'books': [],
                'error': str(e),
                'raw_response': response.text
            }


# Example usage
if __name__ == "__main__":
    # Initialize processor (make sure GEMINI_API_KEY is set in environment)
    processor = GeminiBookOCR()
    
    # Process a single book
    # result = processor.process_single_book("path/to/book_cover.jpg")
    # print(result)
    
    # Process a bookshelf
    # result = processor.process_bookshelf("path/to/bookshelf.jpg")
    # print(result)
