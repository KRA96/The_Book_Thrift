"""
OCR Processor for Book Cover Text Extraction
Extracts book information (title, author, ISBN) from book cover images
"""

import easyocr
import cv2
import numpy as np
from PIL import Image
import re
from typing import Dict, List, Optional, Tuple


class BookOCRProcessor:
    """
    Process book cover images to extract text information
    """

    def __init__(self, languages: List[str] = ['en']):
        """
        Initialize OCR reader

        Args:
            languages: List of language codes for OCR (default: English)
        """
        print("Initializing EasyOCR reader... ")
        self.reader = easyocr.Reader(languages, gpu=False)
        print("OCR reader initialized successfully!")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy

        Args:
            image_path: Path to the book cover image

        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Could not read image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding for better text contrast
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Denoise
        denoised = cv2.fastNlMeansDenoising(processed)

        return denoised

    def extract_text(self, image_path: str, preprocess: bool = False) -> List[tuple]:
        """
        Extract all text from book cover image with position information

        Args:
            image_path: Path to the book cover image
            preprocess: Whether to preprocess the image (default: False for better results)

        Returns:
            List of tuples containing (bounding_box, text, confidence)
        """
        if preprocess:
            # Use preprocessed image
            img = self.preprocess_image(image_path)
        else:
            # Use original image - often works better with colored book covers
            img = image_path

        # Perform OCR
        results = self.reader.readtext(img)

        return results

    def clean_text(self, text: str) -> str:
        """
        Clean up OCR text by removing extra spaces and fixing common errors

        Args:
            text: Raw OCR text

        Returns:
            Cleaned text
        """
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_likely_author_name(self, text: str) -> bool:
        """
        Check if text looks like an author name

        Args:
            text: Text to check

        Returns:
            True if text looks like an author name
        """
        # Clean the text first
        text = self.clean_text(text)
        text_lower = text.lower()

        # Common exclusion keywords
        exclude_keywords = [
            'bestseller', 'national', 'autobiography', 'memoir', 'novel',
            'edition', 'time', 'york', 'press', 'publishing', 'book',
            'story', 'tale', 'chronicles', 'series'
        ]

        # Exclude if contains any exclusion keyword
        if any(keyword in text_lower for keyword in exclude_keywords):
            return False

        # Exclude rankings
        if re.search(r'#\d+', text):
            return False

        # Check if it looks like a name
        # Names are typically 2-4 words, mostly alphabetic
        words = text.split()

        # Single word names are possible but less common
        if len(words) == 1:
            # Accept if it's a long capitalized word (could be concatenated name)
            if len(text) >= 8 and text[0].isupper() and sum(c.isalpha() for c in text) > len(text) * 0.8:
                return True
            return False

        if 2 <= len(words) <= 4:
            # Check if words are mostly alphabetic and properly capitalized
            valid_words = 0
            for word in words:
                if word and len(word) >= 2:
                    # Word should be capitalized and mostly letters
                    if word[0].isupper() and sum(c.isalpha() for c in word) > len(word) * 0.7:
                        valid_words += 1

            # If most words look like name parts, it's likely a name
            if valid_words >= len(words) * 0.5:
                return True

        return False

    def parse_book_info_simple(self, ocr_results: List[tuple]) -> Dict[str, Optional[str]]:
        """
        Parse OCR results for a single book cover using improved heuristics

        Args:
            ocr_results: List of tuples from extract_text()

        Returns:
            Dictionary with parsed book information
        """
        # Extract text with position and size information
        text_data = []
        for result in ocr_results:
            bbox, text, confidence = result
            # Calculate y-position (vertical position) and height
            y_top = min(point[1] for point in bbox)
            y_bottom = max(point[1] for point in bbox)
            height = y_bottom - y_top

            text_data.append({
                'text': text,
                'confidence': confidence,
                'y_position': y_top,
                'height': height,
                'bbox': bbox
            })

        # Sort by vertical position (top to bottom)
        text_data.sort(key=lambda x: x['y_position'])

        all_text = " ".join([item['text'] for item in text_data])

        book_info = {
            'title': None,
            'author': None,
            'isbn': None,
            'raw_text': all_text,
            'all_detected_text': [item['text'] for item in text_data]
        }

        # Extract ISBN (10 or 13 digits, may have hyphens)
        isbn_pattern = r'(?:ISBN[-:]?\s*)?(\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dX])'
        isbn_match = re.search(isbn_pattern, all_text, re.IGNORECASE)
        if isbn_match:
            book_info['isbn'] = isbn_match.group(1).replace('-', '').replace(' ', '')

        # Filter out small/low-confidence text
        main_text = [item for item in text_data if item['confidence'] > 0.3]

        # Find author candidates using improved name detection
        author_candidates = []
        for item in main_text:
            if self.is_likely_author_name(item['text']):
                author_candidates.append(item)

        # Title is typically the largest text at the top (excluding author-like text)
        if len(main_text) > 0:
            # Get non-author text for title detection
            non_author_text = [item for item in main_text
                             if not self.is_likely_author_name(item['text'])]

            if non_author_text:
                title_candidate = max(non_author_text, key=lambda x: x['height'])
                book_info['title'] = self.clean_text(title_candidate['text'])
            else:
                # Fallback: use largest text
                title_candidate = max(main_text, key=lambda x: x['height'])
                book_info['title'] = self.clean_text(title_candidate['text'])

        # Author is the largest author candidate (by font size)
        if author_candidates:
            author = max(author_candidates, key=lambda x: x['height'])
            author_text = self.clean_text(author['text'])

            # Try to add spaces if it's a concatenated name (like ANDREAGASSI)
            if len(author_text.split()) == 1 and len(author_text) > 8:
                # Simple heuristic: split at capital letters for names like "ANDREAGASSI"
                spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', author_text)
                book_info['author'] = spaced
            else:
                book_info['author'] = author_text

        return book_info

    def process_book_cover(self, image_path: str) -> Dict[str, Optional[str]]:
        """
        Complete pipeline: extract and parse book information from cover image

        Args:
            image_path: Path to the book cover image

        Returns:
            Dictionary with parsed book information
        """
        print(f"Processing image: {image_path}")

        # Extract text (without preprocessing for better color detection)
        ocr_results = self.extract_text(image_path, preprocess=False)

        print(f"Detected {len(ocr_results)} text regions")

        # Parse information
        book_info = self.parse_book_info_simple(ocr_results)

        return book_info


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = BookOCRProcessor()

    # Process a book cover
    # result = processor.process_book_cover("path/to/book_cover.jpg")
    # print(result)
