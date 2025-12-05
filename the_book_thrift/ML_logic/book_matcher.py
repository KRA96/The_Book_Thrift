"""
Book Matcher: Match OCR results to book database
Matches books from OCR against local CSV, with Hardcover API fallback
"""

import pandas as pd
import requests
import os
from typing import Dict, List, Optional
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class BookMatcher:
    """
    Match OCR-detected books to your book database
    """
    
    def __init__(self, csv_path: str = "raw_data/book_titles.csv", threshold: int = 85):
        """
        Initialize BookMatcher
        
        Args:
            csv_path: Path to book_titles.csv
            threshold: Fuzzy matching threshold (0-100, default 85)
        """
        self.threshold = threshold
        self.csv_path = csv_path
        
        # Load book titles CSV
        print(f"Loading book titles from {csv_path}...")
        self.books_df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.books_df)} books from local database")
        
        # Get Hardcover API token
        self.hardcover_token = os.getenv("TOKEN")
        if self.hardcover_token:
            print("Hardcover API token found - fallback enabled")
        else:
            print("Warning: No Hardcover API token found - fallback disabled")
    
    def fuzzy_match_title(self, query_title: str) -> Optional[Dict]:
        """
        Fuzzy match a book title against local CSV
        Returns the most popular work (by work_id frequency) if multiple matches
        
        Args:
            query_title: Book title from OCR
            
        Returns:
            Dictionary with match info, or None if no good match
        """
        if not query_title:
            return None
        
        query = query_title.lower()
        
        # Create searchable strings from CSV with indices
        titles_lower = self.books_df['title'].str.lower()
        
        # Find best match using fuzzy matching
        best_match = process.extractOne(
            query,
            titles_lower,
            scorer=fuzz.token_sort_ratio
        )
        
        if not best_match or best_match[1] < self.threshold:
            return None
        
        # best_match is (matched_text, score, index)
        matched_text, score, match_idx = best_match
        
        # Get the original title from the dataframe
        matched_title = self.books_df.iloc[match_idx]['title']
        
        # Find all books with this exact title
        same_title_books = self.books_df[self.books_df['title'] == matched_title]
        
        # If multiple books with same title, pick most common work_id
        if len(same_title_books) > 1:
            # Count work_ids (excluding NaN)
            work_ids = same_title_books['work_id'].dropna()
            if len(work_ids) > 0:
                most_common_work = work_ids.mode()[0]  # Most frequent work_id
                # Get a book with that work_id
                matched_row = same_title_books[same_title_books['work_id'] == most_common_work].iloc[0]
            else:
                matched_row = same_title_books.iloc[0]
        else:
            matched_row = same_title_books.iloc[0]
        
        return {
            'book_id': int(matched_row['book_id']),
            'work_id': int(matched_row['work_id']) if pd.notna(matched_row['work_id']) else None,
            'title': matched_row['title'],
            'isbn': matched_row['isbn'] if pd.notna(matched_row['isbn']) else None,
            'isbn13': matched_row['isbn13'] if pd.notna(matched_row['isbn13']) else None,
            'match_score': score,
            'source': 'local_csv',
            'found_in_local': True,
            'editions_count': len(same_title_books)
        }
    
    def search_hardcover_api(self, title: str, author: Optional[str] = None) -> Optional[Dict]:
        """
        Search Hardcover API for a book using proper search endpoint
        
        Args:
            title: Book title
            author: Author name (optional)
            
        Returns:
            Dictionary with book info including genres, or None if not found
        """
        if not self.hardcover_token:
            print("Hardcover API token not available")
            return None
        
        url = "https://api.hardcover.app/v1/graphql"
        
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {self.hardcover_token}"
        }
        
        # Build search term
        search_term = f"{title} {author}" if author else title
        
        # Use the correct search query
        query = """
        query Search($query: String!, $page: Int, $per_page: Int) {
            search(query: $query, page: $page, per_page: $per_page, query_type: "book") {
                results
                ids
            }
        }
        """
        
        variables = {
            "query": search_term,
            "page": 1,
            "per_page": 5
        }
        
        try:
            response = requests.post(
                url, 
                json={"query": query, "variables": variables}, 
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for errors
                if 'errors' in data:
                    print(f"API Error: {data['errors']}")
                    return None
                
                # Navigate the nested structure: data -> search -> results -> hits -> document
                search_data = data.get('data', {}).get('search', {})
                results = search_data.get('results', {})
                hits = results.get('hits', [])
                
                if hits and len(hits) > 0:
                    # Get the document from first hit
                    book = hits[0].get('document', {})
                    
                    # Extract author names
                    author_names = book.get('author_names', [])
                    
                    # Extract useful metadata
                    genres = book.get('genres', [])
                    tags = book.get('tags', [])
                    moods = book.get('moods', [])
                    rating = book.get('rating')
                    ratings_count = book.get('ratings_count', 0)
                    pages = book.get('pages')
                    release_year = book.get('release_year')
                    
                    # Combine genres, tags, and moods for category info
                    categories = []
                    if genres:
                        categories.extend(genres)
                    if tags:
                        categories.extend(tags[:3])  # Top 3 tags
                    if moods:
                        categories.extend(moods[:2])  # Top 2 moods
                    
                    # Build helpful message
                    info_parts = []
                    if rating:
                        info_parts.append(f"Rating: {rating:.2f}/5")
                    if ratings_count:
                        info_parts.append(f"({ratings_count} ratings)")
                    if categories:
                        info_parts.append(f"Categories: {', '.join(categories[:5])}")
                    if pages:
                        info_parts.append(f"{pages} pages")
                    
                    message = f"Book found in Hardcover API. {' | '.join(info_parts)}. Not in local database - recommendations unavailable."
                    
                    return {
                        'book_id': None,
                        'hardcover_id': book.get('id'),
                        'title': book.get('title'),
                        'author': ', '.join(author_names) if author_names else None,
                        'isbn': book.get('isbns', [None])[0] if book.get('isbns') else None,
                        'isbn13': None,
                        'genres': genres,
                        'categories': categories,
                        'rating': rating,
                        'ratings_count': ratings_count,
                        'pages': pages,
                        'release_year': release_year,
                        'match_score': 90,
                        'source': 'hardcover_api',
                        'found_in_local': False,
                        'message': message
                    }
            else:
                print(f"API returned status {response.status_code}")
                
        except Exception as e:
            print(f"Error searching Hardcover API: {e}")
        
        return None
    
    def match_book(self, title: str, author: Optional[str] = None) -> Dict:
        """
        Complete matching pipeline: try local CSV first, then Hardcover API
        
        Args:
            title: Book title from OCR
            author: Author name from OCR (optional, used for Hardcover API)
            
        Returns:
            Dictionary with match info and book_id (if found locally)
        """
        print(f"\nMatching: '{title}'" + (f" by {author}" if author else ""))
        
        # Step 1: Try local CSV match (title only, since CSV doesn't have authors)
        local_match = self.fuzzy_match_title(title)
        
        if local_match:
            print(f"✓ Found in local database: '{local_match['title']}' (score: {local_match['match_score']})")
            if local_match.get('editions_count', 0) > 1:
                print(f"  Note: {local_match['editions_count']} editions found, selected most common")
            return local_match
        
        # Step 2: Fallback to Hardcover API (uses author if provided)
        print("✗ Not found in local database, trying Hardcover API...")
        api_match = self.search_hardcover_api(title, author)
        
        if api_match:
            print(f"✓ Found via Hardcover API: '{api_match['title']}'")
            if api_match.get('genres'):
                print(f"  Genres: {', '.join(api_match.get('genres', [])[:3])}")
            return api_match
        
        # Step 3: No match found anywhere
        print("✗ Book not found in either source")
        return {
            'book_id': None,
            'title': title,
            'author': author,
            'match_score': 0,
            'source': 'not_found',
            'found_in_local': False,
            'message': 'Book not found in our database or Hardcover API'
        }
    
    def match_multiple_books(self, books: List[Dict]) -> List[Dict]:
        """
        Match multiple books from OCR results
        
        Args:
            books: List of book dictionaries from OCR (with 'title' and 'author' keys)
            
        Returns:
            List of matched book dictionaries with book_ids
        """
        results = []
        
        for book in books:
            title = book.get('title')
            author = book.get('author')
            
            if title:
                match = self.match_book(title, author)
                # Add original OCR confidence
                match['ocr_confidence'] = book.get('confidence', 0)
                results.append(match)
        
        return results
