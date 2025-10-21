import time
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


class ScrapeWeb:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.default_headers = {
            "User-Agent": (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
        self.http_timeout = 15

    def _search_web(self, query: str) -> List[dict]:
        """
        Returns a list of dictionaries with 'title', 'href', and 'body' keys
        """
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                return results
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def _fetch_page_content(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetch and parse the content from a single URL
        Returns a dictionary with 'url', 'title', 'text', and 'status'
        """
        try:
            print(f"Fetching: {url}")
            response = requests.get(url, headers=self.default_headers, timeout=self.http_timeout)

            if response.status_code != 200:
                print(f"  Failed with status code: {response.status_code}")
                return {
                    'url': url,
                    'title': None,
                    'text': None,
                    'status': f'Error: {response.status_code}'
                }

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get title
            title = soup.title.string if soup.title else "No title"

            # Get main text content
            # Try to find main content areas first
            main_content = soup.find('main') or soup.find('article') or soup.find('body')

            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)

            # Clean up whitespace
            text = ' '.join(text.split())

            print(f"  Success! Extracted {len(text)} characters")

            return {
                'url': url,
                'title': title,
                'text': text,
                'status': 'Success'
            }

        except requests.Timeout:
            print(f"  Timeout error")
            return {'url': url, 'title': None, 'text': None, 'status': 'Timeout'}
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            return {'url': url, 'title': None, 'text': None, 'status': f'Error: {str(e)}'}
        except Exception as e:
            print(f"  Parsing error: {e}")
            return {'url': url, 'title': None, 'text': None, 'status': f'Parsing error: {str(e)}'}

    def _search_and_read_all(self, query: str, delay: float = 1.0) -> List[Dict[str, str]]:
        """
        Search and fetch content from all result URLs
        delay: seconds to wait between requests (be respectful to servers)
        """
        search_results = self._search_web(query=query)
        all_content: List[Dict[str, str]] = []

        for i, result in enumerate(search_results, 1):
            url = result.get('href')
            if not url:
                continue

            print(f"\n[{i}/{len(search_results)}] Processing: {result.get('title', 'No title')}")

            content = self._fetch_page_content(url)
            if content:
                # Add search result metadata
                content['search_title'] = result.get('title')
                content['search_description'] = result.get('body')
                all_content.append(content)

            # Be respectful - don't hammer servers
            if i < len(search_results):
                time.sleep(delay)

        return all_content

    def search(self, query: str) -> List[Dict[str, str]]:
        return self._search_and_read_all(query=query)