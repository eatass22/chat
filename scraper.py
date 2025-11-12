import requests
from bs4 import BeautifulSoup
import time
import os
import re
from urllib.parse import urljoin

class IMSDbScraper:
    def __init__(self, base_url="https://imsdb.com", delay=1):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
    
    def get_script_links_from_movie_list(self):
        """Get script links from the movie list pages"""
        try:
            script_links = []
            
            # IMSDb has movie list pages like this
            movie_list_pages = [
                "https://imsdb.com/movie%20scripts.html",
                "https://imsdb.com/all-scripts.html",
                "https://imsdb.com/alphabetical-list.html"
            ]
            
            for page_url in movie_list_pages:
                try:
                    print(f"Scanning {page_url}")
                    response = self.session.get(page_url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find all links that point to script pages
                    all_links = soup.find_all('a', href=True)
                    
                    for link in all_links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        
                        # Script URLs follow pattern: /scripts/Movie-Name.html
                        if href.startswith('/scripts/') and href.endswith('.html'):
                            full_url = urljoin(self.base_url, href)
                            if text and text not in ['', 'Scripts', 'Home']:
                                script_links.append({
                                    'url': full_url,
                                    'title': text,
                                    'filename': href.split('/')[-1].replace('.html', '')
                                })
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    print(f"Error scanning {page_url}: {e}")
                    continue
            
            # Remove duplicates
            unique_links = []
            seen_urls = set()
            for movie in script_links:
                if movie['url'] not in seen_urls:
                    unique_links.append(movie)
                    seen_urls.add(movie['url'])
            
            print(f"Found {len(unique_links)} script links")
            return unique_links
            
        except Exception as e:
            print(f"Error getting script links: {e}")
            return []

    def get_popular_script_links(self):
        """Get scripts from popular/most viewed pages"""
        try:
            script_links = []
            
            # Try to find scripts by common patterns
            popular_pages = [
                "https://imsdb.com/genre/Comedy",
                "https://imsdb.com/genre/Action", 
                "https://imsdb.com/genre/Drama",
                "https://imsdb.com/genre/Horror"
            ]
            
            for page_url in popular_pages:
                try:
                    print(f"Scanning {page_url}")
                    response = self.session.get(page_url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Look for links in tables or lists
                    all_links = soup.find_all('a', href=True)
                    
                    for link in all_links:
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        
                        # Script URLs pattern
                        if href and '/scripts/' in href and href.endswith('.html'):
                            full_url = urljoin(self.base_url, href)
                            if text and len(text) > 2:
                                script_links.append({
                                    'url': full_url,
                                    'title': text,
                                    'genre': page_url.split('/')[-1]
                                })
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    print(f"Error scanning {page_url}: {e}")
                    continue
            
            # Remove duplicates
            unique_links = []
            seen_urls = set()
            for movie in script_links:
                if movie['url'] not in seen_urls:
                    unique_links.append(movie)
                    seen_urls.add(movie['url'])
            
            print(f"Found {len(unique_links)} script links from popular pages")
            return unique_links
            
        except Exception as e:
            print(f"Error getting popular scripts: {e}")
            return []

    def extract_script_text(self, script_url):
        """Extract script text from a script page"""
        try:
            print(f"  Downloading from: {script_url}")
            response = self.session.get(script_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe']):
                element.decompose()
            
            # IMSDb scripts are usually in <pre> tags or specific tables
            script_text = ""
            
            # Method 1: Look for <pre> tags (most common)
            pre_tags = soup.find_all('pre')
            for pre in pre_tags:
                text = pre.get_text(strip=False)
                if len(text) > 1000 and self.looks_like_script(text):
                    script_text = text
                    break
            
            # Method 2: Look for the script in table cells
            if not script_text:
                td_elements = soup.find_all('td')
                for td in td_elements:
                    text = td.get_text(strip=False)
                    if len(text) > 2000 and self.looks_like_script(text):
                        script_text = text
                        break
            
            # Method 3: Some scripts might be in the main content
            if not script_text:
                # Look for the main content area
                main_content = soup.find('td', {'class': re.compile('scrtext', re.I)}) or \
                              soup.find('div', {'class': re.compile('script', re.I)})
                if main_content:
                    script_text = main_content.get_text(strip=False)
            
            # Clean the text
            if script_text:
                script_text = self.clean_script_text(script_text)
                # Check if it's actually a script
                if len(script_text) < 500 or not self.looks_like_script(script_text):
                    return ""
            
            return script_text
            
        except Exception as e:
            print(f"  Error extracting script: {e}")
            return ""

    def looks_like_script(self, text):
        """Check if the text looks like a movie script"""
        script_patterns = [
            r'INT\.|EXT\.',  # Scene headings
            r'^[A-Z][A-Z\s]+$',  # Character names in all caps
            r'FADE IN|FADE OUT',  # Common script terms
            r'CLOSE UP|CUT TO',  # Direction
            r'[A-Z][a-z]+:\s',  # Character: dialogue
        ]
        
        lines = text.split('\n')[:20]  # Check first 20 lines
        sample = '\n'.join(lines)
        
        score = 0
        for pattern in script_patterns:
            if re.search(pattern, sample, re.MULTILINE):
                score += 1
        
        return score >= 2

    def clean_script_text(self, text):
        """Clean up the script text"""
        # Remove excessive whitespace but keep line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove common unwanted text
        unwanted = [
            r'<!--.*?-->',
            r'<script.*?</script>',
            r'<style.*?</style>',
            r'ADVERTISEMENT',
            r'Page \d+',
            r'Back to IMSDb',
            r'All scripts.*?IMSDb',
        ]
        
        for pattern in unwanted:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        return text.strip()

    def scrape_scripts(self, output_dir="movie_scripts", max_scripts=20):
        """Main scraping function"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Finding script links...")
        
        # Try multiple methods to find scripts
        script_links = self.get_popular_script_links()
        
        if not script_links:
            print("Trying alternative method...")
            script_links = self.get_script_links_from_movie_list()
        
        if not script_links:
            # Last resort: try some known script URLs
            script_links = self.get_fallback_scripts()
        
        if not script_links:
            print("No script links found! The site structure may have changed.")
            return []
        
        print(f"Found {len(script_links)} scripts. Starting download...")
        
        successful = []
        
        for i, movie in enumerate(script_links[:max_scripts]):
            print(f"[{i+1}/{min(len(script_links), max_scripts)}] {movie['title']}")
            
            script_text = self.extract_script_text(movie['url'])
            
            if script_text:
                # Create safe filename
                filename = re.sub(r'[^\w\s-]', '', movie['title'])
                filename = re.sub(r'[-\s]+', '_', filename)
                filename = f"{filename}.txt"
                filepath = os.path.join(output_dir, filename)
                
                # Save as .txt
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(script_text)
                
                successful.append(movie['title'])
                print(f"  ✓ Saved: {filename} ({len(script_text)} chars)")
            else:
                print(f"  ✗ Failed to extract script")
            
            time.sleep(self.delay)
        
        print(f"\n=== COMPLETE ===")
        print(f"Successfully downloaded {len(successful)} scripts")
        print(f"Saved to: {output_dir}/")
        
        return successful

    def get_fallback_scripts(self):
        """Fallback: try some known script URLs"""
        print("Using fallback script list...")
        
        # Some example script URLs (you can add more)
        fallback_scripts = [
            {"url": "https://imsdb.com/scripts/American-Psycho.html", "title": "American Psycho"},
            {"url": "https://imsdb.com/scripts/Pulp-Fiction.html", "title": "Pulp Fiction"},
            {"url": "https://imsdb.com/scripts/Good-Will-Hunting.html", "title": "Good Will Hunting"},
            {"url": "https://imsdb.com/scripts/Fight-Club.html", "title": "Fight Club"},
            {"url": "https://imsdb.com/scripts/Forrest-Gump.html", "title": "Forrest Gump"},
            {"url": "https://imsdb.com/scripts/Shawshank-Redemption,-The.html", "title": "The Shawshank Redemption"},
            {"url": "https://imsdb.com/scripts/Godfather,-The.html", "title": "The Godfather"},
            {"url": "https://imsdb.com/scripts/Inception.html", "title": "Inception"},
            {"url": "https://imsdb.com/scripts/Matrix,-The.html", "title": "The Matrix"},
            {"url": "https://imsdb.com/scripts/Social-Network,-The.html", "title": "The Social Network"},
        ]
        
        return fallback_scripts

def main():
    scraper = IMSDbScraper(delay=1)
    scraper.scrape_scripts(
        output_dir="movie_scripts",
        max_scripts=10
    )

if __name__ == "__main__":
    main()