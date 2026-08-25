import re
import trafilatura

url = "https://en.wikipedia.org/wiki/2022_United_States_elections"

# Method 1: Using Trafilatura (Recommended for clean text extraction)
downloaded = trafilatura.fetch_url(url)
text_only = trafilatura.extract(downloaded, include_links=False, include_images=False)
text_only = re.sub(r"\[\d+\]", "", text_only or "")

with open("2022_US_elections.txt", "w", encoding="utf-8") as f:
    f.write(text_only)