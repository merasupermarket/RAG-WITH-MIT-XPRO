# Install the required libraries if you haven't already
# pip install trafilatura
# it will create the file under WikiFiles folder in the current directory
# have a WikiFiles folder in the current directory before running the code

import re
import trafilatura

url = "https://en.wikipedia.org/wiki/Apollo_11"

# Method 1: Using Trafilatura (Recommended for clean text extraction)
downloaded = trafilatura.fetch_url(url)
text_only = trafilatura.extract(downloaded, include_links=False, include_images=False)
text_only = re.sub(r"\[\d+\]", "", text_only or "")


# get the file name form the url
file_name = url.split("/")[-1] + ".txt"

with open(f".\\WikiFiles\\{file_name}", "w", encoding="utf-8") as f:
    f.write(text_only)