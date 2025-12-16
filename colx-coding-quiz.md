# Python Quiz Study Guide - Text Data Science
## COLX 521 Comprehensive Quiz Preparation

---

## 1. Python Fundamentals

### 1.1 Extract Information from Nested Data Structures

**Question:** Given a dictionary where keys are categories and values are lists of items, extract all items from a specific category.

**What's Required:** Access dictionary values and iterate through nested lists.

**Solution:**
```python
# Example nested structure
data = {
    'fruits': ['apple', 'banana', 'orange'],
    'vegetables': ['carrot', 'broccoli', 'spinach'],
    'grains': ['rice', 'wheat', 'oats']
}

# Extract items from 'fruits' category
fruits = data['fruits']
print(fruits)  # ['apple', 'banana', 'orange']

# Extract all items from all categories
all_items = []
for category in data:
    all_items.extend(data[category])
print(all_items)

# Or using list comprehension
all_items = [item for category in data.values() for item in category]
```

### 1.2 Open a Text File with Appropriate Encoding

**Question:** Open a text file for reading with UTF-8 encoding.

**What's Required:** Use `open()` with correct mode and encoding parameter.

**Solution:**
```python
# Reading a file with UTF-8 encoding
with open('myfile.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Writing to a file with UTF-8 encoding
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write("Hello, world!")
```

### 1.3 Read or Write to Text File (Complete vs Line by Line)

**Question:** Read a file's entire content at once, then read it line by line.

**What's Required:** Use `.read()` for complete reading and iteration for line-by-line.

**Solution:**
```python
# Read entire file at once
with open('file.txt', 'r', encoding='utf-8') as f:
    all_content = f.read()
    print(all_content)

# Read file line by line
with open('file.txt', 'r', encoding='utf-8') as f:
    for line in f:
        print(line.strip())  # strip() removes trailing newline

# Read all lines into a list
with open('file.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# Write multiple lines
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write("Line 1\n")
    f.write("Line 2\n")
    
# Write all at once
text = "Line 1\nLine 2\nLine 3"
with open('output.txt', 'w', encoding='utf-8') as f:
    f.write(text)
```

---

## 2. Statistics

### 2.1 Create a Dictionary of Counts (Word Counts)

**Question:** Count the frequency of each word in a text.

**What's Required:** Use `Counter` from collections or build manually with a dictionary.

**Solution:**
```python
from collections import Counter
from nltk.corpus import brown

# Method 1: Using Counter (recommended)
text = "the cat sat on the mat the cat"
words = text.split()
counts = Counter(words)
print(counts)  # Counter({'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1})

# Method 2: Manual dictionary
counts_manual = {}
for word in words:
    if word in counts_manual:
        counts_manual[word] += 1
    else:
        counts_manual[word] = 1
        
# Method 3: With .get()
counts_manual2 = {}
for word in words:
    counts_manual2[word] = counts_manual2.get(word, 0) + 1

# With Brown corpus
brown_counts = Counter(word.lower() for word in brown.words())
print(brown_counts.most_common(10))  # Top 10 most common words
```

### 2.2 Convert Counts to Probability Distribution

**Question:** Convert a dictionary of word counts into probabilities.

**What's Required:** Divide each count by the total number of words.

**Solution:**
```python
from collections import Counter
from nltk.corpus import brown
import numpy as np

# Create counts
counts = Counter(brown.words())

# Convert to probabilities
probs = {}
total = sum(counts.values())

for word in counts:
    probs[word] = counts[word] / total

# Verify probabilities sum to 1
print(sum(probs.values()))  # Should be 1.0 (or very close)
print(np.sum(list(probs.values())))  # Using numpy

# Check specific probability
print(f"Probability of 'the': {probs['the']}")
```

### 2.3 Sort Words by Their Counts

**Question:** Sort words from most frequent to least frequent based on their counts.

**What's Required:** Use `sorted()` with appropriate key function.

**Solution:**
```python
from collections import Counter

# Create counts
counts = Counter(['the', 'cat', 'sat', 'on', 'the', 'mat', 'the'])

# Method 1: Sort dictionary items by count (descending)
sorted_words = sorted(counts.items(), key=lambda x: x[1], reverse=True)
print(sorted_words)  # [('the', 3), ('cat', 1), ('sat', 1), ...]

# Method 2: Just get sorted words (not counts)
sorted_word_list = sorted(counts, key=counts.get, reverse=True)
print(sorted_word_list)  # ['the', 'cat', 'sat', ...]

# Method 3: Using Counter's most_common()
top_10 = counts.most_common(10)
print(top_10)
```

### 2.4 NLTK Sentence Segmentation and Word Tokenization

**Question:** Break a text into sentences and then tokenize each sentence into words.

**What's Required:** Use `sent_tokenize()` and `word_tokenize()` from NLTK.

**Solution:**
```python
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required resources
nltk.download('punkt')

# Example text
text = "Hello world! This is a sentence. Is this another sentence? Yes, it is."

# Sentence segmentation
sentences = sent_tokenize(text)
print(sentences)
# ['Hello world!', 'This is a sentence.', 'Is this another sentence?', 'Yes, it is.']

# Word tokenization
words = word_tokenize(text)
print(words)
# ['Hello', 'world', '!', 'This', 'is', 'a', 'sentence', '.', ...]

# Tokenize each sentence separately
for sentence in sentences:
    tokens = word_tokenize(sentence)
    print(tokens)
```

### 2.5 POS Tagging with NLTK

**Question:** Perform Part-of-Speech tagging on tokenized sentences.

**What's Required:** Use `pos_tag()` from NLTK on tokenized text.

**Solution:**
```python
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Example sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Tokenize first
tokens = word_tokenize(sentence)

# POS tagging
tagged = pos_tag(tokens)
print(tagged)
# [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ('fox', 'NN'), ...]

# POS tag multiple sentences
sentences = ["I love Python.", "She writes code daily."]
for sent in sentences:
    tokens = word_tokenize(sent)
    tagged = pos_tag(tokens)
    print(tagged)
```

---

## 3. String and Regex Operations

### 3.1 Extract a Substring

**Question:** Extract a portion of a string using various methods.

**What's Required:** Use slicing, string methods like `.split()`, `.find()`, etc.

**Solution:**
```python
# String slicing
text = "Hello, World!"
substring = text[0:5]  # "Hello"
substring = text[7:]   # "World!"
substring = text[-6:]  # "World!"

# Using .split()
email = "user@example.com"
username = email.split('@')[0]  # "user"
domain = email.split('@')[1]    # "example.com"

# Using .find() and slicing
text = "The cat sat on the mat"
start = text.find('cat')
end = start + 3
cat = text[start:end]  # "cat"

# Using string methods
text = "   Hello World   "
cleaned = text.strip()  # "Hello World"
first_word = text.strip().split()[0]  # "Hello"
```

### 3.2 Identify/Filter Particular Strings

**Question:** Filter strings based on certain criteria (e.g., contains substring, starts with, etc.).

**What's Required:** Use string methods like `.startswith()`, `.endswith()`, `in` operator.

**Solution:**
```python
# Check if string contains substring
text = "Hello, World!"
if "World" in text:
    print("Found!")

# Filter words starting with a letter
words = ["apple", "banana", "apricot", "cherry", "avocado"]
a_words = [w for w in words if w.startswith('a')]
print(a_words)  # ['apple', 'apricot', 'avocado']

# Filter words ending with a suffix
words = ["running", "jumping", "walk", "swimming"]
ing_words = [w for w in words if w.endswith('ing')]
print(ing_words)  # ['running', 'jumping', 'swimming']

# Check if string is alphabetic
text = "Hello123"
if text.isalpha():
    print("All letters")
else:
    print("Contains non-letters")

# Filter alphabetic words
words = ["hello", "world123", "python", "3.14"]
alpha_words = [w for w in words if w.isalpha()]
print(alpha_words)  # ['hello', 'python']
```

### 3.3 Generate a Larger String

**Question:** Build a larger string from smaller components.

**What's Required:** Use string concatenation, `.join()`, or f-strings.

**Solution:**
```python
# Method 1: Concatenation
word1 = "Hello"
word2 = "World"
sentence = word1 + " " + word2 + "!"
print(sentence)  # "Hello World!"

# Method 2: .join() (more efficient)
words = ["Hello", "World", "from", "Python"]
sentence = " ".join(words)
print(sentence)  # "Hello World from Python"

# Method 3: f-strings (Python 3.6+)
name = "Alice"
age = 30
message = f"My name is {name} and I am {age} years old."
print(message)

# Build from list
lines = ["Line 1", "Line 2", "Line 3"]
text = "\n".join(lines)
print(text)
```

### 3.4 Convert Numerical Strings to Numbers and Vice Versa

**Question:** Convert between strings and numbers (int/float).

**What's Required:** Use `int()`, `float()`, and `str()` functions.

**Solution:**
```python
# String to int
num_str = "42"
num_int = int(num_str)
print(num_int, type(num_int))  # 42 <class 'int'>

# String to float
price_str = "19.99"
price_float = float(price_str)
print(price_float, type(price_float))  # 19.99 <class 'float'>

# Int to string
age = 25
age_str = str(age)
print(age_str, type(age_str))  # "25" <class 'str'>

# Float to string
pi = 3.14159
pi_str = str(pi)
print(pi_str)  # "3.14159"

# Formatting numbers as strings
value = 123.456
formatted = f"{value:.2f}"  # "123.46" (2 decimal places)
print(formatted)

# Handle errors
try:
    num = int("not a number")
except ValueError:
    print("Cannot convert to int")
```

### 3.5 String Literals and Regex with Escaping

**Question:** Write string literals with proper escaping for special characters.

**What's Required:** Use backslash escaping or raw strings (r"").

**Solution:**
```python
# Escaping quotes
S1 = '"I don\'t think I need to ever use escaping", he said.'
S2 = "\"I don't think I need to ever use escaping\", he said."
S3 = "\"I don't think I need to ever use escaping\", he said."

print(S1)
print(S2)

# Escaping backslashes
path1 = "c:\\files\\videos"
print(path1)  # c:\files\videos

# Raw strings (no escaping needed)
path2 = r"c:\files\videos"
print(path2)  # c:\files\videos

# Newlines and tabs
text = "Line 1\nLine 2\tTabbed"
print(text)

# Regex patterns (use raw strings)
import re
pattern = r"\d+"  # Match one or more digits
pattern2 = r"\w+@\w+\.\w+"  # Email pattern
```

### 3.6 Code Simple Regex Patterns

**Question:** Write a regex pattern to match specific text patterns.

**What's Required:** Understand basic regex syntax and special characters.

**Solution:**
```python
import re

# Match digits
pattern = r"\d+"  # One or more digits
text = "I have 3 cats and 2 dogs"
matches = re.findall(pattern, text)
print(matches)  # ['3', '2']

# Match words
pattern = r"\w+"  # One or more word characters
text = "Hello, World!"
matches = re.findall(pattern, text)
print(matches)  # ['Hello', 'World']

# Match email addresses
pattern = r"\w+@\w+\.\w+"
text = "Contact us at info@example.com or support@test.org"
matches = re.findall(pattern, text)
print(matches)  # ['info@example.com', 'support@test.org']

# Match phone numbers
pattern = r"\d{3}-\d{3}-\d{4}"
text = "Call 555-123-4567 or 555-987-6543"
matches = re.findall(pattern, text)
print(matches)  # ['555-123-4567', '555-987-6543']

# Character classes
pattern = r"[aeiou]"  # Match any vowel
text = "hello"
matches = re.findall(pattern, text)
print(matches)  # ['e', 'o']

# Negation
pattern = r"[^aeiou]"  # Match any non-vowel
text = "hello"
matches = re.findall(pattern, text)
print(matches)  # ['h', 'l', 'l']
```

### 3.7 Find Matches in Strings Using Regex

**Question:** Search for pattern matches in text using regex functions.

**What's Required:** Use `re.search()`, `re.findall()`, `re.finditer()`, `re.match()`.

**Solution:**
```python
import re

text = "The cat sat on the mat. The cat was fat."

# Find first match
pattern = r"cat"
match = re.search(pattern, text)
if match:
    print(f"Found '{match.group()}' at position {match.start()}")
    # Found 'cat' at position 4

# Find all matches
matches = re.findall(pattern, text)
print(matches)  # ['cat', 'cat']

# Find all matches with positions
for match in re.finditer(pattern, text):
    print(f"Found '{match.group()}' at {match.start()}-{match.end()}")

# Match at beginning of string
pattern = r"The"
if re.match(pattern, text):
    print("String starts with 'The'")

# Extract captured groups
pattern = r"<keyword>([^<]+)</keyword>"
xml = "<keyword>declarative</keyword> and <keyword>imperative</keyword>"
for match in re.finditer(pattern, xml):
    print(match.group(1))  # Prints content inside tags
    # declarative
    # imperative

# Replace matches
new_text = re.sub(r"cat", "dog", text)
print(new_text)  # "The dog sat on the mat. The dog was fat."
```

---

## 4. NLTK Operations

### 4.1 Iterate Over Sentences and Words in NLTK Corpus

**Question:** Access and iterate through sentences and words in an NLTK corpus.

**What's Required:** Use corpus methods like `.words()`, `.sents()`, `.fileids()`.

**Solution:**
```python
import nltk
from nltk.corpus import brown

# Download corpus
nltk.download('brown')

# Iterate over words
for word in brown.words()[:10]:  # First 10 words
    print(word)

# Get total word count
word_count = len(brown.words())
print(f"Total words: {word_count}")  # ~1,000,000 words

# Iterate over sentences
for sent in brown.sents()[:5]:  # First 5 sentences
    print(sent)
    print(" ".join(sent))

# Iterate over specific file
for word in brown.words(fileids='ca01'):
    print(word)
    
# Get all file IDs
file_ids = brown.fileids()
print(file_ids[:10])
```

### 4.2 Get Vocabulary (Set of Types) from NLTK Corpus

**Question:** Extract the unique words (types) from an NLTK corpus.

**What's Required:** Convert corpus words to a set to get unique types.

**Solution:**
```python
from nltk.corpus import brown
import nltk

nltk.download('brown')

# Get all tokens (words)
tokens = brown.words()
print(f"Total tokens: {len(tokens)}")  # ~1,000,000

# Get all types (unique words)
types = set(brown.words())
print(f"Total types: {len(types)}")  # ~50,000

# Convert to lowercase for case-insensitive vocabulary
vocab = set(word.lower() for word in brown.words())
print(f"Case-insensitive vocabulary size: {len(vocab)}")

# Check if word is in vocabulary
if "linguistics" in vocab:
    print("'linguistics' is in the vocabulary")

# Get alphabetically sorted vocabulary
sorted_vocab = sorted(vocab)
print(sorted_vocab[:10])  # First 10 words alphabetically
```

### 4.3 Get Set from NLTK Lexicon

**Question:** Access word lists from NLTK lexicons (like stopwords, names, etc.).

**What's Required:** Import and convert NLTK lexicons to sets.

**Solution:**
```python
import nltk
from nltk.corpus import stopwords, names, words

# Download lexicons
nltk.download('stopwords')
nltk.download('names')
nltk.download('words')

# Get English stopwords as a set
stop_words = set(stopwords.words('english'))
print(f"Number of stopwords: {len(stop_words)}")
print(stop_words)

# Get names
male_names = set(names.words('male.txt'))
female_names = set(names.words('female.txt'))
print(f"Male names: {len(male_names)}")
print(f"Female names: {len(female_names)}")

# Get English words
english_words = set(words.words())
print(f"English words: {len(english_words)}")

# Check if word is in lexicon
if "the" in stop_words:
    print("'the' is a stopword")
    
if "Python" in english_words:
    print("'Python' is in the dictionary")
```

### 4.4 Set Operations on Lexicons

**Question:** Perform intersection, union, and difference operations on sets/lexicons.

**What's Required:** Use set operations: `.intersection()`, `.union()`, `.difference()`, `-`, `&`, `|`.

**Solution:**
```python
import nltk
from nltk.corpus import names

nltk.download('names')

# Get name sets
male_names = set(names.words('male.txt'))
female_names = set(names.words('female.txt'))

# Intersection: Names that appear in both sets
common_names = male_names.intersection(female_names)
# Or: common_names = male_names & female_names
print(f"Unisex names: {len(common_names)}")
print(sorted(common_names)[:10])

# Union: All names (male or female)
all_names = male_names.union(female_names)
# Or: all_names = male_names | female_names
print(f"Total unique names: {len(all_names)}")

# Difference: Names that are only male
male_only = male_names.difference(female_names)
# Or: male_only = male_names - female_names
print(f"Male-only names: {len(male_only)}")

# Difference: Names that are only female
female_only = female_names.difference(male_names)
# Or: female_only = female_names - male_names
print(f"Female-only names: {len(female_only)}")

# Symmetric difference: Names in either set but not both
unique_to_each = male_names.symmetric_difference(female_names)
# Or: unique_to_each = male_names ^ female_names
print(f"Names unique to each gender: {len(unique_to_each)}")

# Check membership
if "Jordan" in common_names:
    print("Jordan is a unisex name")
```

---

## 5. BeautifulSoup and XML

### 5.1 Load XML from the Web into BeautifulSoup

**Question:** Fetch and parse XML content from a URL using BeautifulSoup.

**What's Required:** Use `requests` or `urllib` to fetch, then parse with BeautifulSoup.

**Solution:**
```python
from bs4 import BeautifulSoup
import requests

# Method 1: Using requests (recommended)
url = "http://example.com/data.xml"
response = requests.get(url)
soup = BeautifulSoup(response.content, 'lxml')

# Method 2: Using urllib
from urllib.request import urlopen
url = "http://example.com/data.xml"
html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')

# Parse XML string directly
xml_string = '''
<root>
    <item id="1">First item</item>
    <item id="2">Second item</item>
</root>
'''
soup = BeautifulSoup(xml_string, 'lxml')
print(soup.prettify())
```

### 5.2 Look for Particular Nodes in XML Tree

**Question:** Find specific elements/nodes in an XML document.

**What's Required:** Use BeautifulSoup methods like `.find()`, `.find_all()`, `.select()`.

**Solution:**
```python
from bs4 import BeautifulSoup

xml_example = '''
<text type="example"> 
    <sent type="declarative" n="1"> This is a <keyword>declarative</keyword> sentence<punct>.</punct></sent>
    <sent type="imperative" n="2"> Read this <keyword>imperative</keyword> sentence<punct>!</punct></sent>
    <sent type="interrogative" n="3"> Is this <keyword>interrogative</keyword> sentence okay<punct>?</punct></sent>
</text>
'''

soup = BeautifulSoup(xml_example, 'lxml')

# Find first occurrence of a tag
first_sent = soup.find('sent')
print(first_sent)

# Find all occurrences of a tag
all_sents = soup.find_all('sent')
print(f"Found {len(all_sents)} sentences")
for sent in all_sents:
    print(sent)

# Find with attributes
declarative = soup.find('sent', {'type': 'declarative'})
print(declarative)

# Find all with specific attribute
all_keywords = soup.find_all('keyword')
for kw in all_keywords:
    print(kw.text)

# CSS selectors
keywords = soup.select('keyword')
print([kw.text for kw in keywords])

# Complex selector
interrogative_sent = soup.select('sent[type="interrogative"]')
print(interrogative_sent)
```

### 5.3 Access Attributes and Text in XML

**Question:** Extract attribute values and text content from XML elements.

**What's Required:** Use `.attrs`, `.get()`, `.text`, `.string` on BeautifulSoup elements.

**Solution:**
```python
from bs4 import BeautifulSoup

xml_example = '''
<text type="example"> 
    <sent type="declarative" n="1"> This is a <keyword>declarative</keyword> sentence<punct>.</punct></sent>
    <sent type="imperative" n="2"> Read this <keyword>imperative</keyword> sentence<punct>!</punct></sent>
    <sent type="interrogative" n="3"> Is this <keyword>interrogative</keyword> sentence okay<punct>?</punct></sent>
</text>
'''

soup = BeautifulSoup(xml_example, 'lxml')

# Access attributes
text_element = soup.find('text')
print(text_element.get('type'))  # "example"
# Or: print(text_element['type'])

# Get all attributes as dictionary
first_sent = soup.find('sent')
print(first_sent.attrs)  # {'type': 'declarative', 'n': '1'}

# Access specific attribute
print(first_sent.get('type'))  # "declarative"
print(first_sent['n'])  # "1"

# Get text content
keywords = soup.find_all('keyword')
for kw in keywords:
    print(kw.text)  # or kw.get_text()
    # declarative
    # imperative
    # interrogative

# Get text with stripped whitespace
for kw in keywords:
    print(kw.text.strip())

# Get text of entire element (including nested)
sent = soup.find('sent')
print(sent.text.strip())
# "This is a declarative sentence."

# Extract both attribute and text
for sent in soup.find_all('sent'):
    sent_type = sent.get('type')
    sent_num = sent.get('n')
    sent_text = sent.text.strip()
    print(f"Sentence {sent_num} ({sent_type}): {sent_text}")
```

---

## 6. Complete Examples and Practice Problems

### Practice Problem 1: Word Frequency Analysis

**Question:** Load a text file, count word frequencies, and save the top 20 words to a new file.

**Solution:**
```python
from collections import Counter

# Read file
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenize and count
words = text.lower().split()
counts = Counter(words)

# Get top 20
top_20 = counts.most_common(20)

# Write to file
with open('top_words.txt', 'w', encoding='utf-8') as f:
    for word, count in top_20:
        f.write(f"{word}: {count}\n")
```

### Practice Problem 2: Extract Emails from Text

**Question:** Use regex to find all email addresses in a text file.

**Solution:**
```python
import re

# Read file
with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Regex pattern for emails
pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

# Find all emails
emails = re.findall(pattern, text)

# Print unique emails
unique_emails = set(emails)
for email in sorted(unique_emails):
    print(email)
```

### Practice Problem 3: XML Data Extraction

**Question:** Parse XML data to extract specific information and create a CSV.

**Solution:**
```python
from bs4 import BeautifulSoup
import csv

xml_data = '''
<catalog>
    <book id="1">
        <title>Python Programming</title>
        <author>John Doe</author>
        <price>29.99</price>
    </book>
    <book id="2">
        <title>Data Science</title>
        <author>Jane Smith</author>
        <price>39.99</price>
    </book>
</catalog>
'''

soup = BeautifulSoup(xml_data, 'lxml')

# Extract data
books = []
for book in soup.find_all('book'):
    book_data = {
        'id': book.get('id'),
        'title': book.find('title').text,
        'author': book.find('author').text,
        'price': float(book.find('price').text)
    }
    books.append(book_data)

# Write to CSV
with open('books.csv', 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['id', 'title', 'author', 'price'])
    writer.writeheader()
    writer.writerows(books)
```

### Practice Problem 4: NLTK Corpus Analysis

**Question:** Analyze a corpus to find the most common nouns using POS tagging.

**Solution:**
```python
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import brown
from collections import Counter

nltk.download('brown')
nltk.download('averaged_perceptron_tagger')

# Get sentences from Brown corpus
sentences = brown.sents()[:1000]  # First 1000 sentences

# Collect all nouns
nouns = []
for sent in sentences:
    # POS tag the sentence
    tagged = pos_tag(sent)
    # Extract nouns (NN, NNS, NNP, NNPS)
    nouns.extend([word.lower() for word, tag in tagged 
                  if tag.startswith('NN')])

# Count nouns
noun_counts = Counter(nouns)
print("Top 20 most common nouns:")
for noun, count in noun_counts.most_common(20):
    print(f"{noun}: {count}")
```

### Practice Problem 5: Text Preprocessing Pipeline

**Question:** Create a complete text preprocessing pipeline.

**Solution:**
```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """Complete text preprocessing pipeline"""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # 3. Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # 4. Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Tokenize
    tokens = word_tokenize(text)
    
    # 6. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    
    # 7. Remove short tokens
    tokens = [t for t in tokens if len(t) > 2]
    
    return tokens

# Example usage
text = """
Visit our website at http://example.com or email us at info@example.com!
This is a sample text with URLs, emails, and stopwords.
"""

cleaned_tokens = preprocess_text(text)
print(cleaned_tokens)
```

---

## Quick Reference Card

### Essential Imports
```python
import re
import nltk
from collections import Counter
from bs4 import BeautifulSoup
from nltk.corpus import brown, stopwords, names
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
```

### Common Patterns
```python
# File I/O
with open('file.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# Word counts
counts = Counter(words)
counts.most_common(10)

# Probability distribution
probs = {w: counts[w]/sum(counts.values()) for w in counts}

# Regex
pattern = r'\d+'
matches = re.findall(pattern, text)

# NLTK corpus
tokens = brown.words()
types = set(brown.words())

# BeautifulSoup
soup = BeautifulSoup(xml, 'lxml')
elements = soup.find_all('tag')
text = element.text
attr = element.get('attribute')
```

---

## Tips for Success

1. **Always use `with open()` for file operations** - it automatically closes files
2. **Use raw strings (r"") for regex patterns** - avoids escaping issues
3. **Remember to download NLTK data** - use `nltk.download()` for resources
4. **Convert to sets for fast lookups** - checking membership in sets is O(1)
5. **Use Counter for counting** - it's cleaner and more Pythonic than manual dictionaries
6. **Use f-strings for formatting** - modern and readable string formatting
7. **Don't forget encoding='utf-8'** - especially important for text files
8. **Test regex patterns incrementally** - build complex patterns step by step
9. **Use .lower() for case-insensitive comparisons** - normalizes text
10. **Remember list comprehensions** - concise and efficient for filtering/transforming

---

## Study Checklist

- [ ] Can extract data from nested dictionaries
- [ ] Can open and read files with proper encoding
- [ ] Can read files line-by-line vs all at once
- [ ] Can create word count dictionaries
- [ ] Can convert counts to probabilities
- [ ] Can sort words by frequency
- [ ] Can use NLTK for tokenization and POS tagging
- [ ] Can extract and manipulate substrings
- [ ] Can filter strings based on criteria
- [ ] Can convert between strings and numbers
- [ ] Can write strings with proper escaping
- [ ] Can write basic regex patterns
- [ ] Can use re.findall(), re.search(), re.finditer()
- [ ] Can iterate over NLTK corpus words and sentences
- [ ] Can get vocabulary (set) from corpus
- [ ] Can work with NLTK lexicons
- [ ] Can perform set operations (intersection, difference, union)
- [ ] Can load XML with BeautifulSoup
- [ ] Can find nodes in XML tree
- [ ] Can access XML attributes and text

---

**Good luck with your quiz! Practice these examples and understand the underlying concepts.**
