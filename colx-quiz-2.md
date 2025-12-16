# COLX 521 Quiz 2 Review - Q&A Study Guide

## Lecture 5: Regular Expressions

### Short Questions (1 point each)

### 1. List one advantage that regular expressions have over building an ML pattern-matcher, and one disadvantage.

**Advantage:** Regular expressions are deterministic, transparent, and require no training data. They execute quickly and their behavior is completely predictable and explainable.

**Disadvantage:** Regular expressions require manual pattern design and cannot generalize to unseen patterns. They are brittle and fail on variations that weren't explicitly coded, whereas ML models can learn patterns from examples and potentially handle novel cases.

### 2. There are two ways of discovering a pattern against the start of a string. Describe them.

**Method 1:** Use the `^` anchor at the beginning of the pattern (e.g., `^pattern`), which matches only at the start of the string.

**Method 2:** Use the `match()` function instead of `search()`. The `match()` function only checks at the beginning of the string by default.

```python
import re
# Method 1: Using ^ anchor
re.search(r'^pattern', text)

# Method 2: Using match()
re.match(r'pattern', text)
```

### 3. How would you detect tokens that contain non-English characters (assume that digits and punctuation like .?!;, are all "English")? Only use methods described in class.

Use the `[^...]` negated character class to match anything that is NOT in the English character set:

```python
import re
# Match tokens containing non-English characters
pattern = r'\w*[^A-Za-z0-9.?!;,\s]\w*'
```

Alternatively, match tokens and then check if they contain characters outside the ASCII range or the defined English set.

### 4. Describe the purpose of the various types of brackets in regexes, and how they differ.

**Square brackets `[]`:** Define character classes - match any ONE character from the set inside.
- Example: `[abc]` matches 'a', 'b', or 'c'

**Parentheses `()`:** Create capture groups - group parts of the pattern together and save matched text for later use.
- Example: `(cat|dog)` matches and captures either "cat" or "dog"

**Curly braces `{}`:** Specify exact repetition counts or ranges.
- Example: `a{3}` matches exactly 3 'a's; `a{2,5}` matches 2 to 5 'a's

### 5. Could you build a regular expression that detects regular expressions? If so, what kind of patterns would you look for? If not, what difficulties are there?

**Partially yes, but with limitations.** You could detect simple regex patterns by looking for:
- Special characters: `.*+?[](){}^$|\`
- Escape sequences: `\d`, `\w`, `\s`
- Character classes: `[...]`
- Quantifiers after characters

However, regular expressions cannot fully validate nested structures or balanced brackets, which are context-free properties. A regex cannot count bracket depth or ensure proper nesting, so it would fail on complex patterns with multiple levels of grouping.

### Medium Questions (2 points each)

### 6. Do you think that regexes could be used to discover palindromes (words or sentences)? What features would you need to use?

**Word palindromes:** No, regular expressions cannot reliably detect palindromes because regexes process text left-to-right and cannot reference previously matched content in reverse order. While you could use backreferences for very short palindromes (e.g., `(.)(.)` for 4-letter palindromes), this doesn't scale.

**Palindromic sentences:** No, this is even harder because you need to match word sequences in reverse order. Regexes lack the memory and bidirectional processing needed.

**Why not:** Palindrome detection requires comparing the string with its reverse, which is a context-sensitive operation beyond regular expression capabilities. You would need a full programming language to reverse and compare strings.

### 7. Imagine we have a spell-checker that can identify common misspellings by replacing certain letters with a capture group that contains letters nearby on the keyboard. How aggressive of a regex would we want to write? Are there any types of typos that regexes couldn't handle?

**Aggressiveness:** We should be moderately conservative - replacing 1-2 characters per word with keyboard-proximity groups. Being too aggressive (replacing many characters) would:
- Create too many false positives
- Make the regex extremely slow (exponential complexity)
- Match words that are completely different

For a 5-7 letter word, replacing 1-2 characters is reasonable. Longer words might allow 2-3 replacements.

**Typos regexes can't handle:**
- Transpositions (swapped adjacent letters): "teh" instead of "the"
- Missing letters: "speling" instead of "spelling"
- Extra letters: "spellling" instead of "spelling"
- Phonetic errors: "fone" instead of "phone"

While you could build specific patterns for these, a single regex can't handle all typo types efficiently.

### Long Questions (3 points each)

### 8. Discuss circumfixes, infixes, and reduplication in terms of regex suitability. Which are best suited for regexes and which features do they exploit?

**Circumfixes (e.g., German ge-t):** Well-suited for regexes. Can use patterns like:
```python
r'ge(\w+)t'  # Matches ge-[stem]-t
```
Exploits: Anchoring at both ends, capture groups for the middle stem.

**Infixes (e.g., cupful → cupsful):** Moderately suited. Can detect with:
```python
r'cup(s)ful'  # Detect infix position
```
Exploits: Specific position matching, capture groups. However, difficult to generalize across different words.

**Reduplication (e.g., aray → arayaray):** Best suited for regexes! Can use backreferences:
```python
r'(\w+)'  # Matches repeated sequences
r'(ba)basa'  # Partial reduplication
```
Exploits: Backreferences are perfect for detecting repetition. Can match full or partial reduplication patterns.

**Mostly unsuited:** Infixes are the hardest to generalize because they require knowing the exact insertion point within arbitrary stems. Regexes work best when the pattern position is predictable (like circumfixes and reduplication).

---

## Lecture 6: XML

### Short Questions (1 point each)

### 9. Why is XML well-suited to representing linguistic data?

XML is well-suited for linguistic data because:
1. **Hierarchical structure:** Linguistic data has natural hierarchies (documents → paragraphs → sentences → words → morphemes) Tree structure
2. **Flexibility:** Can represent overlapping annotations and multiple layers of annotation
3. **Self-documenting:** Tag names can be descriptive of linguistic categories (rich meta data)
4. **Standardized:** Widely supported, platform-independent format
5. **Extensible:** Can create custom tag sets for specific linguistic phenomena
6. **Preserves structure and metadata:** Can encode both content and linguistic annotations together

### 10. How would we find all images in an HTML document?

Using Beautiful Soup:

```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_content, 'html.parser')

# Method 1: Find all img tags
images = soup.find_all('img')

# Method 2: Get src attributes
image_urls = [img.get('src') for img in soup.find_all('img')]
```

### 11. What kinds of tags might be useful in the following text: "But you liked Rashomon!" "That's not how I remember it!" (describe at least two)

Useful tags could include:

1. **Dialogue tags:** `<dialogue speaker="person1">...</dialogue>` to mark who is speaking
2. **Entity tags:** `<movie>Rashomon</movie>` to identify the named entity (film title)
3. **Sentiment/emotion tags:** `<emphasis type="exclamation">` to mark emotional intensity
4. **Intertextuality tags:** To mark the self-referential joke (Rashomon is about perspective/memory)
5. **Quotation tags:** `<quote>...</quote>` to mark direct speech
6. **Reference tags:** To link "it" to its antecedent "Rashomon"

### 12. XML can be opened by most plain-text text editors. Name a benefit and a disadvantage of this feature.

**Benefit:** Human-readable and editable - no special software required. Easy to version control, debug, and manually inspect/modify. Promotes transparency and accessibility.

**Disadvantage:** Large files become unwieldy and slow to edit in text editors. Easy to accidentally break syntax (unclosed tags, typos). No validation or error-checking during manual editing. Can be verbose and inefficient for storage.

### 13. Beautiful Soup parses the children of a tag as a list. Why didn't they use a set? Give 2 reasons.

**Reason 1: Order matters.** XML/HTML elements appear in a specific sequence that carries meaning. A set would lose this ordering, making it impossible to reconstruct the document or process elements in the correct order.

**Reason 2: Duplicates are allowed.** The same type of element can appear multiple times as children (e.g., multiple `<p>` tags in a `<div>`). Sets don't allow duplicates, so we'd lose information.

**Additional reason:** Sequential access patterns - we often iterate through children in order, which lists handle efficiently.

### Medium Questions (2 points each)

### 14. Your NER model fails to recognize locations. What steps would you take to determine if the problem lies with the model, the training data, or both?

**Step 1: Inspect training data**
- Check if location annotations exist and are consistent
- Verify tag names are correct (e.g., `<LOCATION>` vs `<LOC>`)
- Count frequency of location examples vs. other entity types
- Look for annotation errors or inconsistencies

**Step 2: Analyze model predictions**
- Test on known locations to see if ANY are recognized
- Check if model confuses locations with other entity types
- Examine confidence scores

**Step 3: Data quality assessment**
- Calculate inter-annotator agreement if multiple annotators
- Check for missing or incomplete location annotations
- Verify geographic diversity in examples

**Resources needed:**
- Annotated training corpus (XML files)
- Evaluation dataset with gold-standard locations
- Confusion matrix to see error patterns
- XML parsing tools to analyze tag distribution
- Potentially: more training data with location examples

### 15. A colleague reorganizes an XML file's hierarchy and ordering, but other software no longer interprets it correctly. Identify two approaches to locate and fix the problem.

**Approach 1: Schema validation**
- Obtain or create the XML Schema/DTD that defines the expected structure
- Validate both the old and new versions against the schema
- Identify which structural rules were violated
- Restore the required element order and hierarchy

**Approach 2: Differential analysis**
- Use XML diff tools to compare old vs. new file structure
- Parse both files and compare the tree structures programmatically
- Check if required parent-child relationships changed
- Verify if mandatory elements are still present and in the right positions

```python
# Example approach
from bs4 import BeautifulSoup

old_soup = BeautifulSoup(old_xml, 'xml')
new_soup = BeautifulSoup(new_xml, 'xml')

# Compare structure
print(old_soup.prettify())
print(new_soup.prettify())
```

### Long Questions (3 points each)

### 16. Describe your process for creating a data validator for NQAXML (custom XML variant with specific rules).

**NQAXML Rules:**
- Tag names: all uppercase, max 10 characters
- No nested spans with identical tag names
- Required tags (like HTML)
- Every tag must have "lang" attribute

**Validation Process:**

**Step 1: Parse and basic validation**
```python
from bs4 import BeautifulSoup
import re

def validate_nqaxml(xml_file):
    with open(xml_file, 'r') as f:
        soup = BeautifulSoup(f, 'xml')
```

**Step 2: Tag name validation**
- Extract all tag names
- Check: `tag.isupper()` and `len(tag) <= 10`
- Report violations

**Step 3: Check for nested identical tags**
- Traverse tree recursively
- For each tag, check if any ancestor has the same name
- Flag violations

**Step 4: Verify required tags**
- Define list of mandatory tags (e.g., ROOT, HEADER, BODY)
- Check presence of each required tag
- Validate they appear at correct hierarchy levels

**Step 5: Validate "lang" attribute**
- Iterate through all tags
- Verify each has a "lang" attribute: `tag.get('lang')`
- Check attribute value is valid language code

**Step 6: Generate validation report**
- List all errors with line numbers/tag paths
- Categorize by error type
- Suggest fixes

```python
def validate_tag_names(soup):
    errors = []
    for tag in soup.find_all():
        if not tag.name.isupper():
            errors.append(f"Tag {tag.name} is not uppercase")
        if len(tag.name) > 10:
            errors.append(f"Tag {tag.name} exceeds 10 characters")
    return errors

def check_nested_duplicates(tag, ancestors=set()):
    errors = []
    if tag.name in ancestors:
        errors.append(f"Nested duplicate tag: {tag.name}")
    
    new_ancestors = ancestors | {tag.name}
    for child in tag.children:
        if hasattr(child, 'name'):
            errors.extend(check_nested_duplicates(child, new_ancestors))
    return errors

def validate_lang_attribute(soup):
    errors = []
    for tag in soup.find_all():
        if not tag.get('lang'):
            errors.append(f"Tag {tag.name} missing 'lang' attribute")
    return errors
```

---

## Lecture 7: Files and Preprocessing

### Short Questions (1 point each)

### 17. Can you think of any classes of words in English where the stem and the lemma will always be identical? Why is that of little interest to us?

**Word classes:** Function words (articles, prepositions, conjunctions) and words that don't inflect:
- Articles: "the", "a", "an"
- Prepositions: "of", "in", "on", "at"
- Some adverbs: "very", "always"
- Conjunctions: "and", "but", "or"

**Why of little interest:** These words are often considered stopwords and removed during text processing because they carry little semantic content. Since they don't change form, stemming/lemmatization provides no benefit. We're more interested in content words (nouns, verbs, adjectives) that DO inflect and where reducing to a common form helps with analysis.

### 18. What might the training data for a sentence segmenter look like? Do you think it would be easy or hard to train?

**Training data format:**
- Text with sentence boundaries marked (e.g., special tokens like `<S>` and `</S>`)
- Or: Each line is one sentence
- Features might include: punctuation context, capitalization patterns, abbreviation lists

**Difficulty: Moderately hard**

**Challenges:**
- Abbreviations (Dr., Mr., U.S.A.) have periods but aren't sentence ends
- Quotation marks can be tricky
- Numbers (3.14, 5.0) contain periods
- Some sentences don't end with periods (lists, headings)

**Why trainable:**
- Lots of freely available text data
- Patterns are somewhat regular
- Can use features like capitalization of next word
- But requires handling many edge cases

### 19. What impact does lemmatization or stemming have with respect to the Zipfian curve? How might that affect our algorithms?

**Impact on Zipfian curve:**
- Reduces vocabulary size by collapsing inflected forms
- Makes the curve "steeper" - fewer unique types, higher frequency for each type
- Example: "run", "runs", "running", "ran" all become "run"

**Effects on algorithms:**

**Positive:**
- Reduces sparsity - more data per word form
- Smaller vocabulary → less memory, faster processing
- Better generalization (connecting related word forms)
- More reliable statistics for each lemma

**Negative:**
- May lose grammatical information (tense, number, etc.)
- Some meaning distinctions lost (e.g., "bank" the institution vs. "bank" the riverbank)
- Can create false equivalences

### 20. If you were to encounter an alien text, which encoding might you want to use to digitize it? Explain briefly.

**Answer: UTF-32**

**Reasoning:**
- UTF-32 uses a fixed 4 bytes per character, supporting over 1 million possible characters
- Maximum coverage of any possible symbol system
- Simplifies processing because every character is the same byte length
- No need to worry about multi-byte sequences or continuation bytes
- Can handle completely unknown character sets

**Alternative: UTF-8**
- More storage efficient if the alien text uses mostly basic symbols
- But requires variable-length encoding knowledge

UTF-32 provides the most flexibility and simplicity for completely unknown writing systems.

### 21. What are two advantages of using .py files over .ipynb files for deployment, and two reasons why .ipynb files are preferred for prototyping?

**Advantages of .py for deployment:**

1. **Cleaner version control:** Plain text, easier to diff and merge, no JSON metadata clutter
2. **Standard execution:** Can run directly with python interpreter, easier to integrate into production pipelines, better for automated testing

**Advantages of .ipynb for prototyping:**

1. **Interactive development:** Can run cells individually, see outputs immediately, experiment with code snippets without running entire script
2. **Documentation integration:** Mix code, results, visualizations, and markdown explanations in one place - better for exploration and sharing results

### Medium Questions (2 points each)

### 22. Could we do lemmatization before machine translation? Provide arguments for and against.

**Assumption:** We're lemmatizing the source language before translation.

**Argument FOR:**

Reduces vocabulary size and sparsity, which could help the model learn better alignments between languages. Words like "running" and "runs" would both map to "run", providing more training examples for each lemma and potentially improving translation of rare inflected forms.

**Argument AGAINST:**

Loss of grammatical information that's crucial for translation. Many target languages require knowing the tense, number, or aspect of source words. For example:
- English "run" vs "ran" → Spanish "corre" vs "corrió"
- English "cat" vs "cats" → French "chat" vs "chats"

Lemmatization removes exactly the information needed to produce grammatically correct translations. The translation model would need to guess tense and number without evidence.

**Conclusion:** Generally not recommended for machine translation - grammatical features are crucial for accurate translation.

### 23. A linguist stores data in .docx files scattered on their desktop. What arguments would you make to convert to .tsv or .json, and how would you alleviate their worries about accessibility?

**Arguments for .tsv/.json:**

1. **Durability:** Plain text formats are future-proof and will be readable in 50 years, unlike proprietary formats
2. **Programmatic access:** Can analyze data with Python, R, or other tools for quantitative research
3. **Version control:** Can track changes, collaborate with others using Git
4. **Size:** Much smaller files, faster to backup and share
5. **Structured data:** Better for tabular data (speakers, utterances, annotations)

**Alleviating accessibility worries:**

1. **Spreadsheet programs:** "Excel, Google Sheets, and LibreOffice all open .tsv files perfectly - you can edit them just like a spreadsheet"
2. **Text editors:** "Even Notepad can open these files if needed"
3. **Conversion tools:** "I can provide simple scripts that convert .tsv to .xlsx whenever you need Excel format"
4. **Templates:** "I'll set up templates and show you exactly how to add new data"
5. **Demonstration:** Show them opening and editing .tsv in Excel - they'll see it's just as easy

### Long Questions (3 points each)

### 24. You find a corrupted file with flipped and deleted bits. How would you try to restore the data without knowing the encoding or language?

**Step 1: Determine byte structure**
- Try reading file in chunks of 8 bits (1 byte), 16 bits (2 bytes), 32 bits (4 bytes)
- Look for patterns: UTF-8 has specific bit patterns for multi-byte characters
  - Single byte: 0xxxxxxx
  - Two byte: 110xxxxx 10xxxxxx
  - Three byte: 1110xxxx 10xxxxxx 10xxxxxx
- UTF-16: Look for null bytes in regular positions (every other byte for ASCII-range characters)
- UTF-32: Every 4 bytes is one character

**Step 2: Detect deletions**
- Look for incomplete multi-byte sequences in UTF-8 (continuation byte without lead byte)
- Check for unexpected null bytes or broken patterns
- Use statistical analysis: natural language has expected character frequency distributions

**Step 3: Identify bit flips**
- Single bit flips create specific error patterns
- In UTF-8, flips might create invalid byte sequences
- Test hypothesis: try flipping each bit and see if valid character results
- Use checksums or redundancy if available

**Step 4: Statistical recovery**
- Even corrupted text has statistical properties
- Letter frequency analysis (e-t-a-o-i-n most common in English)
- Look for word-like patterns (consonant-vowel structure)
- Try multiple encoding hypotheses and see which produces most "natural" looking text

**Step 5: Incremental reconstruction**
- Identify clean sections (no corruption)
- Use clean sections to infer encoding
- Work outward from clean sections
- Use context to guess corrupted portions

```python
# Example: Test for UTF-8 validity
def is_valid_utf8_sequence(bytes):
    # Check if byte sequence follows UTF-8 rules
    if bytes[0] & 0b10000000 == 0:  # Single byte
        return len(bytes) == 1
    elif bytes[0] & 0b11100000 == 0b11000000:  # Two bytes
        return len(bytes) == 2 and bytes[1] & 0b11000000 == 0b10000000
    # ... more checks
```

---

## Lecture 8: Neural Networks and Deep Learning

### Short Questions (1 point each)

### 25. Briefly describe the role of "recurrence", and when we need it for linguistic data.

**Recurrence** allows neural networks to maintain memory of previous inputs by feeding outputs back as inputs to subsequent time steps. Each hidden state depends on both the current input and the previous hidden state.

**When needed for linguistic data:**
- Processing sequences where order matters (almost all language)
- Understanding context from earlier words in a sentence
- Tasks like language modeling, where next word depends on all previous words
- Handling variable-length inputs (sentences of different lengths)
- Capturing long-distance dependencies (e.g., subject-verb agreement across clauses)

**Example:** In "The dogs that were barking loudly were annoying", recurrence helps track that "dogs" (plural) requires "were" (plural verb), despite intervening words.

### 26. Why is stackability so important in neural networks?

**Stackability** (ability to chain multiple layers) is crucial because:

1. **Hierarchical feature learning:** Each layer learns increasingly abstract representations
   - Layer 1: Basic features (edges, simple patterns)
   - Layer 2: Combinations (shapes, character sequences)
   - Layer 3: Higher-level concepts (words, syntactic patterns)
   - Layer 4: Semantic meanings

2. **Increased representational power:** Deeper networks can learn more complex functions than shallow ones

3. **Modularity:** Can add/remove layers to adjust model complexity

4. **Transfer learning:** Pre-trained layers can be reused for new tasks

Without stackability, networks would be limited to simple, linear transformations.

### 27. Describe why "positional embeddings" allowed transformers to increase performance significantly over previous models.

**Positional embeddings** encode the position of each token in the sequence directly into the input representation.

**Why crucial for transformers:**

Transformers process all tokens simultaneously (parallel processing), unlike RNNs which process sequentially. This means transformers naturally have **no sense of word order**. Without positional information, "dog bites man" and "man bites dog" would be identical.

**How they help:**
- Add position information to word embeddings
- Allow model to learn position-dependent patterns
- Preserve word order information while maintaining parallel processing
- Enable attention mechanisms to consider both content AND position

**Performance impact:**
- Can process sequences much faster than RNNs (parallelization)
- Learn long-range dependencies better than RNNs
- Combined position + content information = better understanding

### 28. What is attention, and why is it important when processing linguistic data? Give a linguistic example.

**Attention** is a mechanism that allows the model to focus on different parts of the input when producing each output, by learning weighted combinations of input representations.

**Why important for linguistic data:**
- Not all words are equally relevant for understanding each part of a sentence
- Captures dependencies between distant words
- Enables model to "look back" at relevant context
- Handles variable-length inputs effectively

**Linguistic example (not from class):**

Sentence: "The trophy didn't fit in the suitcase because it was too large."

When processing "it", attention helps the model determine whether "it" refers to "trophy" or "suitcase" by attending to "large" (trophy is large) vs if the sentence said "too small" (suitcase is small). The model learns to attend to the relevant antecedent based on semantic clues.

### 29. What do encoders "encode", and how does it help linguistic processing?

**Encoders encode** input sequences into dense, continuous vector representations (embeddings) that capture semantic and syntactic information.

**What they encode:**
- Word meanings in context
- Grammatical relationships
- Semantic features
- Positional information
- Long-range dependencies

**How it helps linguistic processing:**

1. **Dimensionality reduction:** Convert sparse one-hot encodings or discrete tokens into dense, informative vectors
2. **Context sensitivity:** Same word gets different representations in different contexts
3. **Semantic similarity:** Similar meanings have similar vector representations
4. **Task-agnostic features:** Learned representations can transfer to multiple tasks
5. **Efficient computation:** Dense vectors are more efficient for neural computation than sparse representations

**Example:** BERT encoder creates contextual embeddings where "bank" has different representations in "river bank" vs. "savings bank".

### Medium Questions (2 points each)

### 30. You're building a machine translator with transformers. The data has extra annotation: POS tags, lemmas, dependency information. How might you incorporate this information?

**Approach 1: Multi-input encoding**
- Create separate embedding layers for words, POS tags, lemmas
- Concatenate or sum these embeddings as input to encoder
- Model learns to use different information types

```python
word_embed = WordEmbedding(token)
pos_embed = POSEmbedding(pos_tag)
lemma_embed = LemmaEmbedding(lemma)
combined = word_embed + pos_embed + lemma_embed
```

**Approach 2: Auxiliary tasks**
- Add prediction heads for POS tagging, lemmatization during training
- Multi-task learning forces encoder to learn these features
- Main translation benefits from richer representations

**Approach 3: Dependency-aware attention**
- Modify attention mechanism to bias toward syntactic dependencies
- Use dependency parse to create attention masks
- Model attends more to syntactically related words

**Approach 4: Specialized tokens**
- Add special tokens encoding annotations: `<VERB>`, `<NOUN>`, etc.
- Interleave with regular tokens or use parallel streams

**Benefits:**
- Reduces ambiguity (POS helps with word sense)
- Preserves grammatical structure (dependencies)
- Handles morphologically rich languages better

### 31. You're building a speech recognizer that produces too many fillers ("uh", "um"). How would you approach solving this problem?

**Problem analysis:** The model is working correctly (accurately transcribing speech) but producing output that's too faithful to actual speech.

**Approach 1: Post-processing filter**
- Simplest solution: Remove fillers after transcription
- Create list of filler words/patterns
- Filter output before returning to user
```python
fillers = ['uh', 'um', 'er', 'ah', 'like', 'you know']
clean_text = ' '.join([word for word in text.split() if word.lower() not in fillers])
```

**Approach 2: Training data modification**
- Create new training set with fillers removed from transcripts
- Retrain model on cleaned transcriptions
- Model learns to not output fillers
- **Caution:** May affect accuracy on actual filler words that are meaningful

**Approach 3: Two-stage processing**
- Stage 1: Transcribe everything (including fillers)
- Stage 2: Separate model to classify and remove fillers
- Allows user to choose cleaned or full transcription

**Approach 4: Conditional generation**
- Add a "clean speech" flag/token during training
- Model learns both behaviors
- At inference, use flag to request clean output

**Recommendation:** Start with post-processing (easiest, no retraining). If that's insufficient, retrain with cleaned data. Keep option for users who want verbatim transcription.

### Long Questions (3 points each)

### 32. Your NLP pipeline produces worrying biases against certain communities. How would you search for the source, propose fixes, present benefits to your boss, and justify costs?

**Step 1: Identify and measure the bias**
- Define what bias means in this context (e.g., sentiment toward community X is more negative)
- Create test sets with parallel examples (same sentence, different demographic groups)
- Quantify bias magnitude: sentiment scores, word associations, output distributions
- Document specific examples of biased outputs

**Step 2: Search for the source**

**A. Training data analysis:**
- Examine corpus for representation imbalances
- Check for stereotypical associations in text
- Measure demographic representation in data
- Look for historical biases in source material

**B. Model architecture:**
- Check if certain features encode protected attributes
- Analyze attention patterns for bias signals
- Examine embedding spaces for stereotypical clusters

**C. Evaluation/deployment:**
- Check if evaluation metrics miss bias
- See if user feedback loops amplify bias

**Step 3: Propose fixes**

**Data-level interventions:**
- Augment training data with counter-stereotypical examples
- Balance representation across communities
- Remove overtly biased examples
- Use data from diverse sources

**Model-level interventions:**
- Add fairness constraints during training
- Debias word embeddings
- Adversarial debiasing (prevent model from encoding protected attributes)
- Fine-tune on balanced dataset

**Output-level interventions:**
- Add bias detection filters
- Modify outputs to reduce bias while preserving utility
- Flag potentially biased outputs for human review

**Step 4: Present benefits to boss**

**Business case:**
1. **Legal/compliance:** Biased AI systems create liability and regulatory risk
2. **Reputation:** Bias scandals damage brand and customer trust
3. **Market access:** Biased systems exclude potential users/customers
4. **Product quality:** Debiasing often improves overall performance
5. **Employee morale:** Engineers want to work on ethical products
6. **Long-term sustainability:** Responsible AI is increasingly expected by customers

**Step 5: Justify costs**

**Cost breakdown:**
- Engineer time for analysis and implementation
- Additional compute for retraining
- New data collection/annotation if needed
- Ongoing monitoring and testing

**ROI arguments:**
- Prevention cheaper than PR disaster recovery
- Expanding to underserved markets
- Competitive advantage as responsible AI leader
- Reduced legal risk (hard to quantify but real)

**Phased approach:**
- Start with low-cost interventions (post-processing filters)
- Measure impact
- Justify larger investments based on demonstrated improvement

---

*End of Study Guide*
