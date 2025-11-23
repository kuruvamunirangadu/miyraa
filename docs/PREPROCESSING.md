# Preprocessing Rules & Guidelines

Complete documentation of text preprocessing strategies, augmentation techniques, and data quality standards for the Miyraa NLP Emotion Engine.

---

## Table of Contents

1. [Text Normalization](#text-normalization)
2. [Noisy Text Preprocessing](#noisy-text-preprocessing)
3. [Text Augmentation](#text-augmentation)
4. [Emotion Taxonomy](#emotion-taxonomy)
5. [Data Quality Standards](#data-quality-standards)
6. [Pipeline Architecture](#pipeline-architecture)

---

## Text Normalization

### Core Principles

- **Preserve Meaning**: Never alter text in ways that change emotional content
- **Consistency**: Apply same rules across training and inference
- **Reversibility**: Document transformations for debugging

### Standard Preprocessing Steps

#### 1. HTML Unescaping
```python
# Convert HTML entities to characters
"&amp; &lt; &gt;" → "& < >"
"&#8217;" → "'"
```

**Why**: User-generated content often contains HTML entities from web scraping.

#### 2. Whitespace Normalization
```python
# Multiple spaces → single space
"I    am    happy" → "I am happy"

# Remove leading/trailing whitespace
"  Hello world  " → "Hello world"

# Convert tabs/newlines to spaces
"Line1\n\nLine2" → "Line1 Line2"
```

**Why**: Inconsistent whitespace breaks tokenization and increases vocabulary size.

#### 3. Case Normalization
```python
# Lowercase (default for embeddings)
"I Love This!" → "i love this!"

# Preserve case (for BERT-style models)
"I Love This!" → "I Love This!"
```

**When to lowercase**: Always for bag-of-words, TF-IDF, word embeddings. Optional for transformer models (BERT handles case internally).

---

## Noisy Text Preprocessing

### Social Media Artifacts

#### URLs
```python
# Replace with token
"Check out http://example.com/cool" → "Check out [URL]"
```

**Why**: URLs add no emotional information and explode vocabulary.

#### Mentions
```python
# Replace with token
"@john this is amazing" → "[USER] this is amazing"

# Or remove entirely
"@john this is amazing" → "this is amazing"
```

**When to remove**: If mention is not emotionally relevant.

#### Hashtags
```python
# Convert to words (preserve meaning)
"#BestDayEver" → "best day ever"
"#blessed" → "blessed"

# Or remove if spammy
"#like #follow #subscribe" → ""
```

**Best practice**: Convert hashtags that express emotion, remove promotional ones.

### Emoji Processing

#### Option 1: Convert to Text (Recommended)
```python
"I love this! 😍😍" → "I love this! love love"
"So happy 😊" → "So happy happy"
```

**Pros**: Preserves emotional signal
**Cons**: Can be verbose

#### Option 2: Remove Entirely
```python
"I love this! 😍😍" → "I love this!"
```

**Pros**: Cleaner text
**Cons**: Loses emotional information

#### Option 3: Keep as Unicode
```python
"I love this! 😍😍" → "I love this! 😍😍"
```

**Pros**: Full preservation
**Cons**: Tokenizer must handle Unicode

**Recommendation**: Convert to text for emotion classification (preserves signal).

### Character Repetition

```python
# Normalize excessive repetition
"sooooo happy" → "soo happy"
"yesssssss" → "yess"
"!!!!!" → "!!"
```

**Why**: Repetition adds emphasis but shouldn't create new vocabulary items.

**Rule**: Keep max 2 repetitions (configurable).

### Slang & Abbreviations

```python
# Expand common slang
"lol u r so funny" → "laughing out loud you are so funny"
"omg this is gr8" → "oh my god this is great"
"idk tbh" → "i don't know to be honest"
```

**Dictionary**: See `src/nlp/preprocessing/text_cleaner.py` for full list.

**When to expand**: Always for emotion classification (improves model understanding).

### Typos & Misspellings

**Current approach**: No auto-correction (risk of changing meaning)

**Future enhancement**: 
- Use spell checker with confidence threshold
- Only correct if >95% confidence
- Never correct slang or domain-specific terms

---

## Text Augmentation

### Strategies

#### 1. Synonym Replacement (SR)
```python
# Original
"I am so happy today"

# Augmented
"I am so joyful today"
"I am so cheerful today"
```

**Parameters**:
- `alpha_sr = 0.1`: Replace 10% of words
- `exclude_stopwords = True`: Don't replace "the", "a", etc.

**Use case**: Increase vocabulary diversity without changing meaning.

#### 2. Random Insertion (RI)
```python
# Original
"I am happy"

# Augmented
"I am joyful happy"
"I am cheerful happy"
```

**Parameters**:
- `alpha_ri = 0.1`: Insert synonyms for 10% of words

**Use case**: Add redundancy and context.

#### 3. Random Swap (RS)
```python
# Original
"I am so happy today"

# Augmented
"I so am happy today"
"I am happy so today"
```

**Parameters**:
- `alpha_rs = 0.1`: Swap 10% of word pairs

**Use case**: Test model robustness to word order.

#### 4. Random Deletion (RD)
```python
# Original
"I am so happy today"

# Augmented
"I so happy today"
"I am happy today"
```

**Parameters**:
- `alpha_rd = 0.1`: Delete each word with 10% probability

**Use case**: Force model to rely on key words only.

#### 5. Back-Translation (BT)
```python
# Original (English)
"I am very happy today"

# Translate to French
"Je suis très heureux aujourd'hui"

# Translate back to English
"I am extremely happy today"
```

**Models**: MarianMT, Google Translate, DeepL

**Use case**: Generate paraphrases with different phrasing.

**Status**: Placeholder implementation (requires translation models).

### Augmentation Guidelines

**When to augment**:
- Class imbalance (oversample minority classes)
- Small dataset (< 1000 samples per class)
- Rare emotion categories (fear, disgust, calm)

**When NOT to augment**:
- Validation/test sets (keep original)
- Already balanced dataset
- Safety-critical data (augmentation might alter toxicity)

**Best practices**:
```python
# Combine multiple methods
augment_text(text, method="all", num_aug=3)

# Balance classes
augment_dataset_balanced(texts, labels, target_count=500)

# Check similarity (avoid duplicates)
if not is_similar(original, augmented, threshold=0.8):
    dataset.append(augmented)
```

---

## Emotion Taxonomy

### Core Emotions (11 Categories)

| Emotion | VAD Values | Description |
|---------|-----------|-------------|
| **joy** | V:+0.85, A:+0.65, D:+0.70 | Happiness, contentment, pleasure |
| **love** | V:+0.80, A:+0.40, D:+0.50 | Affection, care, attachment |
| **surprise** | V:+0.20, A:+0.80, D:+0.30 | Unexpected events, shock |
| **sadness** | V:-0.70, A:-0.30, D:-0.50 | Grief, disappointment, melancholy |
| **anger** | V:-0.60, A:+0.70, D:+0.60 | Frustration, rage, irritation |
| **fear** | V:-0.65, A:+0.60, D:-0.60 | Anxiety, terror, worry |
| **disgust** | V:-0.60, A:+0.35, D:+0.40 | Revulsion, contempt, aversion |
| **calm** | V:+0.40, A:-0.60, D:+0.30 | Peace, tranquility, relaxation |
| **excitement** | V:+0.70, A:+0.85, D:+0.65 | Anticipation, enthusiasm |
| **confusion** | V:-0.20, A:+0.30, D:-0.40 | Uncertainty, bewilderment |
| **neutral** | V:0.00, A:0.00, D:0.00 | No strong emotion |

### VAD Dimensions

**Valence**: Pleasantness (-1 = unpleasant, +1 = pleasant)
**Arousal**: Intensity (-1 = calm, +1 = excited)
**Dominance**: Control (-1 = submissive, +1 = dominant)

### GoEmotions Mapping

Maps 27 GoEmotions categories to 11 Miyraa categories with confidence weights:

```python
"disappointment": [("sadness", 0.8), ("anger", 0.2)]
"relief": [("calm", 0.7), ("joy", 0.3)]
"pride": [("joy", 0.6), ("love", 0.2), ("excitement", 0.2)]
```

**See**: `src/nlp/data/taxonomy.py` for complete mapping.

---

## Data Quality Standards

### Minimum Requirements

#### Text Length
- **Minimum**: 3 characters
- **Maximum**: 1000 characters
- **Optimal**: 10-200 characters

#### Alphabetic Content
- **Minimum**: 30% alphabetic characters
- **Why**: Ensures text is human-readable, not just symbols/numbers

#### Encoding
- **Standard**: UTF-8
- **Fallback**: ASCII with Unicode normalization

### Quality Checks

```python
# Check before adding to dataset
is_valid, reason = is_valid_text(text, min_length=3, max_length=1000)

if not is_valid:
    print(f"Rejected: {reason}")
```

### Label Quality

#### Confidence Scores
- **High confidence** (>0.9): Single clear emotion
- **Medium confidence** (0.7-0.9): Dominant emotion with secondary
- **Low confidence** (<0.7): Mixed or ambiguous emotions

#### Edge Cases
- **Sarcasm**: Mark with lower confidence
- **Mixed emotions**: Label primary + secondary
- **Context-dependent**: Add notes field

---

## Pipeline Architecture

### Training Pipeline

```
Raw Text
    ↓
[HTML Unescape]
    ↓
[Social Media Normalization]
    ↓
[Emoji Conversion]
    ↓
[Character Repetition Normalization]
    ↓
[Slang Expansion]
    ↓
[Whitespace Normalization]
    ↓
[Case Normalization]
    ↓
[Text Augmentation] (optional)
    ↓
[Tokenization]
    ↓
Model Input
```

### Inference Pipeline

```
User Input
    ↓
[Quality Check]
    ↓
[Same Preprocessing Steps as Training]
    ↓
[Tokenization]
    ↓
Model Prediction
    ↓
[Threshold Calibration]
    ↓
Final Output
```

### Configuration

```python
# Minimal preprocessing (preserve original)
preprocess_text(
    text,
    lowercase=False,
    expand_slang=False,
    handle_emojis='keep',
    normalize_social=False
)

# Standard preprocessing (recommended)
preprocess_text(
    text,
    lowercase=True,
    expand_slang=True,
    handle_emojis='convert',
    normalize_social=True,
    normalize_reps=True
)

# Aggressive preprocessing (maximum normalization)
preprocess_text(
    text,
    lowercase=True,
    expand_slang=True,
    handle_emojis='remove',
    normalize_social=True,
    normalize_reps=True,
    normalize_unicode_chars=True,
    max_length=500
)
```

---

## Code Examples

### Full Preprocessing
```python
from src.nlp.preprocessing.text_cleaner import preprocess_text

text = "OMG @john this is sooooo amazing!!! 😍😍 #BestDay"
clean = preprocess_text(text, lowercase=True)
# Output: "oh my god [USER] this is soo amazing !! love love best day"
```

### Augmentation
```python
from src.nlp.preprocessing.augmentation import augment_text

text = "I am so happy today"
augmented = augment_text(text, method="sr", num_aug=3)
# Output: ["I am so joyful today", "I am so cheerful today", ...]
```

### Balanced Dataset
```python
from src.nlp.preprocessing.augmentation import augment_dataset_balanced

texts = ["happy text", "sad text", ...]
labels = ["joy", "sadness", ...]

aug_texts, aug_labels = augment_dataset_balanced(
    texts, labels, target_count=500
)
```

---

## Version History

- **v1.0** (2024-01): Initial preprocessing rules
- **v1.1** (2024-01): Added emoji conversion, slang expansion
- **v1.2** (2024-01): Added text augmentation strategies
- **v1.3** (2024-01): Comprehensive taxonomy integration

---

## References

- **VAD Model**: Russell, J. A. (1980). A circumplex model of affect.
- **GoEmotions**: Demszky et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions
- **EDA**: Wei & Zou (2019). EDA: Easy Data Augmentation Techniques
- **Text Normalization**: Best practices from spaCy, NLTK, HuggingFace

---

## Copilot Context

This documentation provides comprehensive preprocessing guidelines for the Miyraa emotion classification system. Key points for Copilot:

1. **Always preprocess consistently** between training and inference
2. **Preserve emotional meaning** - never alter sentiment
3. **Use augmentation** for class balancing and robustness
4. **Follow taxonomy** for consistent labeling
5. **Check quality** before adding samples to dataset
6. **Document edge cases** with confidence scores and notes

For implementation details, see:
- `src/nlp/preprocessing/text_cleaner.py` - Text normalization
- `src/nlp/preprocessing/augmentation.py` - Data augmentation
- `src/nlp/data/taxonomy.py` - Emotion labels and mappings
- `scripts/generate_curated_samples.py` - Example dataset creation
