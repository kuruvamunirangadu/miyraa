"""Text preprocessing utilities for noisy, social media, and informal text.

Handles:
- Emoji processing and normalization
- Slang and abbreviation expansion
- Typo correction
- Social media artifacts (@mentions, hashtags, URLs)
- Repeated characters (e.g., "sooo" → "so")
- Case normalization
- Whitespace cleanup
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import html


# =============================================================================
# EMOJI HANDLING
# =============================================================================

# Common emoji to text mappings
EMOJI_TO_TEXT = {
    "😊": " happy ",
    "😀": " happy ",
    "😃": " happy ",
    "😄": " happy ",
    "😁": " happy ",
    "🙂": " happy ",
    "😍": " love ",
    "❤️": " love ",
    "💕": " love ",
    "😢": " sad ",
    "😭": " crying ",
    "😡": " angry ",
    "😠": " angry ",
    "😱": " scared ",
    "😨": " scared ",
    "😰": " anxious ",
    "😕": " confused ",
    "🤔": " thinking ",
    "😴": " tired ",
    "😎": " cool ",
    "🔥": " fire ",
    "👍": " thumbs up ",
    "👎": " thumbs down ",
    "🙏": " please ",
    "💯": " perfect ",
    "✨": " sparkle ",
}


def convert_emojis_to_text(text: str) -> str:
    """Convert common emojis to their text representations.
    
    Args:
        text: Input text with emojis
        
    Returns:
        Text with emojis converted to words
    """
    for emoji, replacement in EMOJI_TO_TEXT.items():
        text = text.replace(emoji, replacement)
    return text


def remove_emojis(text: str) -> str:
    """Remove all emojis from text.
    
    Args:
        text: Input text
        
    Returns:
        Text without emojis
    """
    # Remove emoji characters
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)


# =============================================================================
# SLANG AND ABBREVIATION EXPANSION
# =============================================================================

SLANG_DICT = {
    # Common internet slang
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "rofl": "rolling on floor laughing",
    "omg": "oh my god",
    "omfg": "oh my fucking god",
    "wtf": "what the fuck",
    "btw": "by the way",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "fyi": "for your information",
    "tbh": "to be honest",
    "afaik": "as far as I know",
    "asap": "as soon as possible",
    "brb": "be right back",
    "gtg": "got to go",
    "idk": "i don't know",
    "iirc": "if i recall correctly",
    "jk": "just kidding",
    "nvm": "never mind",
    "smh": "shaking my head",
    "thx": "thanks",
    "ty": "thank you",
    "ur": "your",
    "u": "you",
    "r": "are",
    "y": "why",
    "bc": "because",
    "bf": "boyfriend",
    "gf": "girlfriend",
    "dm": "direct message",
    "rn": "right now",
    "fr": "for real",
    "ngl": "not gonna lie",
    "irl": "in real life",
    "fomo": "fear of missing out",
    "yolo": "you only live once",
    "bae": "baby",
    "bff": "best friend forever",
    "pls": "please",
    "plz": "please",
    "sry": "sorry",
    "rly": "really",
    "thru": "through",
    "tho": "though",
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "dunno": "don't know",
    "lemme": "let me",
    "gimme": "give me",
    "kinda": "kind of",
    "sorta": "sort of",
    "outta": "out of",
    "ain't": "is not",
    "y'all": "you all",
    "c'mon": "come on",
    "cuz": "because",
    "prolly": "probably",
    "shoulda": "should have",
    "woulda": "would have",
    "coulda": "could have",
}


def expand_slang(text: str, slang_dict: Optional[Dict[str, str]] = None) -> str:
    """Expand slang and abbreviations to formal text.
    
    Args:
        text: Input text with slang
        slang_dict: Custom slang dictionary (uses default if None)
        
    Returns:
        Text with expanded slang
    """
    if slang_dict is None:
        slang_dict = SLANG_DICT
    
    words = text.split()
    expanded = []
    
    for word in words:
        word_lower = word.lower()
        # Check if word (without punctuation) is in slang dict
        clean_word = re.sub(r'[^\w]', '', word_lower)
        if clean_word in slang_dict:
            expanded.append(slang_dict[clean_word])
        else:
            expanded.append(word)
    
    return ' '.join(expanded)


# =============================================================================
# SOCIAL MEDIA ARTIFACT HANDLING
# =============================================================================

def normalize_social_media_text(text: str, 
                                remove_urls: bool = True,
                                remove_mentions: bool = False,
                                remove_hashtags: bool = False,
                                convert_hashtags: bool = True) -> str:
    """Normalize social media text by handling URLs, mentions, and hashtags.
    
    Args:
        text: Input text
        remove_urls: Remove HTTP/HTTPS URLs
        remove_mentions: Remove @mentions
        remove_hashtags: Remove #hashtags completely
        convert_hashtags: Convert #hashtags to regular words (ignored if remove_hashtags=True)
        
    Returns:
        Normalized text
    """
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' [URL] ', text)
        text = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' [URL] ', text)
    
    # Handle mentions
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
    else:
        text = re.sub(r'@\w+', ' [USER] ', text)
    
    # Handle hashtags
    if remove_hashtags:
        text = re.sub(r'#\w+', '', text)
    elif convert_hashtags:
        # Convert #CamelCase to "camel case"
        def convert_hashtag(match):
            tag = match.group(0)[1:]  # Remove #
            # Split on capital letters
            words = re.sub(r'([A-Z])', r' \1', tag).strip().lower()
            return ' ' + words + ' '
        text = re.sub(r'#\w+', convert_hashtag, text)
    
    return text


# =============================================================================
# CHARACTER REPETITION NORMALIZATION
# =============================================================================

def normalize_repetitions(text: str, max_repetitions: int = 2) -> str:
    """Normalize repeated characters (e.g., "sooooo" → "soo").
    
    Args:
        text: Input text
        max_repetitions: Maximum allowed character repetitions
        
    Returns:
        Text with normalized repetitions
    """
    # Pattern to match 3+ repeated characters
    pattern = re.compile(r'(.)\1{' + str(max_repetitions) + r',}')
    return pattern.sub(r'\1' * max_repetitions, text)


# =============================================================================
# HTML AND SPECIAL CHARACTER HANDLING
# =============================================================================

def unescape_html(text: str) -> str:
    """Unescape HTML entities.
    
    Args:
        text: Text with HTML entities
        
    Returns:
        Text with unescaped characters
    """
    return html.unescape(text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to their closest ASCII equivalents.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Normalize to NFKD form and encode to ASCII, ignoring non-ASCII chars
    nfkd_form = unicodedata.normalize('NFKD', text)
    return nfkd_form.encode('ASCII', 'ignore').decode('ASCII')


# =============================================================================
# CASE AND WHITESPACE NORMALIZATION
# =============================================================================

def normalize_case(text: str, mode: str = 'preserve') -> str:
    """Normalize text case.
    
    Args:
        text: Input text
        mode: 'lower', 'upper', 'title', or 'preserve'
        
    Returns:
        Text with normalized case
    """
    if mode == 'lower':
        return text.lower()
    elif mode == 'upper':
        return text.upper()
    elif mode == 'title':
        return text.title()
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace: remove extra spaces, tabs, newlines.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()


# =============================================================================
# COMPREHENSIVE PREPROCESSING PIPELINE
# =============================================================================

def preprocess_text(text: str,
                   lowercase: bool = False,
                   expand_slang: bool = True,
                   handle_emojis: str = 'convert',  # 'convert', 'remove', or 'keep'
                   normalize_social: bool = True,
                   normalize_reps: bool = True,
                   unescape: bool = True,
                   normalize_unicode_chars: bool = False,
                   max_length: Optional[int] = None) -> str:
    """Comprehensive text preprocessing pipeline.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        expand_slang: Expand slang and abbreviations
        handle_emojis: How to handle emojis ('convert', 'remove', or 'keep')
        normalize_social: Normalize social media artifacts
        normalize_reps: Normalize repeated characters
        unescape: Unescape HTML entities
        normalize_unicode_chars: Convert unicode to ASCII
        max_length: Maximum text length (truncate if exceeded)
        
    Returns:
        Preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unescape HTML
    if unescape:
        text = unescape_html(text)
    
    # Handle social media artifacts
    if normalize_social:
        text = normalize_social_media_text(text)
    
    # Handle emojis
    if handle_emojis == 'convert':
        text = convert_emojis_to_text(text)
    elif handle_emojis == 'remove':
        text = remove_emojis(text)
    # 'keep' does nothing
    
    # Normalize character repetitions
    if normalize_reps:
        text = normalize_repetitions(text)
    
    # Expand slang
    if expand_slang:
        text = expand_slang(text)
    
    # Normalize unicode
    if normalize_unicode_chars:
        text = normalize_unicode(text)
    
    # Normalize whitespace
    text = normalize_whitespace(text)
    
    # Case normalization
    if lowercase:
        text = text.lower()
    
    # Truncate if needed
    if max_length and len(text) > max_length:
        text = text[:max_length]
    
    return text


# =============================================================================
# QUALITY CHECKS
# =============================================================================

def is_valid_text(text: str, min_length: int = 3, max_length: int = 1000) -> Tuple[bool, str]:
    """Check if text is valid for emotion analysis.
    
    Args:
        text: Input text
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        
    Returns:
        (is_valid, reason) tuple
    """
    if not text or not isinstance(text, str):
        return False, "Empty or invalid input"
    
    text = text.strip()
    
    if len(text) < min_length:
        return False, f"Text too short (min {min_length} chars)"
    
    if len(text) > max_length:
        return False, f"Text too long (max {max_length} chars)"
    
    # Check if text is mostly non-alphabetic
    alpha_chars = sum(c.isalpha() for c in text)
    if alpha_chars / len(text) < 0.3:
        return False, "Text contains too few alphabetic characters"
    
    return True, "Valid"


if __name__ == "__main__":
    # Test preprocessing
    test_texts = [
        "OMG this is sooooo amazing!!! 😍😍😍 #BestDayEver",
        "lol u r so funny btw thx for the help",
        "@john check this out https://example.com/cool-stuff",
        "I'm gonna go to the store cuz we're outta milk",
        "YELLING IN ALL CAPS IS NOT COOL!!!",
    ]
    
    print("=== TEXT PREPROCESSING EXAMPLES ===\n")
    for text in test_texts:
        processed = preprocess_text(text, lowercase=True)
        print(f"Original:  {text}")
        print(f"Processed: {processed}")
        print()
