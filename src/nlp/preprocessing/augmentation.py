"""Text augmentation utilities for training data expansion.

Strategies:
- Synonym replacement using contextual understanding
- Back-translation (if translation models available)
- Random insertion, swap, and deletion
- Contextual word replacement
- Paraphrase generation
"""

import random
import re
from typing import List, Optional, Tuple
import warnings


# =============================================================================
# SYNONYM REPLACEMENT
# =============================================================================

# Basic synonym dictionary for common words
# In production, use WordNet or a better synonym database
SYNONYM_DICT = {
    "happy": ["joyful", "cheerful", "pleased", "content", "delighted"],
    "sad": ["unhappy", "sorrowful", "dejected", "melancholy", "down"],
    "angry": ["mad", "furious", "irritated", "annoyed", "upset"],
    "good": ["great", "excellent", "fine", "wonderful", "nice"],
    "bad": ["poor", "terrible", "awful", "horrible", "unpleasant"],
    "big": ["large", "huge", "enormous", "massive", "giant"],
    "small": ["little", "tiny", "miniature", "petite", "compact"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
    "slow": ["sluggish", "gradual", "leisurely", "unhurried", "delayed"],
    "love": ["adore", "cherish", "treasure", "appreciate", "enjoy"],
    "hate": ["detest", "despise", "loathe", "dislike", "abhor"],
    "beautiful": ["gorgeous", "lovely", "stunning", "attractive", "pretty"],
    "ugly": ["unattractive", "hideous", "unsightly", "homely", "plain"],
    "smart": ["intelligent", "clever", "bright", "brilliant", "sharp"],
    "stupid": ["foolish", "dumb", "idiotic", "silly", "senseless"],
    "easy": ["simple", "effortless", "straightforward", "uncomplicated", "basic"],
    "hard": ["difficult", "challenging", "tough", "demanding", "complex"],
    "new": ["fresh", "recent", "modern", "latest", "novel"],
    "old": ["ancient", "aged", "elderly", "vintage", "antique"],
    "great": ["excellent", "superb", "outstanding", "remarkable", "fantastic"],
}


def get_synonyms(word: str) -> List[str]:
    """Get synonyms for a word.
    
    Args:
        word: Input word
        
    Returns:
        List of synonyms
    """
    word_lower = word.lower()
    return SYNONYM_DICT.get(word_lower, [])


def synonym_replacement(text: str, n: int = 1, exclude_stopwords: bool = True) -> str:
    """Replace n random words with their synonyms.
    
    Args:
        text: Input text
        n: Number of words to replace
        exclude_stopwords: Don't replace common stopwords
        
    Returns:
        Augmented text
    """
    # Basic stopwords to exclude
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "my", "your",
        "his", "her", "its", "our", "their"
    }
    
    words = text.split()
    replaceable_indices = []
    
    # Find indices of replaceable words
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?;:"\'')
        if word_lower not in stopwords or not exclude_stopwords:
            if word_lower in SYNONYM_DICT:
                replaceable_indices.append(i)
    
    # Replace n random words
    if not replaceable_indices:
        return text
    
    n = min(n, len(replaceable_indices))
    indices_to_replace = random.sample(replaceable_indices, n)
    
    for idx in indices_to_replace:
        word = words[idx]
        word_lower = word.lower().strip('.,!?;:"\'')
        synonyms = get_synonyms(word_lower)
        
        if synonyms:
            synonym = random.choice(synonyms)
            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()
            words[idx] = synonym
    
    return ' '.join(words)


# =============================================================================
# RANDOM INSERTION
# =============================================================================

def random_insertion(text: str, n: int = 1) -> str:
    """Randomly insert n synonyms of random words.
    
    Args:
        text: Input text
        n: Number of words to insert
        
    Returns:
        Augmented text
    """
    words = text.split()
    
    for _ in range(n):
        # Find a word with synonyms
        candidates = [w for w in words if w.lower().strip('.,!?;:"\'') in SYNONYM_DICT]
        if not candidates:
            break
        
        word = random.choice(candidates)
        word_lower = word.lower().strip('.,!?;:"\'')
        synonyms = get_synonyms(word_lower)
        
        if synonyms:
            synonym = random.choice(synonyms)
            random_idx = random.randint(0, len(words))
            words.insert(random_idx, synonym)
    
    return ' '.join(words)


# =============================================================================
# RANDOM SWAP
# =============================================================================

def random_swap(text: str, n: int = 1) -> str:
    """Randomly swap positions of n word pairs.
    
    Args:
        text: Input text
        n: Number of swaps
        
    Returns:
        Augmented text
    """
    words = text.split()
    
    if len(words) < 2:
        return text
    
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    
    return ' '.join(words)


# =============================================================================
# RANDOM DELETION
# =============================================================================

def random_deletion(text: str, p: float = 0.1) -> str:
    """Randomly delete words with probability p.
    
    Args:
        text: Input text
        p: Probability of deleting each word
        
    Returns:
        Augmented text
    """
    words = text.split()
    
    # If only one word, don't delete
    if len(words) == 1:
        return text
    
    new_words = [word for word in words if random.random() > p]
    
    # If all words deleted, return a random word
    if not new_words:
        return random.choice(words)
    
    return ' '.join(new_words)


# =============================================================================
# COMPREHENSIVE AUGMENTATION
# =============================================================================

def augment_text(text: str,
                method: str = "random",
                num_aug: int = 1,
                alpha_sr: float = 0.1,  # Synonym replacement rate
                alpha_ri: float = 0.1,  # Random insertion rate
                alpha_rs: float = 0.1,  # Random swap rate
                alpha_rd: float = 0.1) -> List[str]:
    """Apply augmentation to generate new training samples.
    
    Args:
        text: Input text
        method: Augmentation method ('sr', 'ri', 'rs', 'rd', 'random', 'all')
        num_aug: Number of augmented samples to generate
        alpha_sr: Synonym replacement rate (fraction of words)
        alpha_ri: Random insertion rate
        alpha_rs: Random swap rate
        alpha_rd: Random deletion probability
        
    Returns:
        List of augmented texts
    """
    augmented = []
    words = text.split()
    num_words = len(words)
    
    for _ in range(num_aug):
        if method == "sr":
            # Synonym replacement
            n_sr = max(1, int(alpha_sr * num_words))
            aug = synonym_replacement(text, n=n_sr)
        
        elif method == "ri":
            # Random insertion
            n_ri = max(1, int(alpha_ri * num_words))
            aug = random_insertion(text, n=n_ri)
        
        elif method == "rs":
            # Random swap
            n_rs = max(1, int(alpha_rs * num_words))
            aug = random_swap(text, n=n_rs)
        
        elif method == "rd":
            # Random deletion
            aug = random_deletion(text, p=alpha_rd)
        
        elif method == "random":
            # Randomly choose one method
            method_choice = random.choice(["sr", "ri", "rs", "rd"])
            aug = augment_text(text, method=method_choice, num_aug=1,
                             alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                             alpha_rs=alpha_rs, alpha_rd=alpha_rd)[0]
        
        else:  # "all" - apply multiple methods
            aug = text
            # Apply SR
            n_sr = max(1, int(alpha_sr * num_words))
            aug = synonym_replacement(aug, n=n_sr)
            # Apply RI
            n_ri = max(1, int(alpha_ri * num_words))
            aug = random_insertion(aug, n=n_ri)
            # Apply RS
            n_rs = max(1, int(alpha_rs * num_words))
            aug = random_swap(aug, n=n_rs)
        
        augmented.append(aug)
    
    return augmented


# =============================================================================
# BACK-TRANSLATION (PLACEHOLDER)
# =============================================================================

def back_translate(text: str, 
                  source_lang: str = "en",
                  intermediate_lang: str = "fr") -> str:
    """Back-translate text through an intermediate language.
    
    This is a placeholder. In production, use:
    - transformers MarianMT models
    - Google Translate API
    - DeepL API
    
    Args:
        text: Input text
        source_lang: Source language code
        intermediate_lang: Intermediate language for back-translation
        
    Returns:
        Back-translated text
    """
    warnings.warn(
        "Back-translation not implemented. "
        "Install transformers and use MarianMT models for production."
    )
    return text


# =============================================================================
# BALANCED AUGMENTATION
# =============================================================================

def augment_dataset_balanced(texts: List[str],
                            labels: List[str],
                            target_count: int,
                            method: str = "random") -> Tuple[List[str], List[str]]:
    """Augment dataset to balance class distribution.
    
    Args:
        texts: List of input texts
        labels: List of corresponding labels
        target_count: Target number of samples per class
        method: Augmentation method
        
    Returns:
        (augmented_texts, augmented_labels) tuple
    """
    from collections import Counter
    
    # Count samples per label
    label_counts = Counter(labels)
    
    augmented_texts = list(texts)
    augmented_labels = list(labels)
    
    # Augment underrepresented classes
    for label, count in label_counts.items():
        if count < target_count:
            # Get all texts with this label
            label_texts = [t for t, l in zip(texts, labels) if l == label]
            
            # Calculate how many augmentations needed
            needed = target_count - count
            
            # Generate augmentations
            for i in range(needed):
                # Sample a random text from this class
                source_text = random.choice(label_texts)
                # Augment it
                aug_texts = augment_text(source_text, method=method, num_aug=1)
                # Add to dataset
                augmented_texts.append(aug_texts[0])
                augmented_labels.append(label)
    
    return augmented_texts, augmented_labels


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_similar(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two texts are too similar (simple word overlap check).
    
    Args:
        text1: First text
        text2: Second text
        threshold: Similarity threshold (0-1)
        
    Returns:
        True if texts are too similar
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return False
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    jaccard = intersection / union if union > 0 else 0
    return jaccard > threshold


if __name__ == "__main__":
    # Test augmentation
    test_text = "I am so happy today because the weather is beautiful"
    
    print("=== TEXT AUGMENTATION EXAMPLES ===\n")
    print(f"Original: {test_text}\n")
    
    print("Synonym Replacement:")
    for aug in augment_text(test_text, method="sr", num_aug=3):
        print(f"  {aug}")
    
    print("\nRandom Insertion:")
    for aug in augment_text(test_text, method="ri", num_aug=3):
        print(f"  {aug}")
    
    print("\nRandom Swap:")
    for aug in augment_text(test_text, method="rs", num_aug=3):
        print(f"  {aug}")
    
    print("\nRandom Deletion:")
    for aug in augment_text(test_text, method="rd", num_aug=3):
        print(f"  {aug}")
    
    print("\nRandom Method:")
    for aug in augment_text(test_text, method="random", num_aug=3):
        print(f"  {aug}")
