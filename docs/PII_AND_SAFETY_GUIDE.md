# PII Detection & Safety Scoring Guide

**Version**: 1.0  
**Date**: November 2024  
**Miyraa Project**

---

## Table of Contents

1. [Overview](#overview)
2. [PII Detection](#pii-detection)
   - [Features](#pii-features)
   - [Quick Start](#pii-quick-start)
   - [Advanced Usage](#pii-advanced-usage)
3. [Safety Scoring](#safety-scoring)
   - [Features](#safety-features)
   - [Quick Start](#safety-quick-start)
   - [Advanced Usage](#safety-advanced-usage)
4. [Integration](#integration)
5. [Best Practices](#best-practices)
6. [API Reference](#api-reference)

---

## Overview

The Miyraa PII & Safety system provides enterprise-grade content moderation with:

- **Hybrid PII Detection**: Combines regex (fast) with Presidio NER (accurate)
- **Confidence Scoring**: Every entity has a 0.0-1.0 confidence score
- **Reversible Anonymization**: Optionally restore original text
- **Safety Risk Scoring**: Convert binary predictions to 0-1 risk index
- **Adaptive Thresholds**: Context-aware safety filtering

### Architecture

```
Input Text
    │
    ├─→ PII Detection (optional)
    │   ├─→ Regex Patterns (fast)
    │   ├─→ Presidio NER (accurate)
    │   └─→ Merge & Score
    │
    ├─→ Anonymization (optional)
    │   └─→ Replace PII with placeholders
    │
    ├─→ Model Inference
    │   ├─→ Emotion Classification
    │   ├─→ Sentiment Analysis
    │   └─→ Safety Classification
    │
    └─→ Safety Scoring
        ├─→ Risk Index (0-1)
        ├─→ Risk Level (safe/low/medium/high/critical)
        └─→ Explanation
```

---

## PII Detection

### PII Features

✅ **17 Entity Types**:
- `EMAIL` - Email addresses (confidence: 0.95)
- `PHONE` - Phone numbers (0.85)
- `SSN` - Social Security Numbers (0.90)
- `CREDIT_CARD` - Credit card numbers (0.65)
- `IBAN` - International bank account numbers (0.80)
- `URL` - Web URLs (0.95)
- `IP_ADDRESS` - IPv4/IPv6 addresses (0.85)
- `HANDLE` - Social media handles (0.80)
- `PERSON_NAME` - Names (via Presidio)
- `LOCATION` - Addresses (via Presidio)
- And more...

✅ **Detection Methods**:
- **Regex**: Fast pattern matching with high precision
- **Presidio**: NER-based contextual detection (optional)
- **Hybrid**: Combines both, boosts confidence to 0.99

✅ **Anonymization**:
- Hash-based unique placeholders: `[EMAIL_8e621e3d]`
- Reversible mapping for restoration
- Type-aware replacement

### PII Quick Start

```python
from src.nlp.safety.pii_detection import (
    PIIDetector,
    PIIAnonymizer,
    anonymize_for_inference
)

# 1. Basic Detection
detector = PIIDetector()
text = "Contact john.smith@example.com or call 555-123-4567"

entities = detector.detect_pii(text)
for entity in entities:
    print(f"{entity.entity_type.value}: {entity.text}")
    print(f"  Confidence: {entity.confidence:.2f}")
    print(f"  Method: {entity.detection_method}")

# Output:
# EMAIL: john.smith@example.com
#   Confidence: 0.95
#   Method: regex
# PHONE: 555-123-4567
#   Confidence: 0.85
#   Method: regex
```

```python
# 2. Anonymization (Irreversible)
anonymizer = PIIAnonymizer(detector)
anonymized = anonymizer.anonymize(text, reversible=False)
print(anonymized)

# Output:
# Contact [EMAIL] or call [PHONE]
```

```python
# 3. Anonymization (Reversible)
anonymized, reverse_map = anonymizer.anonymize(text, reversible=True)
print(anonymized)
# Output: Contact [EMAIL_8e621e3d] or call [PHONE_4793ec20]

# Restore original
original = anonymizer.deanonymize(anonymized, reverse_map)
print(original)
# Output: Contact john.smith@example.com or call 555-123-4567
```

```python
# 4. Convenience API (for inference)
anonymized, entities = anonymize_for_inference(
    text,
    min_confidence=0.7,
    use_presidio=True
)
print(f"Anonymized: {anonymized}")
print(f"Detected {len(entities)} PII entities")
```

### PII Advanced Usage

#### Detection Statistics

```python
stats = detector.get_detection_stats(text)
print(f"Total entities: {stats['total']}")
print(f"Average confidence: {stats['avg_confidence']:.2f}")

for entity_type, count in stats['by_type'].items():
    print(f"  {entity_type}: {count}")

for method, count in stats['by_method'].items():
    print(f"  {method}: {count}")
```

#### Custom Confidence Threshold

```python
# Only detect high-confidence entities
detector = PIIDetector(min_confidence=0.8)
entities = detector.detect_pii(text)
# Will skip CREDIT_CARD (0.65) and HANDLE (0.80)
```

#### Regex-Only Mode (Fast)

```python
# Skip Presidio for speed
detector = PIIDetector(use_presidio=False)
entities = detector.detect_pii(text)
# 10x faster, but misses contextual entities like names
```

#### Batch Processing

```python
texts = ["Text 1 with john@ex.com", "Text 2 with 555-1234", ...]

for text in texts:
    entities = detector.detect_pii(text)
    if entities:
        anonymized, _ = anonymize_for_inference(text)
        # Process anonymized text...
```

---

## Safety Scoring

### Safety Features

✅ **Multiple Modalities**:
- Binary classification (safe/unsafe)
- Continuous risk index (0.0 = safe, 1.0 = very unsafe)
- Risk levels (safe/low/medium/high/critical)
- Category-specific scores (hate, violence, sexual, harassment)

✅ **Context-Aware**:
- Adaptive thresholds based on user age, platform, content type
- Strict mode for sensitive contexts
- A/B testing support

✅ **Confidence-Weighted**:
- Model confidence included in scoring
- Entropy-based uncertainty measurement
- Calibration support (optional)

### Safety Quick Start

```python
from src.nlp.safety.safety_scoring import SafetyScorer
import torch

# 1. Basic Scoring
scorer = SafetyScorer(threshold=0.5)

# Simulate model output (logits)
safe_logits = torch.tensor([2.0, -1.5])  # [safe, unsafe]
unsafe_logits = torch.tensor([-1.5, 2.0])

safe_score = scorer.score_from_logits(safe_logits)
print(safe_score)

# Output:
# Safety Assessment: ✓ SAFE
#   Risk Index: 0.029 (safe)
#   Confidence: 97.07%
#   Content appears safe (risk: 2.9%)

unsafe_score = scorer.score_from_logits(unsafe_logits)
print(unsafe_score)

# Output:
# Safety Assessment: ✗ UNSAFE
#   Risk Index: 0.971 (critical)
#   Confidence: 97.07%
#   Potential safety issues detected: sexual_content: 93.4%
```

```python
# 2. Access Score Components
print(f"Is safe: {safe_score.is_safe}")
print(f"Risk index: {safe_score.risk_index:.3f}")
print(f"Risk level: {safe_score.risk_level.value}")
print(f"Confidence: {safe_score.confidence:.2%}")
print(f"Explanation: {safe_score.explanation}")
```

```python
# 3. Batch Scoring
batch_logits = torch.tensor([
    [2.0, -1.5],   # Safe
    [-1.5, 2.0],   # Unsafe
    [0.5, 0.3],    # Borderline
])

batch_scores = scorer.batch_score(batch_logits)
for i, score in enumerate(batch_scores):
    status = "✓" if score.is_safe else "✗"
    print(f"Text {i+1}: {status} | Risk: {score.risk_index:.3f}")
```

### Safety Advanced Usage

#### Adaptive Thresholds

```python
from src.nlp.safety.safety_scoring import AdaptiveSafetyScorer

scorer = AdaptiveSafetyScorer(base_threshold=0.5)

# Context 1: Adult on private platform
score1 = scorer.score_with_context(
    logits,
    context={"user_age": 25, "platform": "private"}
)

# Context 2: Minor on public platform (stricter)
score2 = scorer.score_with_context(
    logits,
    context={"user_age": 15, "platform": "public"}
)

# Context 3: Educational with moderation (more lenient)
score3 = scorer.score_with_context(
    logits,
    context={
        "content_type": "educational",
        "has_moderation": True
    }
)
```

#### Content Filtering

```python
texts = ["Safe text", "Unsafe text", "Borderline text", ...]
scores = scorer.batch_score(batch_logits)

# Filter out unsafe content
safe_texts, safe_indices = scorer.filter_unsafe(
    texts,
    scores,
    return_indices=True
)

print(f"Filtered {len(safe_texts)}/{len(texts)} safe texts")
```

#### Batch Statistics

```python
stats = scorer.get_statistics(batch_scores)

print(f"Total texts: {stats['total_texts']}")
print(f"Safe: {stats['safe_count']} ({stats['safe_percentage']:.1f}%)")
print(f"Average risk: {stats['avg_risk_index']:.3f}")

for level, count in stats['risk_level_distribution'].items():
    print(f"  {level}: {count}")
```

#### Custom Risk Index Modes

```python
# Mode 1: Pure probability (default)
scorer1 = SafetyScorer(risk_index_mode="probability")

# Mode 2: Entropy-weighted (accounts for uncertainty)
scorer2 = SafetyScorer(risk_index_mode="entropy")

# Mode 3: Calibrated (requires calibration step)
scorer3 = SafetyScorer(risk_index_mode="calibrated")
```

---

## Integration

### Inference Pipeline

```python
from src.nlp.inference.dummy_engine import get_engine

# Create engine with PII anonymization + safety scoring
engine = get_engine(
    anonymize_pii=True,
    pii_min_confidence=0.7,
    safety_threshold=0.5
)

# Make prediction
text = "Contact john@example.com about urgent matter!"
result = engine.predict(text)

print(f"Original: {result['text']}")
print(f"Processed: {result['processed_text']}")

if result.get('pii_info'):
    print(f"PII detected: {result['pii_info']['pii_count']} entities")
    print(f"Types: {result['pii_info']['pii_types']}")

if result['safety'].get('score'):
    score = result['safety']['score']
    print(f"Safety risk: {score['risk_index']:.3f} ({score['risk_level']})")
    print(f"Explanation: {score['explanation']}")
```

### API Server Integration

```python
# In src/api/main.py or similar

from src.nlp.safety.pii_detection import anonymize_for_inference
from src.nlp.safety.safety_scoring import SafetyScorer

scorer = SafetyScorer(threshold=0.5)

@app.post("/predict")
async def predict(request: PredictRequest):
    text = request.text
    
    # Step 1: Optional PII anonymization
    if request.anonymize_pii:
        text, pii_entities = anonymize_for_inference(text)
    
    # Step 2: Model inference
    outputs = model.predict(text)
    
    # Step 3: Safety scoring
    safety_score = scorer.score_from_logits(outputs['safety_logits'])
    
    # Step 4: Block if unsafe
    if not safety_score.is_safe:
        return {
            "error": "Content blocked for safety",
            "risk_index": safety_score.risk_index,
            "explanation": safety_score.explanation
        }
    
    return {
        "emotions": outputs['emotions'],
        "sentiment": outputs['sentiment'],
        "safety": safety_score.to_dict()
    }
```

---

## Best Practices

### PII Detection

✅ **DO**:
- Use hybrid mode (regex + Presidio) for best accuracy
- Set appropriate confidence thresholds (0.7-0.8 recommended)
- Cache Presidio engines for performance
- Test with real-world examples from your domain
- Document what PII types you detect and why

❌ **DON'T**:
- Skip PII detection for user-generated content
- Use irreversible anonymization if you need to display original text
- Assume regex catches everything (use Presidio for names, locations)
- Set confidence too low (0.5) - increases false positives

### Safety Scoring

✅ **DO**:
- Use adaptive thresholds for different contexts
- Monitor false positive/negative rates
- Provide clear explanations for blocked content
- Log risk scores for auditing
- Combine with human moderation for edge cases

❌ **DON'T**:
- Use single threshold for all contexts
- Block content without explanation
- Ignore model confidence scores
- Skip calibration evaluation
- Deploy without testing on diverse examples

### Performance

✅ **Optimization Tips**:
- Use regex-only mode for latency-critical paths (10x faster)
- Batch PII detection for multiple texts
- Cache Presidio analyzers (lazy-loaded by default)
- Pre-filter obviously safe content
- Use async processing for non-blocking inference

---

## API Reference

### PII Detection

#### `PIIDetector`

```python
PIIDetector(
    use_presidio: bool = True,
    min_confidence: float = 0.5,
    language: str = "en"
)
```

**Methods**:
- `detect_pii(text: str) -> List[PIIEntity]`
- `get_detection_stats(text: str) -> Dict`

#### `PIIAnonymizer`

```python
PIIAnonymizer(
    detector: PIIDetector,
    include_hash: bool = True
)
```

**Methods**:
- `anonymize(text: str, reversible: bool = False) -> Union[str, Tuple[str, Dict]]`
- `deanonymize(text: str, reverse_map: Dict) -> str`

#### `PIIEntity`

```python
@dataclass
class PIIEntity:
    entity_type: PIIEntityType
    start: int
    end: int
    text: str
    confidence: float
    detection_method: str
    text_hash: Optional[str]
```

### Safety Scoring

#### `SafetyScorer`

```python
SafetyScorer(
    threshold: float = 0.5,
    risk_index_mode: str = "probability",
    category_names: Optional[List[str]] = None
)
```

**Methods**:
- `score_from_logits(logits: Tensor, category_logits: Optional[Tensor] = None) -> SafetyScore`
- `score_from_probabilities(probabilities: np.ndarray) -> SafetyScore`
- `batch_score(logits: Tensor, category_logits: Optional[Tensor] = None) -> List[SafetyScore]`
- `filter_unsafe(texts: List[str], scores: List[SafetyScore], return_indices: bool = False) -> List[str]`
- `get_statistics(scores: List[SafetyScore]) -> Dict`

#### `AdaptiveSafetyScorer`

```python
AdaptiveSafetyScorer(
    base_threshold: float = 0.5,
    strict_mode: bool = False
)
```

**Methods**:
- `score_with_context(logits: Tensor, context: Dict, category_logits: Optional[Tensor] = None) -> SafetyScore`

#### `SafetyScore`

```python
@dataclass
class SafetyScore:
    is_safe: bool
    risk_index: float
    risk_level: RiskLevel
    confidence: float
    categories: Dict[str, float]
    explanation: str
```

**Methods**:
- `to_dict() -> Dict`
- `__str__() -> str`

---

## Examples

### Example 1: Full Pipeline

```python
from src.nlp.safety.pii_detection import PIIDetector, PIIAnonymizer
from src.nlp.safety.safety_scoring import SafetyScorer
import torch

# Setup
pii_detector = PIIDetector(use_presidio=True, min_confidence=0.7)
pii_anonymizer = PIIAnonymizer(pii_detector)
safety_scorer = SafetyScorer(threshold=0.5)

# Input
text = "Email john.smith@example.com about inappropriate content"

# Step 1: Detect PII
entities = pii_detector.detect_pii(text)
print(f"Found {len(entities)} PII entities")

# Step 2: Anonymize
anonymized, reverse_map = pii_anonymizer.anonymize(text, reversible=True)
print(f"Anonymized: {anonymized}")

# Step 3: Model inference (simulated)
safety_logits = torch.tensor([-1.0, 1.5])  # [safe, unsafe]

# Step 4: Safety scoring
safety_score = safety_scorer.score_from_logits(safety_logits)
print(safety_score)

# Step 5: Decision
if safety_score.is_safe:
    # Restore original text
    original = pii_anonymizer.deanonymize(anonymized, reverse_map)
    print(f"Safe content: {original}")
else:
    print(f"Blocked: {safety_score.explanation}")
```

### Example 2: Batch Processing

```python
texts = [
    "Contact support@company.com for help",
    "This is hateful content",
    "Normal safe message",
    "Call 555-1234 immediately"
]

# PII detection
for text in texts:
    entities = pii_detector.detect_pii(text)
    if entities:
        anonymized, _ = pii_anonymizer.anonymize(text)
        print(f"Anonymized: {anonymized}")

# Safety scoring
batch_logits = torch.tensor([
    [2.0, -1.0],   # Safe
    [-2.0, 2.5],   # Unsafe
    [1.5, -0.5],   # Safe
    [1.0, -0.8]    # Safe
])

scores = safety_scorer.batch_score(batch_logits)
stats = safety_scorer.get_statistics(scores)

print(f"Safe: {stats['safe_count']}/{stats['total_texts']}")
print(f"Average risk: {stats['avg_risk_index']:.3f}")
```

---

## Troubleshooting

### Presidio Not Available

If you see:
```
⚠️  Note: Presidio not available
```

**Solution**:
```bash
pip install presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_sm
```

### Low PII Detection Accuracy

**Symptoms**: Missing entities or too many false positives

**Solutions**:
- Adjust `min_confidence` threshold (try 0.7-0.8)
- Enable Presidio for contextual detection
- Add custom regex patterns for domain-specific PII
- Test with representative examples

### High False Positive Rate (Safety)

**Symptoms**: Blocking too much safe content

**Solutions**:
- Increase safety threshold (try 0.6-0.7)
- Use adaptive scoring with context
- Check calibration (see `docs/EVALUATION_GUIDE.md`)
- Review edge cases and adjust category weights

### Performance Issues

**Symptoms**: Slow inference

**Solutions**:
- Use regex-only mode: `PIIDetector(use_presidio=False)`
- Batch process multiple texts together
- Cache Presidio engines (done automatically)
- Pre-filter obviously safe content

---

## References

- [Presidio Documentation](https://microsoft.github.io/presidio/)
- [spaCy NER Guide](https://spacy.io/usage/linguistic-features#named-entities)
- [Evaluation Guide](./EVALUATION_GUIDE.md)
- [Model Training Guide](./TRAINING_GUIDE.md)

---

**Last Updated**: November 2024  
**Version**: 1.0  
**Contact**: Miyraa Team
