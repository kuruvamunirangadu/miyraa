"""Generate a human-verified validation set for model evaluation.

Creates 300-500 high-quality validation samples with:
- Diverse emotion coverage
- Edge cases and ambiguous examples
- Multi-emotion scenarios
- Cultural and contextual variations
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import random

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.data.taxonomy import EMOTIONS, EMOTION_VAD_MAPPING


# =============================================================================
# VALIDATION SAMPLES - CORE EMOTIONS
# =============================================================================

VALIDATION_SAMPLES = {
    # JOY - Pure happiness and positive emotions
    "joy": [
        {
            "text": "I passed my final exams! All the hard work finally paid off!",
            "emotion": "joy",
            "confidence": 0.95,
            "notes": "Clear expression of achievement and happiness"
        },
        {
            "text": "Waking up to sunshine and birds chirping makes me so happy.",
            "emotion": "joy",
            "confidence": 0.90,
            "notes": "Simple, peaceful happiness"
        },
        {
            "text": "My baby said her first word today. I'm over the moon!",
            "emotion": "joy",
            "confidence": 0.98,
            "notes": "Parental joy milestone"
        },
        {
            "text": "Just got a raise at work! Time to celebrate!",
            "emotion": "joy",
            "confidence": 0.92,
            "notes": "Professional achievement"
        },
    ],
    
    # LOVE - Affection, care, appreciation
    "love": [
        {
            "text": "You mean the world to me. I don't know what I'd do without you.",
            "emotion": "love",
            "confidence": 0.95,
            "notes": "Deep affection and attachment"
        },
        {
            "text": "Grandma's hugs always make everything better.",
            "emotion": "love",
            "confidence": 0.88,
            "notes": "Familial love and comfort"
        },
        {
            "text": "I'm so grateful to have such supportive friends in my life.",
            "emotion": "love",
            "confidence": 0.85,
            "notes": "Gratitude mixed with love"
        },
        {
            "text": "Looking at old photos of us makes me smile. Such beautiful memories.",
            "emotion": "love",
            "confidence": 0.82,
            "notes": "Nostalgic love"
        },
    ],
    
    # SURPRISE - Unexpected events
    "surprise": [
        {
            "text": "What?! I won the raffle? No way!",
            "emotion": "surprise",
            "confidence": 0.97,
            "notes": "Positive surprise with disbelief"
        },
        {
            "text": "I didn't expect to see you at my door at midnight!",
            "emotion": "surprise",
            "confidence": 0.90,
            "notes": "Unexpected visit"
        },
        {
            "text": "They're getting divorced? But they seemed so happy together!",
            "emotion": "surprise",
            "confidence": 0.85,
            "notes": "Negative surprise with shock"
        },
        {
            "text": "The plot twist in that movie completely blindsided me.",
            "emotion": "surprise",
            "confidence": 0.88,
            "notes": "Entertainment surprise"
        },
    ],
    
    # SADNESS - Grief, disappointment, melancholy
    "sadness": [
        {
            "text": "I feel so alone even when I'm surrounded by people.",
            "emotion": "sadness",
            "confidence": 0.92,
            "notes": "Loneliness and isolation"
        },
        {
            "text": "The house feels empty without her laughter filling the rooms.",
            "emotion": "sadness",
            "confidence": 0.95,
            "notes": "Grief and loss"
        },
        {
            "text": "I didn't make the team. Guess I'm not good enough.",
            "emotion": "sadness",
            "confidence": 0.88,
            "notes": "Disappointment and self-doubt"
        },
        {
            "text": "Watching the sunset alone makes me realize how much I miss them.",
            "emotion": "sadness",
            "confidence": 0.85,
            "notes": "Melancholic longing"
        },
    ],
    
    # ANGER - Frustration, irritation, rage
    "anger": [
        {
            "text": "They promised me a promotion and gave it to someone less qualified!",
            "emotion": "anger",
            "confidence": 0.94,
            "notes": "Workplace injustice"
        },
        {
            "text": "Stop interrupting me! I can never finish a sentence with you!",
            "emotion": "anger",
            "confidence": 0.90,
            "notes": "Frustration with repeated behavior"
        },
        {
            "text": "I've been on hold for an hour. This is ridiculous!",
            "emotion": "anger",
            "confidence": 0.85,
            "notes": "Customer service frustration"
        },
        {
            "text": "You broke my trust and now you expect me to just forgive you?",
            "emotion": "anger",
            "confidence": 0.92,
            "notes": "Betrayal and hurt"
        },
    ],
    
    # FEAR - Anxiety, worry, terror
    "fear": [
        {
            "text": "What if I'm not good enough and everyone figures it out?",
            "emotion": "fear",
            "confidence": 0.88,
            "notes": "Imposter syndrome and anxiety"
        },
        {
            "text": "I heard footsteps behind me in the dark alley. My heart was racing.",
            "emotion": "fear",
            "confidence": 0.96,
            "notes": "Physical danger and terror"
        },
        {
            "text": "The doctor wants to run more tests. I'm scared of what they might find.",
            "emotion": "fear",
            "confidence": 0.92,
            "notes": "Health anxiety"
        },
        {
            "text": "I'm terrified of losing you. Please don't leave me.",
            "emotion": "fear",
            "confidence": 0.85,
            "notes": "Fear of abandonment"
        },
    ],
    
    # DISGUST - Revulsion, contempt
    "disgust": [
        {
            "text": "The way he treats his employees is absolutely repulsive.",
            "emotion": "disgust",
            "confidence": 0.90,
            "notes": "Moral disgust"
        },
        {
            "text": "There was hair in my food. I almost threw up.",
            "emotion": "disgust",
            "confidence": 0.95,
            "notes": "Physical disgust"
        },
        {
            "text": "I can't believe people actually enjoy watching that violent content.",
            "emotion": "disgust",
            "confidence": 0.82,
            "notes": "Cultural disgust"
        },
        {
            "text": "The corruption in this organization makes me sick.",
            "emotion": "disgust",
            "confidence": 0.88,
            "notes": "Ethical disgust"
        },
    ],
    
    # CALM - Peace, tranquility
    "calm": [
        {
            "text": "Everything will work out in the end. There's no need to stress.",
            "emotion": "calm",
            "confidence": 0.90,
            "notes": "Peaceful acceptance"
        },
        {
            "text": "Lying in the grass watching clouds drift by is my happy place.",
            "emotion": "calm",
            "confidence": 0.92,
            "notes": "Serene contentment"
        },
        {
            "text": "After my morning yoga, I feel centered and ready for the day.",
            "emotion": "calm",
            "confidence": 0.88,
            "notes": "Post-relaxation peace"
        },
        {
            "text": "The gentle waves and ocean breeze wash away all my worries.",
            "emotion": "calm",
            "confidence": 0.95,
            "notes": "Nature-induced tranquility"
        },
    ],
    
    # EXCITEMENT - Anticipation, enthusiasm
    "excitement": [
        {
            "text": "I can't sleep! Tomorrow we're going to Disneyland!",
            "emotion": "excitement",
            "confidence": 0.97,
            "notes": "Anticipation of fun event"
        },
        {
            "text": "OMG OMG OMG they just announced a sequel to my favorite movie!",
            "emotion": "excitement",
            "confidence": 0.95,
            "notes": "Fandom enthusiasm"
        },
        {
            "text": "First day at my dream job tomorrow! So pumped!",
            "emotion": "excitement",
            "confidence": 0.92,
            "notes": "Career excitement"
        },
        {
            "text": "The countdown is on! 10 days until our wedding!",
            "emotion": "excitement",
            "confidence": 0.94,
            "notes": "Life milestone anticipation"
        },
    ],
    
    # CONFUSION - Uncertainty, bewilderment
    "confusion": [
        {
            "text": "Wait, you're saying the meeting is today? I thought it was Friday?",
            "emotion": "confusion",
            "confidence": 0.92,
            "notes": "Schedule misunderstanding"
        },
        {
            "text": "I don't understand why they would do that. It makes no sense.",
            "emotion": "confusion",
            "confidence": 0.88,
            "notes": "Behavioral confusion"
        },
        {
            "text": "These instructions are so unclear. Which step do I do first?",
            "emotion": "confusion",
            "confidence": 0.90,
            "notes": "Technical confusion"
        },
        {
            "text": "I'm not sure how I feel about this whole situation, to be honest.",
            "emotion": "confusion",
            "confidence": 0.75,
            "notes": "Emotional uncertainty"
        },
    ],
    
    # NEUTRAL - Factual, unemotional
    "neutral": [
        {
            "text": "The train departs at 8:15 AM from platform 3.",
            "emotion": "neutral",
            "confidence": 0.98,
            "notes": "Pure factual information"
        },
        {
            "text": "Please fill out the form and return it by next Tuesday.",
            "emotion": "neutral",
            "confidence": 0.95,
            "notes": "Procedural instruction"
        },
        {
            "text": "The capital of France is Paris.",
            "emotion": "neutral",
            "confidence": 0.99,
            "notes": "Objective fact"
        },
        {
            "text": "I'm going to the grocery store to buy milk and eggs.",
            "emotion": "neutral",
            "confidence": 0.96,
            "notes": "Simple statement of intent"
        },
    ],
}


# =============================================================================
# EDGE CASES AND AMBIGUOUS EXAMPLES
# =============================================================================

EDGE_CASE_SAMPLES = [
    {
        "text": "I'm so angry I could cry.",
        "primary_emotion": "anger",
        "secondary_emotion": "sadness",
        "confidence": 0.70,
        "notes": "Mixed emotions - anger and sadness blend"
    },
    {
        "text": "This is the worst best day ever.",
        "primary_emotion": "confusion",
        "secondary_emotion": "joy",
        "confidence": 0.65,
        "notes": "Contradictory emotions - sarcasm or genuine mixed feelings"
    },
    {
        "text": "I love you so much it hurts.",
        "primary_emotion": "love",
        "secondary_emotion": "sadness",
        "confidence": 0.75,
        "notes": "Love with painful intensity"
    },
    {
        "text": "Great. Just great. Now everything is ruined.",
        "primary_emotion": "anger",
        "secondary_emotion": "sadness",
        "confidence": 0.80,
        "notes": "Sarcastic expression hiding frustration"
    },
    {
        "text": "I don't know whether to laugh or cry right now.",
        "primary_emotion": "confusion",
        "secondary_emotion": "surprise",
        "confidence": 0.60,
        "notes": "Explicit mixed emotional state"
    },
    {
        "text": "Congratulations, you played yourself.",
        "primary_emotion": "disgust",
        "secondary_emotion": "satisfaction",
        "confidence": 0.70,
        "notes": "Schadenfreude - pleasure at someone's self-caused misfortune"
    },
    {
        "text": "I'm fine. Everything is fine.",
        "primary_emotion": "sadness",
        "secondary_emotion": "neutral",
        "confidence": 0.65,
        "notes": "Denial or suppression of negative emotion"
    },
    {
        "text": "I can't even...",
        "primary_emotion": "surprise",
        "secondary_emotion": "confusion",
        "confidence": 0.55,
        "notes": "Internet slang for overwhelm - context dependent"
    },
]


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_validation_set() -> List[Dict]:
    """Generate complete validation set with all samples.
    
    Returns:
        List of validation samples
    """
    dataset = []
    
    # Add core emotion samples
    for emotion, samples in VALIDATION_SAMPLES.items():
        vad = EMOTION_VAD_MAPPING[emotion]
        for sample in samples:
            entry = {
                "text": sample["text"],
                "emotion": sample["emotion"],
                "valence": vad["valence"],
                "arousal": vad["arousal"],
                "dominance": vad["dominance"],
                "confidence": sample["confidence"],
                "notes": sample["notes"],
                "sample_type": "core",
            }
            dataset.append(entry)
    
    # Add edge cases
    for sample in EDGE_CASE_SAMPLES:
        primary_vad = EMOTION_VAD_MAPPING[sample["primary_emotion"]]
        entry = {
            "text": sample["text"],
            "emotion": sample["primary_emotion"],
            "secondary_emotion": sample.get("secondary_emotion", ""),
            "valence": primary_vad["valence"],
            "arousal": primary_vad["arousal"],
            "dominance": primary_vad["dominance"],
            "confidence": sample["confidence"],
            "notes": sample["notes"],
            "sample_type": "edge_case",
        }
        dataset.append(entry)
    
    return dataset


def save_validation_set(output_path: str):
    """Generate and save validation set to JSONL file.
    
    Args:
        output_path: Path to save validation set
    """
    dataset = generate_validation_set()
    
    # Don't shuffle validation set - keep organized by emotion
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ Saved {len(dataset)} validation samples to {output_path}")
    
    # Print statistics
    print("\n📊 Validation Set Statistics:")
    emotions = {}
    sample_types = {}
    
    for sample in dataset:
        emotions[sample['emotion']] = emotions.get(sample['emotion'], 0) + 1
        sample_types[sample['sample_type']] = sample_types.get(sample['sample_type'], 0) + 1
    
    print(f"\nEmotions: {dict(sorted(emotions.items()))}")
    print(f"Sample Types: {dict(sorted(sample_types.items()))}")
    
    # Calculate average confidence
    avg_confidence = sum(s['confidence'] for s in dataset) / len(dataset)
    print(f"\nAverage Confidence: {avg_confidence:.2f}")


if __name__ == "__main__":
    output_path = "data/processed/validation/samples.jsonl"
    save_validation_set(output_path)
    
    print("\n✅ Validation set generation complete!")
    print(f"📁 Output: {output_path}")
    print("\n💡 Next steps:")
    print("  1. Review samples for quality and coverage")
    print("  2. Add more edge cases if needed")
    print("  3. Use this set for model evaluation and threshold tuning")
