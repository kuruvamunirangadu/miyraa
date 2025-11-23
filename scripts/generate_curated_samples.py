"""Generate curated emotion samples using the Miyraa taxonomy.

Creates high-quality, hand-crafted samples covering:
- All 11 core emotions
- Balanced VAD distribution
- All 4 safety categories
- All 5 style categories
- All 6 intent categories
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp.data.taxonomy import (
    EMOTIONS, EMOTION_VAD_MAPPING, SAFETY_CATEGORIES,
    STYLE_CATEGORIES, INTENT_CATEGORIES
)


# =============================================================================
# CURATED EMOTION SAMPLES
# =============================================================================

CURATED_EMOTION_SAMPLES = {
    "joy": [
        "I just got accepted into my dream university! This is the best day ever!",
        "Spending time with my family during the holidays fills me with pure happiness.",
        "Finally finished that difficult project at work. Feeling accomplished and proud!",
        "The concert was absolutely amazing! Dancing all night was so much fun.",
        "My best friend surprised me with tickets to see my favorite band. So excited!",
    ],
    
    "love": [
        "I love my partner more than words can express. They complete me.",
        "Watching my child take their first steps melted my heart completely.",
        "My grandmother's cooking always reminds me of her endless love and care.",
        "Thank you for always being there for me. I truly appreciate our friendship.",
        "Adopting our rescue dog was the best decision. He brings so much joy to our lives.",
    ],
    
    "surprise": [
        "I can't believe they threw me a surprise birthday party! I had no idea!",
        "Wait, you're getting married? That's such unexpected news!",
        "Wow, I never expected to run into you here after all these years.",
        "They announced the winner and I was shocked to hear my name called.",
        "I opened the box and found something I never thought I'd see again.",
    ],
    
    "sadness": [
        "I miss my grandfather every single day since he passed away.",
        "Breaking up after five years together has left me feeling empty inside.",
        "Moving to a new city means leaving behind all my childhood friends.",
        "The doctor's diagnosis wasn't what we were hoping to hear.",
        "Watching my favorite show end after so many seasons makes me emotional.",
    ],
    
    "anger": [
        "They lied to my face and I'm absolutely furious about it!",
        "This is the third time my order has been wrong. I'm seriously frustrated.",
        "How dare they take credit for my work! This is completely unacceptable.",
        "The traffic is unbearable today and I'm already late for an important meeting.",
        "I can't believe they canceled the event without any notice. I'm livid!",
    ],
    
    "fear": [
        "Walking alone at night in this neighborhood makes me really nervous.",
        "I have a big presentation tomorrow and I'm terrified of public speaking.",
        "The horror movie was so scary I couldn't sleep all night.",
        "I'm worried about my test results. What if it's bad news?",
        "Flying in this turbulence has me gripping the armrests in panic.",
    ],
    
    "disgust": [
        "The food had mold on it. Absolutely revolting!",
        "I can't stand people who are rude to service workers.",
        "The bathroom was filthy. I couldn't even go inside.",
        "Hearing about animal abuse makes me sick to my stomach.",
        "The smell coming from the dumpster is nauseating.",
    ],
    
    "calm": [
        "Sitting by the lake watching the sunset brings me such peace.",
        "Meditation in the morning helps me start my day feeling centered.",
        "The sound of rain on the window is so soothing and relaxing.",
        "Taking a walk in the park clears my mind and calms my nerves.",
        "Everything is going to work out. I'm feeling at peace with my decision.",
    ],
    
    "excitement": [
        "Only two more days until vacation! I can barely contain my enthusiasm!",
        "We're launching our startup next week. This is going to be epic!",
        "I'm so pumped for the championship game tonight! Go team!",
        "Just bought tickets for the world tour! Can't wait to travel!",
        "The new season drops tomorrow and I'm counting down the hours!",
    ],
    
    "confusion": [
        "Wait, I thought the meeting was at 3pm? Now I'm completely lost.",
        "These instructions don't make any sense to me. Can someone explain?",
        "I'm not sure I understand what you're trying to say here.",
        "This math problem has me stumped. I don't know where to start.",
        "How is that even possible? I'm so confused right now.",
    ],
    
    "neutral": [
        "I'm going to the store to buy groceries.",
        "The meeting is scheduled for tomorrow at 10am.",
        "Please send me the report when you have a chance.",
        "The weather forecast shows partly cloudy skies today.",
        "I'll take the train to work this morning.",
    ],
}


# =============================================================================
# CURATED SAFETY SAMPLES
# =============================================================================

CURATED_SAFETY_SAMPLES = {
    "toxic": [
        "You're an idiot and nobody likes you.",
        "This is the dumbest thing I've ever heard in my life.",
        "People like you are what's wrong with society.",
        "Get lost, loser. Nobody wants you here.",
        "You're trash and everyone knows it.",
    ],
    
    "profane": [
        "What the hell were you thinking?",
        "This is complete bullshit and you know it.",
        "I don't give a damn about your opinion.",
        "Screw this, I'm out of here.",
        "That's a bunch of crap and makes no sense.",
    ],
    
    "threatening": [
        "You better watch your back from now on.",
        "I know where you live and I'm coming for you.",
        "You're going to regret crossing me.",
        "Keep talking and see what happens to you.",
        "I'll make sure you pay for this mistake.",
    ],
    
    "harassment": [
        "Hey baby, why don't you smile for me?",
        "I've been watching you for weeks. You're so beautiful.",
        "Come on, just give me your number. I won't stop asking.",
        "You should dress more appropriately for work.",
        "Why are you ignoring my messages? I just want to talk to you.",
    ],
}


# =============================================================================
# CURATED STYLE SAMPLES
# =============================================================================

CURATED_STYLE_SAMPLES = {
    "formal": [
        "Dear Sir or Madam, I am writing to inquire about the position advertised on your website.",
        "Pursuant to our conversation, I would like to formally request an extension.",
        "It is imperative that we address this matter at your earliest convenience.",
        "I hereby submit my resignation, effective two weeks from today's date.",
        "We cordially invite you to attend our annual shareholders meeting.",
    ],
    
    "casual": [
        "Hey, what's up? Wanna grab lunch sometime this week?",
        "Just finished that book you recommended. It was pretty good!",
        "Dude, you have to check out this new coffee place downtown.",
        "Thanks for the help earlier. Really appreciate it!",
        "No worries, I totally understand. Let's reschedule for next week.",
    ],
    
    "assertive": [
        "I need this report completed by end of day today, no exceptions.",
        "Let me be clear: this behavior is unacceptable and must stop immediately.",
        "I've made my decision and I'm not changing my mind.",
        "We will proceed with the original plan as discussed.",
        "I expect a response within 24 hours. This is time-sensitive.",
    ],
    
    "empathetic": [
        "I can imagine how difficult this must be for you right now.",
        "I'm so sorry you're going through this. Please know I'm here for you.",
        "It sounds like you're feeling overwhelmed. That's completely understandable.",
        "I hear what you're saying and I can see why you'd feel that way.",
        "Thank you for sharing that with me. It takes courage to be vulnerable.",
    ],
    
    "humorous": [
        "I'm not saying I'm Batman, but have you ever seen me and Batman in the same room?",
        "My cooking is so good, even the smoke alarm cheers me on.",
        "I told my wife she was drawing her eyebrows too high. She looked surprised.",
        "I'm on a seafood diet. I see food and I eat it!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
    ],
}


# =============================================================================
# CURATED INTENT SAMPLES
# =============================================================================

CURATED_INTENT_SAMPLES = {
    "statement": [
        "The Earth orbits around the Sun once every 365 days.",
        "I graduated from college in 2020 with a degree in computer science.",
        "The restaurant opens at 11am on weekdays.",
        "Climate change is affecting ecosystems around the world.",
        "My brother works as a software engineer in San Francisco.",
    ],
    
    "question": [
        "What time does the train arrive at the station?",
        "How do I reset my password for this account?",
        "Where did you buy that jacket? It looks great!",
        "Can you explain how this process works?",
        "Why did they cancel the event at the last minute?",
    ],
    
    "request": [
        "Could you please send me the updated schedule?",
        "Would you mind helping me move this weekend?",
        "Can I get a glass of water, please?",
        "Please review the attached document and provide your feedback.",
        "I'd appreciate it if you could arrive a bit earlier tomorrow.",
    ],
    
    "command": [
        "Close the door behind you when you leave.",
        "Submit your assignment by Friday at midnight.",
        "Turn left at the next intersection.",
        "Stop talking and listen to the instructions.",
        "Save your work before shutting down the computer.",
    ],
    
    "expression": [
        "Wow, that's incredible!",
        "Oh no, I can't believe that happened!",
        "Ugh, this traffic is killing me.",
        "Yay! We finally finished the project!",
        "Hmm, that's an interesting perspective.",
    ],
    
    "social": [
        "Good morning! Hope you have a wonderful day!",
        "Congratulations on your promotion! You deserve it!",
        "I'm so sorry for your loss. My thoughts are with you.",
        "Happy birthday! Wishing you all the best!",
        "Thank you so much for the thoughtful gift!",
    ],
}


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate_curated_dataset() -> List[Dict]:
    """Generate complete curated dataset with all samples.
    
    Returns:
        List of sample dictionaries with text, emotion, vad, safety, style, intent
    """
    dataset = []
    
    # Add emotion samples
    for emotion, texts in CURATED_EMOTION_SAMPLES.items():
        vad = EMOTION_VAD_MAPPING[emotion]
        for text in texts:
            sample = {
                "text": text,
                "emotion": emotion,
                "valence": vad["valence"],
                "arousal": vad["arousal"],
                "dominance": vad["dominance"],
                "safety": "safe",
                "style": random.choice(["casual", "formal"]),
                "intent": random.choice(["statement", "expression"]),
            }
            dataset.append(sample)
    
    # Add safety samples (mark as unsafe)
    for safety_cat, texts in CURATED_SAFETY_SAMPLES.items():
        for text in texts:
            sample = {
                "text": text,
                "emotion": "anger",  # Most unsafe content is angry
                "valence": -0.6,
                "arousal": 0.7,
                "dominance": 0.5,
                "safety": safety_cat,
                "style": "assertive",
                "intent": "statement",
            }
            dataset.append(sample)
    
    # Add style samples
    for style, texts in CURATED_STYLE_SAMPLES.items():
        for text in texts:
            # Infer emotion from style
            emotion_map = {
                "formal": "neutral",
                "casual": "neutral",
                "assertive": "neutral",
                "empathetic": "love",
                "humorous": "joy",
            }
            emotion = emotion_map.get(style, "neutral")
            vad = EMOTION_VAD_MAPPING[emotion]
            
            sample = {
                "text": text,
                "emotion": emotion,
                "valence": vad["valence"],
                "arousal": vad["arousal"],
                "dominance": vad["dominance"],
                "safety": "safe",
                "style": style,
                "intent": "statement",
            }
            dataset.append(sample)
    
    # Add intent samples
    for intent, texts in CURATED_INTENT_SAMPLES.items():
        for text in texts:
            sample = {
                "text": text,
                "emotion": "neutral",
                "valence": 0.0,
                "arousal": 0.0,
                "dominance": 0.0,
                "safety": "safe",
                "style": "neutral",
                "intent": intent,
            }
            dataset.append(sample)
    
    return dataset


def save_curated_dataset(output_path: str):
    """Generate and save curated dataset to JSONL file.
    
    Args:
        output_path: Path to save dataset
    """
    dataset = generate_curated_dataset()
    
    # Shuffle for better distribution
    random.shuffle(dataset)
    
    # Save as JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    
    print(f"✅ Saved {len(dataset)} curated samples to {output_path}")
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    emotions = {}
    safety = {}
    styles = {}
    intents = {}
    
    for sample in dataset:
        emotions[sample['emotion']] = emotions.get(sample['emotion'], 0) + 1
        safety[sample['safety']] = safety.get(sample['safety'], 0) + 1
        styles[sample['style']] = styles.get(sample['style'], 0) + 1
        intents[sample['intent']] = intents.get(sample['intent'], 0) + 1
    
    print(f"\nEmotions: {dict(sorted(emotions.items()))}")
    print(f"Safety: {dict(sorted(safety.items()))}")
    print(f"Styles: {dict(sorted(styles.items()))}")
    print(f"Intents: {dict(sorted(intents.items()))}")


if __name__ == "__main__":
    output_path = "data/processed/curated/samples.jsonl"
    save_curated_dataset(output_path)
    
    print("\n✅ Curated dataset generation complete!")
    print(f"📁 Output: {output_path}")
