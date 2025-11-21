"""Quick smoke test for label mapping."""
from src.nlp.preprocessing.label_mapping import map_goemotions_to_target, TARGET_TAXONOMY


def run():
    samples = [
        (['joy', 'surprise'], 'What a lovely surprise!'),
        ([], 'I remember when we used to travel every summer.'),
        (['pride'], 'I am so proud of you'),
    ]
    for src, text in samples:
        mapped = map_goemotions_to_target(src, text)
        print(f"src={src} text={text!r} -> mapped={mapped}")


if __name__ == '__main__':
    run()
