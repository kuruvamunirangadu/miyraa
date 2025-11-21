from transformers import AutoTokenizer, AutoModel

print('downloading tokenizer...')
AutoTokenizer.from_pretrained('xlm-roberta-base')
print('downloading model...')
AutoModel.from_pretrained('xlm-roberta-base')
print('done')
