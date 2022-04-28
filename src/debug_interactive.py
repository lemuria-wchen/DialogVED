from pytorch_transformers import BertTokenizer

tok = BertTokenizer.from_pretrained('bert-base-uncased')

# context = 'brother, I\'m a bit annoying'
context = 'You are really not kind'
print(' '.join(tok.tokenize(context)))
