import runway
from runway.data_types import array, text, vector, category
from sentenceencoder import UniSentenceEncXling
from nltk import sent_tokenize

@runway.setup
def setup():
    print('[SETUP] loading model...')
    model = UniSentenceEncXling()
    print('[SETUP] done.')
    return model

desc = """\
Infers embeddings for input. Returns an array with the lines of
text and an array with the corresponding embeddings for each line.
"""
@runway.command(name='embed',
        inputs={
            'text': text(),
            'tokenize_sentences': category(
                choices=['yes', 'no'], default='yes')
        },
        outputs={
            'sentences': array(item_type=text),
            'embeddings': array(item_type=vector(length=512))
        },
        description=desc)
def embed(model, args):
    if args['tokenize_sentences'] == 'yes':
        sentences = sent_tokenize(args['text'])
    else:
        sentences = args['text'].split("\n")
    print('[EMBED] Embedding {} sentences'.format(len(sentences)))
    results = model.embed(sentences)
    return {'sentences': sentences, 'embeddings': results}

if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8000)

