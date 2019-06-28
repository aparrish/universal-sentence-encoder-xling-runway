# Universal Sentence Encoder (Runway port)

This repository contains a simple wrapper to use Google's Universal Sentence
Encoder with Runway.

## Usage

The Runway model exposes one command, `embed`, that
accepts a `text` parameter as input and produces two arrays as output: one with
an array of sentences in `text`, and another with sentence embeddings for each
sentence with the corresponding index in the array of sentences.

An optional `tokenize_sentences` parameter, if set to `yes`, uses NLTK's
`sent_tokenize` function to parse the value in `text` into sentences before
passing them along to the embedder. Otherwise, `text` is interpreted as a list
of sentences separated by newline characters.

## Details

This code uses the [Cross-lingual (XLING) Universal Sentence
Encoder](https://tfhub.dev/google/universal-sentence-encoder-xling-many/1),
which Google advertises as working with English, French, German, Spanish,
Italian, Chinese, Korean, and Japanese. But note that the sentence tokenizer
used in this port is specific to English, and will not return good results for
other languages! If you're getting embeddings for sentences from other
languages, you may want to set the `tokenize_sentences` parameter to `no` and
feed the model pre-tokenized sentences.
