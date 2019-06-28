import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
import os

from typing import List

module_url = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"

class UniSentenceEncXling:
    def __init__(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
            xling_8_embed = hub.Module(module_url)
            self.embedded_text = xling_8_embed(self.text_input)
            init_op = tf.group([
                tf.global_variables_initializer(),
                tf.tables_initializer()])
        self.g.finalize()
        self.session = tf.Session(graph=self.g)
        self.session.run(init_op)
    def embed(self, sentences: List[str]):
        result = self.session.run(
                self.embedded_text,
                feed_dict={self.text_input: sentences})
        return result

