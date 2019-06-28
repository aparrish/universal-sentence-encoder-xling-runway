import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece

# this script just loads the model from tf-hub and then does nothing with it;
# it exists purely to "pre-cache" the model during the build phase.

module_url = "https://tfhub.dev/google/universal-sentence-encoder-xling-many/1"
module = hub.Module(module_url)
