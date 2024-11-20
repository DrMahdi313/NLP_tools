
import pytest
from main import masking_word

def test_masking_word():
    sentence = "This is a test sentence for the pipeline."
    masked_sentence, masked_word, idx = masking_word(sentence)
    assert "[MASK]" in masked_sentence
    assert masked_word in sentence.split()
    assert isinstance(idx, int)
