
    import pytest

    from main import masking_word

    def test_masking_word():
        sentence = "The quick brown fox jumps over the lazy dog."
        masked_sentence, masked_word, rw_idx = masking_word(sentence)
        
        assert masked_sentence is not None, "Masked sentence should not be None"
        assert masked_word in sentence, "Masked word should be in the original sentence"
        assert "[MASK]" in masked_sentence, "Masked sentence should include '[MASK]'"


    from main import masking_word

    def test_filter_synonyms_with_bert():
        sentence = "The quick brown fox jumps over the lazy dog."
        masked_word = "quick"
        synonyms = {"fast", "swift", "rapid"}
        
        scored_synonyms = filter_synonyms_with_bert(sentence, masked_word, synonyms)
        
        assert len(scored_synonyms) > 0, "BERT should return scored synonyms"
        assert all(isinstance(score, float) for _, score in scored_synonyms), "Scores should be floats"

    from main import masking_word

    def test_replace_with_synonym():
        sentence = "The quick brown fox jumps over the lazy dog."
        augmented_sentences = replace_with_synonym(sentence, threshold=0.9)
        
        assert isinstance(augmented_sentences, list), "Output should be a list"
        assert len(augmented_sentences) > 0, "Pipeline should produce at least one augmented sentence"
        assert all("[MASK]" not in sent for sent in augmented_sentences), "No sentence should contain '[MASK]'"

