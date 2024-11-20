from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from transformers import pipeline
from scipy.special import softmax
import random

# Initialize the BERT fill-mask pipeline
bert_mask_filler = pipeline("fill-mask", model="bert-base-uncased")

# Load stopwords
stopwords_set = set(stopwords.words('english'))

def filter_synonyms_with_bert(sentence, masked_word, synonyms):
    """
    Use BERT to filter and score synonyms based on their contextual relevance to the sentence.

    Args:
        sentence (str): The input sentence.
        masked_word (str): The word to be replaced.
        synonyms (set): A set of candidate synonyms.

    Returns:
        list: A list of tuples (synonym, normalized_score) sorted by relevance.
    """
    # Mask the original word in the sentence
    masked_sentence = sentence.replace(masked_word, "[MASK]", 1)
    print(f'Masked sentence: {masked_sentence}\n')
    
    # Get BERT predictions for the [MASK] token
    predictions = bert_mask_filler(masked_sentence)
    print(f'BERT predictions: {predictions}\n')
    
    scored_synonyms = []

    for pred in predictions:
        scored_synonyms.append((pred['token_str'], pred['score']))
    
    print(f'Scored synonyms by BERT (un-normalized): {scored_synonyms}\n')

    # Sort synonyms by score in descending order
    scored_synonyms.sort(key=lambda x: x[1], reverse=True)
    values = [x[1] for x in scored_synonyms]
    normalized_values = softmax(values)

    return [(x[0], sm) for x, sm in zip(scored_synonyms, normalized_values)]


def masking_word(sentence):
    """
    Randomly select a non-stopword from the sentence and mask it.

    Args:
        sentence (str): The input sentence.

    Returns:
        tuple: Masked sentence (str), the masked word (str), and its index (int). 
               Returns (None, None, None) if no valid word is found.
    """
    words = sentence.split()
    non_stopwords = [word for word in words if word.lower() not in stopwords_set and word.isalpha()]
    if not non_stopwords:
        return None, None, None
    
    random_word = random.choice(non_stopwords)
    rw_idx = words.index(random_word)
    masked_sentence = sentence.replace(random_word, '[MASK]', 1)

    return masked_sentence, random_word, rw_idx


def replace_with_synonym(sentence, top_k=5, threshold=0.9):
    """
    Replace a randomly chosen word in the sentence with its contextually relevant synonyms.

    Args:
        sentence (str): The input sentence.
        top_k (int): Number of top BERT predictions to consider.
        threshold (float): Minimum normalized score for a synonym to be considered.

    Returns:
        list: Sentences with replaced synonyms.
    """
    msk_sen, random_word, rw_idx = masking_word(sentence)
    if not random_word:
        print("No valid word to mask.")
        return []

    print(f'Masked word: {random_word}\n')

    synonyms = set()
    # Get synsets and filter synonyms based on POS
    masked_word_synsets = wordnet.synsets(random_word)
    tagged = pos_tag([random_word])[0]  # POS tag for masked word
    masked_word_pos = tagged[1][0].lower()

    for syn in masked_word_synsets:
        if syn.pos() == masked_word_pos:  # Only match same POS
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != random_word.lower():  # Avoid adding the original word
                    synonyms.add(synonym)
    
    print(f'Synonyms: {synonyms}\n')

    # Filter synonyms with BERT
    scored_synonyms = filter_synonyms_with_bert(sentence, random_word, synonyms)
    print(f'Scored synonyms by BERT: {scored_synonyms}\n')
    
    # Select final synonyms based on threshold
    final_synonyms = [syn for syn, score in scored_synonyms if score >= threshold]

    print(f'Final synonyms: {final_synonyms}\n')

    # Replace the mask with synonyms
    return [msk_sen.replace('[MASK]', syn, 1) for syn in final_synonyms]
