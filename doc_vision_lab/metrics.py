import re

def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing extra whitespace."""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate Character Error Rate (CER) using Levenshtein distance.
    If python-Levenshtein is not installed, it falls back to a custom implementation.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    
    if len(ref) == 0:
        return 100.0 if len(hyp) > 0 else 0.0

    try:
        import Levenshtein
        distance = Levenshtein.distance(ref, hyp)
    except ImportError:
        distance = _levenshtein_distance(ref, hyp)

    cer = (distance / len(ref)) * 100
    return min(cer, 100.0) # Cap at 100% for UI readability

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Calculate Word Error Rate (WER).
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if len(ref_words) == 0:
        return 100.0 if len(hyp_words) > 0 else 0.0

    try:
        import Levenshtein
        distance = Levenshtein.distance(ref_words, hyp_words)
    except ImportError:
        # Custom implementation for list of words
        distance = _levenshtein_distance(ref_words, hyp_words)

    wer = (distance / len(ref_words)) * 100
    return min(wer, 100.0)

def _levenshtein_distance(s1, s2):
    """Fallback Levenshtein distance implementation."""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(1 + min((distances[index1], distances[index1+1], new_distances[-1])))
        distances = new_distances
    return distances[-1]
