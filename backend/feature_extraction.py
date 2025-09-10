"""
Advanced feature extraction for AI detection and plagiarism checking.
This module provides sophisticated NLP and code analysis features.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np
import math
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Constants for text analysis
FUNCTION_PATTERNS = {
    'python': r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(',
    'javascript': r'(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?\(?(?:[a-zA-Z_][a-zA-Z0-9_]*(?:,\s*[a-zA-Z_][a-zA-Z0-9_]*)*|\s*)?\)?\s*=>)',
    'java': r'(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^\)]*\)',
    'c': r'[\w\*]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^\)]*\)',
    'cpp': r'[\w\*]+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^\)]*\)',
}


def extract_basic_features(text):
    """Extract basic text features for analysis"""
    if not isinstance(text, str) or not text.strip():
        return None
    
    # Basic text stats
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    if word_count == 0:
        return None  # Can't analyze empty text
    
    # Set of stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()  # Fallback if stopwords nobt available
    
    # Calculate features
    features = {
        # Lexical diversity features
        'unique_words_ratio': len(set(words)) / word_count,
        'unique_non_stop_ratio': len([w for w in set(words) if w not in stop_words]) / max(len([w for w in words if w not in stop_words]), 1),
        
        # Length-based features
        'avg_word_length': sum(len(word) for word in words) / word_count,
        'avg_sentence_length': word_count / max(sentence_count, 1),
        'long_words_ratio': len([w for w in words if len(w) > 8]) / word_count,
        
        # Sentence complexity
        'avg_commas_per_sentence': sum(sentence.count(',') for sentence in sentences) / max(sentence_count, 1),
        'avg_semicolons_per_sentence': sum(sentence.count(';') for sentence in sentences) / max(sentence_count, 1),
        
        # Vocabulary richness
        'hapax_legomena_ratio': len([w for w in set(words) if words.count(w) == 1]) / word_count,
        
        # Punctuation and capitalization
        'punctuation_ratio': len([c for c in text if c in ',.;:!?()[]{}"\'-']) / max(len(text), 1),
        'uppercase_ratio': len([c for c in text if c.isupper()]) / max(len(text), 1),
    }
    
    return features


def extract_advanced_features(text):
    """Extract more sophisticated features for AI content detection"""
    if not isinstance(text, str) or not text.strip():
        return None
        
    # Get basic features first
    basic_features = extract_basic_features(text)
    if basic_features is None:
        return None
        
    # Advanced text analysis
    words = word_tokenize(text.lower())
    sentences = sent_tokenize(text)
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Set of stopwords
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()  # Fallback if stopwords not available
    
    # Type-Token Ratio (measure of lexical diversity)
    type_token_ratio = len(set(words)) / max(word_count, 1)
    
    # Content words analysis (non-stop words)
    content_words = [w for w in words if w not in stop_words]
    stop_words_ratio = (word_count - len(content_words)) / max(word_count, 1)
    
    # POS tagging
    try:
        pos_tags = nltk.pos_tag(words)
        pos_counts = Counter([tag for _, tag in pos_tags])
        total_tags = len(pos_tags)
        
        # Calculate POS ratios
        noun_ratio = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                      pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / max(total_tags, 1)
        verb_ratio = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                      pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                      pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / max(total_tags, 1)
        adj_ratio = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                     pos_counts.get('JJS', 0)) / max(total_tags, 1)
        adv_ratio = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                     pos_counts.get('RBS', 0)) / max(total_tags, 1)
        
        # Conjunction and determiner ratios (often different in AI text)
        conj_ratio = (pos_counts.get('CC', 0) + pos_counts.get('IN', 0)) / max(total_tags, 1)
        det_ratio = pos_counts.get('DT', 0) / max(total_tags, 1)
        
        # Syntactic complexity: measure of subordination
        subordinating_conj = ['although', 'because', 'since', 'unless', 'whereas', 'while']
        subord_ratio = len([w for w in words if w.lower() in subordinating_conj]) / max(word_count, 1)
        
    except Exception as e:
        # Fallback if POS tagging fails
        noun_ratio = verb_ratio = adj_ratio = adv_ratio = conj_ratio = det_ratio = subord_ratio = 0
    
    # Sentence structure analysis
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    sentence_length_variance = np.var(sentence_lengths) if len(sentence_lengths) > 1 else 0
    
    # Sentence complexity analysis
    complex_sentence_markers = [',', ';', 'which', 'that', 'who', 'whom', 'whose']
    complex_sentence_count = sum(1 for s in sentences if any(marker in s.lower() for marker in complex_sentence_markers))
    complex_sentence_ratio = complex_sentence_count / max(sentence_count, 1)
    
    # Word frequency distribution
    word_freqs = Counter(words)
    freq_words = [word for word, count in word_freqs.most_common(int(word_count * 0.1))]
    freq_words_ratio = sum(word_freqs[w] for w in freq_words) / max(word_count, 1) if freq_words else 0
    
    # Perplexity estimation (sophisticated version)
    # A measure of how predictable the text is
    if word_count > 15:
        # Use both unigrams, bigrams and trigrams for better perplexity estimation
        bigrams = list(zip(words[:-1], words[1:]))
        trigrams = list(zip(words[:-2], words[1:-1], words[2:])) if len(words) > 2 else []
        
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        unigram_counts = Counter(words)
        
        # Interpolated language model (combine unigram, bigram, trigram probabilities)
        log_probs = []
        lambda1, lambda2, lambda3 = 0.1, 0.4, 0.5  # Weights for interpolation
        
        for i in range(2, len(words)):
            w1, w2, w3 = words[i-2], words[i-1], words[i]
            
            # Unigram probability
            p_uni = unigram_counts[w3] / max(word_count, 1)
            
            # Bigram probability
            p_bi = bigram_counts[(w2, w3)] / max(unigram_counts[w2], 1)
            
            # Trigram probability
            p_tri = trigram_counts[(w1, w2, w3)] / max(bigram_counts[(w1, w2)], 1)
            
            # Interpolated probability
            p_interp = lambda1 * p_uni + lambda2 * p_bi + lambda3 * p_tri
            log_probs.append(math.log(p_interp) if p_interp > 0 else -10)
        
        perplexity = math.exp(-sum(log_probs) / max(len(log_probs), 1)) if log_probs else 100
    else:
        # Simplified perplexity for short texts
        perplexity = 100
    
    # Burstiness - measure of word usage patterns
    # AI text often has more even word distributions than human text
    if word_count > 10:
        word_freqs_values = np.array(list(word_freqs.values()))
        burstiness = np.std(word_freqs_values) / max(np.mean(word_freqs_values), 1)
        
        # Zipf's law compliance (how well the word distribution follows Zipf's law)
        # Human text often follows Zipf's law more closely
        ranks = np.arange(1, len(word_freqs_values) + 1)
        sorted_freqs = np.sort(word_freqs_values)[::-1]  # Sort in descending order
        
        # Calculate expected frequencies according to Zipf's law
        total_freq = np.sum(sorted_freqs)
        harmonic_sum = np.sum(1 / ranks)
        expected_freqs = total_freq * (1 / ranks) / harmonic_sum
        
        # Measure deviation from Zipf's law (lower is more human-like)
        zipf_deviation = np.mean(np.abs(sorted_freqs - expected_freqs) / expected_freqs)
    else:
        burstiness = 0
        zipf_deviation = 0
        
    # Transition words analysis (expanded list)
    transition_words = [
        'however', 'therefore', 'thus', 'hence', 'consequently', 'moreover', 
        'furthermore', 'nevertheless', 'conversely', 'similarly', 'likewise',
        'in addition', 'in contrast', 'on the other hand', 'specifically',
        'for example', 'in particular', 'indeed', 'in fact', 'in conclusion'
    ]
    
    transition_patterns = [r'\b' + re.escape(w) + r'\b' for w in transition_words]
    transition_count = sum(len(re.findall(pattern, text.lower())) for pattern in transition_patterns)
    transition_ratio = transition_count / max(sentence_count, 1)
        
    # Add advanced features to basic ones
    advanced_features = {
        # Lexical diversity
        'type_token_ratio': type_token_ratio,
        'stop_words_ratio': stop_words_ratio,
        
        # POS tag distributions
        'noun_ratio': noun_ratio,
        'verb_ratio': verb_ratio,
        'adjective_ratio': adj_ratio,
        'adverb_ratio': adv_ratio,
        'conjunction_ratio': conj_ratio,
        'determiner_ratio': det_ratio,
        'subordination_ratio': subord_ratio,
        
        # Sentence structure
        'sentence_length_variance': sentence_length_variance,
        'sentence_length_max': max(sentence_lengths) if sentence_lengths else 0,
        'complex_sentence_ratio': complex_sentence_ratio,
        
        # Predictability and distribution
        'frequent_words_ratio': freq_words_ratio,
        'perplexity': perplexity,
        'burstiness': burstiness,
        'zipf_deviation': zipf_deviation,
        
        # Transition features
        'transition_ratio': transition_ratio,
    }
    
    # Combine basic and advanced features
    combined_features = {**basic_features, **advanced_features}
    return combined_features


def extract_code_features(code, language='python'):
    """Extract features specific to code files"""
    if not isinstance(code, str) or not code.strip():
        return None
    
    # Remove comments based on language
    if language in ['python']:
        # Remove Python comments
        code_clean = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code_clean = re.sub(r'""".*?"""', '', code_clean, flags=re.DOTALL)
        code_clean = re.sub(r"'''.*?'''", '', code_clean, flags=re.DOTALL)
    elif language in ['javascript', 'java', 'c', 'cpp']:
        # Remove C-style comments
        code_clean = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)
    else:
        code_clean = code
    
    # Lines of code
    lines = code_clean.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    # Function detection
    function_pattern = FUNCTION_PATTERNS.get(language, FUNCTION_PATTERNS['python'])
    functions = re.findall(function_pattern, code_clean)
    if isinstance(functions[0], tuple) if functions else False:
        # Handle case where regex has multiple capture groups
        functions = [f[0] if f[0] else f[1] for f in functions]
    
    # Variable detection
    var_pattern = r'(?:var|let|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)' if language == 'javascript' else r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    variables = re.findall(var_pattern, code_clean)
    
    # Indentation consistency
    indentation_pattern = r'^(\s*)[^\s]'
    indentations = [len(re.match(indentation_pattern, line).group(1)) if re.match(indentation_pattern, line) else 0 
                  for line in lines if line.strip()]
    indentation_consistency = np.std(indentations) if indentations else 0
    
    # Control flow complexity
    control_structures = len(re.findall(r'(?:if|for|while|switch)\s*\(', code_clean))
    nesting_depth = 0
    max_nesting = 0
    for line in lines:
        opening = line.count('{') - line.count('}') if language in ['javascript', 'java', 'c', 'cpp'] else line.count(':')
        nesting_depth += opening
        max_nesting = max(max_nesting, nesting_depth)
    
    # Extract features specific to code
    features = {
        'loc': len(non_empty_lines),
        'function_count': len(functions),
        'variable_count': len(variables),
        'control_structures': control_structures,
        'max_nesting_depth': max_nesting,
        'indentation_consistency': indentation_consistency,
        'avg_function_length': len(non_empty_lines) / max(len(functions), 1) if functions else 0,
        'comment_ratio': 1 - len(code_clean) / max(len(code), 1) if code else 0,
        'avg_variable_name_length': sum(len(v) for v in variables) / max(len(variables), 1) if variables else 0,
    }
    
    # Combine with text features
    text_features = extract_basic_features(code_clean)
    if text_features:
        features.update(text_features)
    
    return features


def get_code_fingerprint(code, language='python'):
    """
    Generate a code fingerprint - a structural representation of the code
    that can identify similar code even when variable names change
    """
    if not isinstance(code, str) or not code.strip():
        return []
        
    # Remove comments
    if language in ['python']:
        code_clean = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code_clean = re.sub(r'""".*?"""', '', code_clean, flags=re.DOTALL)
        code_clean = re.sub(r"'''.*?'''", '', code_clean, flags=re.DOTALL)
    elif language in ['javascript', 'java', 'c', 'cpp']:
        code_clean = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        code_clean = re.sub(r'/\*.*?\*/', '', code_clean, flags=re.DOTALL)
    else:
        code_clean = code
    
    # Replace variable names with placeholders
    # This helps detect similar code structure even when variable names change
    var_pattern = r'(?:var|let|const)\s+([a-zA-Z_][a-zA-Z0-9_]*)' if language == 'javascript' else r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    variables = set(re.findall(var_pattern, code_clean))
    
    normalized_code = code_clean
    for i, var in enumerate(variables):
        normalized_code = re.sub(r'\b' + re.escape(var) + r'\b', f'VAR{i}', normalized_code)
    
    # Extract structural patterns
    patterns = {
        'func_signatures': re.findall(r'(?:function|def)\s+\w+\s*\([^)]*\)', normalized_code),
        'class_defs': re.findall(r'class\s+\w+(?:\s+extends|\s+implements|\s*:)?', normalized_code),
        'control_structures': re.findall(r'(?:if|for|while|switch)\s*\([^)]*\)', normalized_code),
        'structure_blocks': re.findall(r'[{}\[\]():;]', normalized_code)
    }
    
    # Combine patterns into a fingerprint
    fingerprint = []
    for pattern_type, matches in patterns.items():
        fingerprint.extend(matches)
    
    # Add simplified representation of code structure 
    # Convert the code to a sequence of control flow tokens
    structure = re.sub(r'[^{}()\[\];:]', '', normalized_code)
    fingerprint.append(structure)
        
    return fingerprint
