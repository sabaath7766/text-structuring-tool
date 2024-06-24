import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
from jinja2 import Environment, FileSystemLoader


# Initialize the spaCy NLP pipeline
nlp = spacy.load("en_core_web_sm")


def filter_pos(text, allowed_pos):
    doc = nlp(text)
    return ' '.join([token.text for token in doc if token.pos_ in allowed_pos])


def hash_features(features):
    return [hashlib.md5(feature.encode()).hexdigest() for feature in features]


def jaccard_similarity(set1, set2):
    set1 = set([str(item).lower() for item in set1])
    set2 = set([str(item).lower() for item in set2])
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / float(union)


def loss_function(predicted, gold_standard):
    loss = 0
    for p, g in zip(predicted, gold_standard):
        p = [str(item).lower() for item in p]
        g = [str(item).lower() for item in g]
        loss += 1 - jaccard_similarity(set(p), set(g))
    return loss / len(predicted)


def new_loss_function(predicted, gold_standard):
    tp = 0  # True Positives
    fp = 0  # False Positives
    fn = 0  # False Negatives

    for p, g in zip(predicted, gold_standard):
        p_set = set([str(item).lower() for item in p])
        g_set = set([str(item).lower() for item in g])

        tp += len(p_set & g_set)
        fp += len(p_set - g_set)
        fn += len(g_set - p_set)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return 1 - f1  # We want to minimize this, so 1 - F1


def is_adjacent_in_text(bigram, original_text):
    word1, word2 = bigram.split()
    pattern = f"{word1} {word2}"
    return pattern in original_text.lower()


def is_proper_noun_phrase(bigram, text):
    doc = nlp(text)
    tokens = {token.text: token for token in doc}
    word1, word2 = bigram.split()
    if word1 in tokens and word2 in tokens:
        token1 = tokens[word1]
        token2 = tokens[word2]

        # Check if both tokens are proper nouns and if token2 is the immediate right child of token1
        return token1.pos_ == "PROPN" and token2.pos_ == "PROPN" and token2 in list(token1.children)
    return False


def run_pipeline(params, num_unigrams, num_bigrams, allowed_pos, profiles, gold_standard_features):
    # Filter profiles based on POS tags
    filtered_profiles = [filter_pos(profile, allowed_pos) for profile in profiles]

    # Create a vectorizer for both unigrams and bigrams
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), **params)

    # Generate TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(filtered_profiles)

    # Convert the matrix to a DataFrame
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Initialize list to store top features for each profile
    final_top_features_list = []

    # Loop through each profile to get top features
    for i in range(df.shape[0]):
        combined_scores = df.iloc[i].sort_values(ascending=False)
        final_top_features = []
        bigrams = []
        top_unigrams = []

        for feature, score in combined_scores.items():
            if ' ' in feature:  # This is a bigram
                is_adj = is_adjacent_in_text(feature, profiles[i])
                is_pnp = is_proper_noun_phrase(feature, profiles[i])
                word1, word2 = feature.split()

                pos1 = nlp(word1)[0].pos_
                pos2 = nlp(word2)[0].pos_

                # Penalize bigrams where the second word is a number or an adjective
                if pos2 in {'NUM', 'ADJ'}:
                    score *= 0.3

                # Boost bigrams that are PROPN and NOUN combos
                if pos1 == 'PROPN' and pos2 == 'NOUN':
                    score *= 1.8

                if pos1 == 'NUM':
                    score *= 1.6

                priority = 1.5 if is_adj and is_pnp else 1.3 if is_adj or is_pnp else 1
                bigrams.append((feature, score * priority))

            else:  # This is a unigram
                pos = nlp(feature)[0].pos_

                # Prioritize numbers, and specific nouns like activities
                if pos in {'NUM'}:
                    score *= 2
                elif pos in {'NOUN'}:
                    score *= 1.5

                # Deprioritize adjectives and other generic terms
                if pos in {'ADJ', 'ADV'}:
                    score *= 0.5

                top_unigrams.append((feature, score))


        # Sort bigrams by priority (higher priority comes first)
        bigrams.sort(key=lambda x: x[1], reverse=True)

        # Add bigrams to final_top_features
        for bigram, _ in bigrams[:num_bigrams]:
            final_top_features.append(bigram)

        # Add a condition to only include the unigram if it's not part of any top bigram
        for unigram, score in top_unigrams:
            if not any(unigram in bigram for bigram in final_top_features):
                final_top_features.append(unigram)

            if len(final_top_features) >= num_unigrams + num_bigrams:
                break

        # Remove redundant features
        final_top_features = remove_redundant_features(final_top_features)

        final_top_features_list.append(final_top_features)

    gold_standard_features = [[feature.lower() for feature in sublist] for sublist in gold_standard_features]

    # Calculate loss
    loss = new_loss_function(final_top_features_list, gold_standard_features)

    return final_top_features_list, loss


def remove_redundant_features(top_features):
    to_remove = []
    for feature in top_features:
        if ' ' not in feature:  # This is a unigram
            if any(feature in bigram.split() for bigram in top_features if ' ' in bigram):  # Check if it's part of any bigram
                to_remove.append(feature)

    return [f for f in top_features if f not in to_remove]


def generate_html_report(final_results):
    # Set up Jinja2 environment and load template
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report_template.html')

    # Render template with final results data
    html_output = template.render(final_results=final_results)

    # Save to file
    with open("profiles_simple_extended.html", "w") as file:
        file.write(html_output)


# Define the best parameters and other settings
best_params = {'min_df': 0.01, 'max_df': 0.6, 'use_idf': True, 'sublinear_tf': True, 'norm': 'l1',
               'smooth_idf': True, 'analyzer': 'word'}
num_unigrams = 3
num_bigrams = 4
allowed_pos = ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM']

# List of profile and gold standard files
profile_files = ["profiles_simple_extended.txt"]
gold_standard_files = ["gold_standard_simple_extended.txt"]

final_results = []

# Loop to collect results
for profile_file, gold_standard_file in zip(profile_files, gold_standard_files):
    with open(profile_file, "r") as pf, open(gold_standard_file, "r") as gf:
        profiles = pf.readlines()
        gold_standard = [[item.lower() for item in line.strip().split(", ")] for line in gf.readlines()]
        gold_standard_flat = [item.lower() for sublist in gold_standard for item in sublist]

    final_top_features, final_loss = run_pipeline(
        best_params, num_unigrams, num_bigrams, allowed_pos, profiles, gold_standard
    )

    modified_top_features = []
    for profile in final_top_features:
        modified_profile = []
        for feature in profile:
            if feature in gold_standard_flat:
                modified_profile.append(f"<b>{feature}</b>")
            else:
                modified_profile.append(f"<i>{feature}</i>")
        modified_top_features.append(modified_profile)

    final_results.append({
        'file': profile_file,
        'params': best_params,
        'loss': final_loss,
        'top_features': modified_top_features
    })
    print(modified_top_features)


# Generate HTML report
generate_html_report(final_results)
