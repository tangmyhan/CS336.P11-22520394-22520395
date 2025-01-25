from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from underthesea import word_tokenize
import pandas as pd

def calculate_bleu(reference, hypothesis):
    """
    Tính BLEU score cho một cặp câu trả lời tham chiếu và câu trả lời của chatbot.

    Args:
    - reference (str): Câu trả lời tham chiếu.
    - hypothesis (str): Câu trả lời của chatbot.

    Returns:
    - float: BLEU score.
    """
    try:
        reference_tokens = word_tokenize(reference, format="text").split()
        hypothesis_tokens = word_tokenize(hypothesis, format="text").split()
        smoothing_function = SmoothingFunction().method1
        return sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)
    except Exception as e:
        print(f"Error calculating BLEU for reference: '{reference}' and hypothesis: '{hypothesis}' -> {e}")
        return 0.0

def calculate_rouge(reference, hypothesis):
    """
    Tính ROUGE score cho một cặp câu trả lời tham chiếu và câu trả lời của chatbot.

    Args:
    - reference (str): Câu trả lời tham chiếu.
    - hypothesis (str): Câu trả lời của chatbot.

    Returns:
    - dict: ROUGE scores.
    """
    try:
        rouge = Rouge()
        reference_tokens = " ".join(word_tokenize(reference, format="text").split())
        hypothesis_tokens = " ".join(word_tokenize(hypothesis, format="text").split())
        scores = rouge.get_scores(hypothesis_tokens, reference_tokens, avg=True)
        return scores
    except Exception as e:
        print(f"Error calculating ROUGE for reference: '{reference}' and hypothesis: '{hypothesis}' -> {e}")
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}

def evaluate_scores_batch(references, hypotheses):
    """
    Tính BLEU và ROUGE score trung bình cho tập dữ liệu.

    Args:
    - references (list of str): Danh sách các câu trả lời tham chiếu.
    - hypotheses (list of str): Danh sách các câu trả lời của chatbot.

    Returns:
    - dict: Trung bình BLEU và ROUGE scores.
    """
    bleu_scores = []
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}

    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:  # Kiểm tra dữ liệu rỗng
            print(f"Skipping evaluation for empty reference or hypothesis: ref='{ref}', hyp='{hyp}'")
            continue

        bleu_scores.append(calculate_bleu(ref, hyp))
        rouge_score = calculate_rouge(ref, hyp)
        for key in rouge_scores:
            rouge_scores[key].append(rouge_score[key]['f'])

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_rouge = {key: (sum(scores) / len(scores) if scores else 0.0) for key, scores in rouge_scores.items()}

    return {
        'bleu': avg_bleu,
        'rouge': avg_rouge
    }

if __name__ == "__main__":
    # Load data from CSV
    try:
        data = pd.read_csv('bleu1.csv')
        df = pd.DataFrame(data)
    except FileNotFoundError:
        print("File 'bleu.csv' not found. Please make sure the file exists in the working directory.")
        exit()

    # Extract queries, references, and hypotheses from DataFrame
    queries = df.get('question', []).tolist()
    references = df.get('answer', []).tolist()
    hypotheses = df.get('bot', []).tolist()

    # Evaluate scores
    avg_scores = evaluate_scores_batch(references, hypotheses)

    # Print results
    print(f"Average BLEU Score: {avg_scores['bleu']:.4f}")
    print(f"Average ROUGE Scores: {avg_scores['rouge']}")
