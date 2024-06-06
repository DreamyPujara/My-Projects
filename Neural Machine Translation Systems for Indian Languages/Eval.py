 from nltk.translate.bleu_score import sentence_bleu

    def evaluate_translation(reference, candidate):
        reference = [reference.split()]
        candidate = candidate.split()
        score = sentence_bleu(reference, candidate)
        return score
