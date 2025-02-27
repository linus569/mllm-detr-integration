import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision

class TrainMetrics:
    def __init__(self, device):
        self.device = device
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(device)
        self.total_bleu_score = 0
        self.total_meteor_score = 0
        self.num_samples = 0

    def reset(self):
        self.metric.reset()
        self.total_bleu_score = 0
        self.total_meteor_score = 0
        self.num_samples = 0

    def update(self, predicted_boxes, target_boxes, generated_text, target_texts):
        self.metric.update(predicted_boxes, target_boxes)
        for pred, target in zip(generated_text, target_texts):
            self.total_bleu_score += sentence_bleu([target.split()], pred.split(), smoothing_function=SmoothingFunction().method1)
            generated_text_tokens = nltk.word_tokenize(pred)
            reference_texts_tokens = [nltk.word_tokenize(target)]
            self.total_meteor_score += meteor_score(reference_texts_tokens, generated_text_tokens)
            self.num_samples += 1

    def compute(self):
        metrics = self.metric.compute()
        avg_bleu_score = self.total_bleu_score / self.num_samples if self.num_samples > 0 else 0
        avg_meteor_score = self.total_meteor_score / self.num_samples if self.num_samples > 0 else 0
        return {
            "map": metrics["map"].item(),
            "map_50": metrics["map_50"].item(),
            "map_75": metrics["map_75"].item(),
            "bleu_score": avg_bleu_score,
            "meteor_score": avg_meteor_score,
        }

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')