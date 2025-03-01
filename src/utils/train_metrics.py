import logging
from typing import Dict, List

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from torchmetrics.detection.mean_ap import MeanAveragePrecision

log = logging.getLogger(__name__)


class TrainMetrics:
    def __init__(self, device, download_nltk=True):
        self.device = device
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox").to(
            device
        )
        self.total_bleu_score = 0
        self.total_meteor_score = 0
        self.num_samples = 0

        if download_nltk:
            """Download NLTK data with proper error handling."""
            nltk_data = ["wordnet", "omw-1.4", "punkt_tab"]
            for resource in nltk_data:
                try:
                    nltk.download(resource, quiet=True)
                    log.info(f"Downloaded NLTK resource: {resource}")
                except Exception as e:
                    log.warning(f"Failed to download NLTK resource '{resource}': {e}")
                    log.warning("Some metrics may not be available")

    def reset(self):
        """Reset metrics."""
        self.metric.reset()
        self.total_bleu_score = 0
        self.total_meteor_score = 0
        self.num_samples = 0

    def update(
        self,
        predicted_boxes: List[Dict],
        target_boxes: List[Dict],
        generated_text: List[str],
        target_texts: List[str],
    ):
        """Update metrics with batch data.

        Args:
            predicted_boxes: Predicted bounding boxes.
            target_boxes: Target bounding boxes.
            generated_text : Generated text.
            target_texts: Target text.
        """
        try:
            self.metric.update(predicted_boxes, target_boxes)

            for pred, target in zip(generated_text, target_texts):
                self.total_bleu_score += sentence_bleu(
                    [target.split()],
                    pred.split(),
                    smoothing_function=SmoothingFunction().method1,
                )

                generated_text_tokens = nltk.word_tokenize(pred)
                reference_texts_tokens = [nltk.word_tokenize(target)]
                self.total_meteor_score += meteor_score(
                    reference_texts_tokens, generated_text_tokens
                )

                self.num_samples += 1
        except Exception as e:
            log.error(f"Error updating metrics: {e}")

    def compute(self) -> Dict[str, float]:
        """Compute metrics.
        
        Returns:
            Dict[str, float]: Computed metrics.
        """
        metrics = self.metric.compute()
        avg_bleu_score = (
            self.total_bleu_score / self.num_samples if self.num_samples > 0 else 0
        )
        avg_meteor_score = (
            self.total_meteor_score / self.num_samples if self.num_samples > 0 else 0
        )

        # TODO: json accuaracy
        
        return {
            "map": metrics["map"].item(),
            "map_50": metrics["map_50"].item(),
            "map_75": metrics["map_75"].item(),
            "bleu_score": avg_bleu_score,
            "meteor_score": avg_meteor_score,
        }
