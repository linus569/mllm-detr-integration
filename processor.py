import torch
from transformers import AutoTokenizer


class Processor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.image_token = "<image>"
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [self.image_token]}
        )

    def prepare_text(self, captions, instance_classes, instance_bboxes):
        # concat captions and class names with special tokens
        text = (
            " ".join(captions)
            + " "
            + " ".join(self.image_token for _ in range(577))
            + " "
            + " ".join(
                [
                    f"<class_{cls}> <bbox_{bbox}>"
                    for cls, bbox in zip(instance_classes, instance_bboxes)
                ]
            )
        )
        return text

    def __call__(self, batch):
        images = [item["image"] for item in batch]
        texts = [
            self.prepare_text(
                item["captions"], item["instance_classes"], item["instance_bboxes"]
            )
            for item in batch
        ]

        # tokenize texts
        tokenized = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        # stack images
        images = torch.stack(images)

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "images": images,
        }


from dataset import train_dataset
from torch.utils.data import DataLoader

processor = Processor(model_name="lmms-lab/llava-onevision-qwen2-0.5b-si")
dataloader = DataLoader(train_dataset, batch_size=8, collate_fn=processor)

# print(next(iter(dataloader)))
