import json
import re
import torch
from dataset.dataset import build_dataloader
from transformers import StoppingCriteria

# TODO: replace with config
TRAIN_DATA_DIR = "data/coco/images/train2017"
TRAIN_ANNOTATIONS_DIR = "data/coco/annotations/instances_train2017.json"
TRAIN_BATCH_SIZE = 1

VAL_DATA_DIR = "data/coco/images/val2017"
VAL_ANNOTATIONS_DIR = "data/coco/annotations/instances_val2017.json"
VAL_BATCH_SIZE = 1

TEST_DATA_DIR = "data/coco/images/test2017"
TEST_ANNOTATIONS_DIR = "data/coco/annotations/image_info_test2017.json"
TEST_BATCH_SIZE = 1

server = "/u/home/salzmann/Documents/dev/master-thesis/"

TRAIN_DATA_DIR = server + "data/coco/images/train2017"
TRAIN_ANNOTATIONS_DIR = server + "data/coco/annotations/instances_train2017.json"
VAL_DATA_DIR = server + "data/coco/images/val2017"
VAL_ANNOTATIONS_DIR = server + "data/coco/annotations/instances_val2017.json"
TEST_DATA_DIR = server + "data/coco/images/test2017"
TEST_ANNOTATIONS_DIR = server + "data/coco/annotations/image_info_test2017.json"


def build_train_dataloader(model, batch_size=TRAIN_BATCH_SIZE, num_samples=None):
    return build_dataloader(
        image_dir=TRAIN_DATA_DIR,
        annotations_file=TRAIN_ANNOTATIONS_DIR,
        batch_size=batch_size,
        model=model,
        train=True,
        num_samples=num_samples,
    )


def build_val_dataloader(model, batch_size=VAL_BATCH_SIZE, num_samples=None, random_index=True):
    return build_dataloader(
        image_dir=VAL_DATA_DIR,
        annotations_file=VAL_ANNOTATIONS_DIR,
        batch_size=batch_size,
        model=model,
        train=False,
        num_samples=num_samples,
        random_idx=random_index,
    )


def build_test_dataloader(model, batch_size=TEST_BATCH_SIZE, num_samples=None, random_index=True):
    return build_dataloader(
        image_dir=TEST_DATA_DIR,
        annotations_file=TEST_ANNOTATIONS_DIR,
        batch_size=batch_size,
        model=model,
        train=False,
        num_samples=num_samples,
        random_idx=random_index,
    )


def seed_everything(seed):
    pass


def save_training_checkpoint(todo):
    pass

def parse_model_output(text):
    try:
        # Replace single quotes with double quotes
        json_text = text.replace("'", '"')
        # Remove any text before first [ and after last ]
        json_text = json_text[json_text.find('['):json_text.rfind(']')+1]
        # Fix missing quotes around keys
        json_text = re.sub(r'(\w+):', r'"\1":', json_text)

        # Try to parse the JSON
        objects = json.loads(json_text)
        
        # Validate format of each object
        for obj in objects:
            if not isinstance(obj, dict):
                return None
            if 'class' not in obj or 'bbox' not in obj:
                return None
            if not isinstance(obj['bbox'], list) or len(obj['bbox']) < 4:
                return None
            if not all(isinstance(x, (int, float)) for x in obj['bbox']):
                return None
                
        return objects
    except json.JSONDecodeError as e:
        print(f"Failed to parse model output text '{text}' (converted to json text '{json_text}') with error {e}")
        return None
    

def parse_model_output_to_boxes(text, dataset, index, device):
    # TODO: improve code
    # parse generated ouput to extract boundingboxes
    pred_boxes = []
    pred_labels = []
    pred_scores = []
    failed_conversion = 0

    # Get original dataset from dataset if Subset is used
    if hasattr(dataset, 'dataset'):
        dataset = dataset.dataset

    text_to_parse = text[index].strip()
    predictions = parse_model_output(text_to_parse)

    if predictions is not None:
        for pred in predictions:
            try:
                # Convert bbox to tensor
                bbox = pred.get("bbox", [])
                if len(bbox) == 4:
                    pred_boxes.append(torch.tensor(bbox, dtype=torch.float32))
                    
                    # Convert class name to index
                    class_name = pred.get("class", "")
                    class_id = dataset.cat_name_to_id.get(class_name, 0)  # Default to 0 if not found
                    pred_labels.append(class_id)
                    
                    pred_scores.append(1.0)
            except (ValueError, TypeError) as e:
                failed_conversion += 1
            

    # TODO: do something with failed_conversion print or log or something


    bbox = torch.stack(pred_boxes) if pred_boxes else torch.zeros((0, 4))
    bbox = unnormalize_bbox(bbox.to(device), width=384, height=384) # TODO: get height and width from config

    # TODO: addd index again for batches
     # Create return tensors with proper types
    return {
        "boxes": bbox,
        "labels": torch.tensor(pred_labels, dtype=torch.long) if pred_labels else torch.zeros(0, dtype=torch.long),
        "scores": torch.tensor(pred_scores, dtype=torch.float32) if pred_scores else torch.zeros(0)
    }

def unnormalize_bbox(bbox: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """
    Unnormalize bounding boxes from [0,1] to pixel coordinates efficiently.
    Args:
        bbox: Tensor of shape (N, 4) or (4,) with normalized coordinates
        width: Original image width
        height: Original image height
    Returns:
        Tensor of same shape with pixel coordinates
    """
    if bbox.numel() == 0:
        return torch.zeros((0,4), device=bbox.device, dtype=bbox.dtype)
    
    if bbox.dim() == 1:
        bbox = bbox.unsqueeze(0)
    
    # Create scaling factor tensor
    scale = torch.tensor([width, height, width, height], 
                        device=bbox.device, 
                        dtype=bbox.dtype)
    
    # Vectorized multiplication
    bbox_unnorm = bbox * scale
    
    return bbox_unnorm if bbox.dim() == 2 else bbox_unnorm.squeeze(0)


class JSONStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.end_sequence = self.tokenizer.encode("]")[
            0
        ]  # Get token ID for closing bracket

    def __call__(self, input_ids, scores, **kwargs):
        # Stop if we find the closing bracket
        return input_ids[0][-1] == self.end_sequence
