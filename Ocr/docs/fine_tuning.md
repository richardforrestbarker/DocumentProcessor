# Fine-tuning LayoutLMv3 for Receipt Processing

This guide explains how to fine-tune LayoutLMv3 on receipt datasets for improved field extraction accuracy.

## Overview

Fine-tuning LayoutLMv3 on domain-specific receipt data significantly improves extraction accuracy for:
- Vendor names
- Dates
- Amounts (total, subtotal, tax)
- Line items
- Payment information

## Prerequisites

- Labeled receipt dataset (hundreds to thousands of receipts)
- GPU with 16GB+ VRAM (NVIDIA A100/V100 recommended)
- PyTorch 2.0+
- Transformers library
- Datasets library

## Dataset Preparation

### 1. Data Collection

Collect diverse receipt samples:
- Multiple vendors and formats
- Various receipt types (grocery, retail, restaurant, etc.)
- Different languages if needed
- Range of quality levels (clear to degraded)

Recommended datasets:
- **CORD** (Consolidated Receipt Dataset): 1,000 receipts with annotations
- **SROIE** (Scanned Receipts OCR and Information Extraction): 1,000 receipts
- **Custom dataset**: Label your own receipts for specific use cases

### 2. Data Labeling

Use annotation tools:
- **Label Studio**: Web-based annotation with OCR support
- **CVAT**: Computer Vision Annotation Tool
- **doccano**: Text annotation for NER

Annotation format (JSONL):
```json
{
  "id": "receipt_001",
  "image_path": "data/receipts/001.jpg",
  "words": [
    {
      "text": "TARGET",
      "box": [120, 50, 280, 90],
      "label": "B-VENDOR"
    },
    {
      "text": "STORE",
      "box": [290, 50, 380, 90],
      "label": "I-VENDOR"
    },
    {
      "text": "Total",
      "box": [100, 800, 200, 840],
      "label": "B-TOTAL"
    },
    {
      "text": "$45.99",
      "box": [450, 800, 550, 840],
      "label": "I-TOTAL"
    }
  ]
}
```

### 3. Data Preprocessing

Convert annotations to LayoutLM format:

```python
from datasets import Dataset
from transformers import LayoutLMv3Processor
from PIL import Image

def preprocess_dataset(annotations, processor):
    """Convert annotations to LayoutLMv3 format."""
    
    examples = []
    for ann in annotations:
        # Load image
        image = Image.open(ann['image_path'])
        
        # Prepare inputs
        encoding = processor(
            image,
            ann['words'],
            boxes=[w['box'] for w in ann['words']],
            word_labels=[label_to_id[w['label']] for w in ann['words']],
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        examples.append(encoding)
    
    return Dataset.from_list(examples)
```

## Training

### 1. Setup Training Script

```python
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer
)

# Load model and processor
model_name = "microsoft/layoutlmv3-base"
processor = LayoutLMv3Processor.from_pretrained(model_name)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(label_to_id)
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,  # Use mixed precision training
    logging_dir="./logs",
    logging_steps=100,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()
```

### 2. Hyperparameter Tuning

Key hyperparameters to tune:
- **Learning rate**: 3e-5 to 5e-5 (typical range)
- **Batch size**: 4-8 (limited by GPU memory)
- **Epochs**: 5-15 (monitor validation loss)
- **Warmup steps**: 500-1000
- **Weight decay**: 0.01-0.1

### 3. Data Augmentation

Improve robustness with augmentation. Install albumentations first:

```bash
pip install albumentations
```

Then use augmentation transforms:

```python
import albumentations as A

transform = A.Compose([
    A.GaussNoise(p=0.3),
    A.Rotate(limit=3, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Blur(blur_limit=3, p=0.3),
])
```

## Evaluation

### Metrics

Track these metrics during training:

1. **Field-level F1**: Precision and recall for each field type
2. **Entity-level accuracy**: Exact match for extracted entities
3. **Token-level accuracy**: Individual token classification accuracy

```python
from seqeval.metrics import f1_score, precision_score, recall_score

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    predictions = np.argmax(predictions, axis=2)
    
    # Remove padding
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }
```

### Test Set Evaluation

Evaluate on held-out test set:

```bash
python evaluate.py \
  --model-path ./results/checkpoint-best \
  --test-data ./data/test.jsonl \
  --output ./evaluation_results.json
```

## Model Optimization

### 1. Model Pruning

Reduce model size for faster inference:

```python
from transformers import LayoutLMv3ForTokenClassification
import torch.nn.utils.prune as prune

# Load fine-tuned model
model = LayoutLMv3ForTokenClassification.from_pretrained("./results/best_model")

# Apply pruning
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

### 2. Quantization

Reduce model precision for faster inference:

```python
import torch

# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 3. ONNX Export

Export for optimized inference:

```python
from transformers.onnx import export

# Export to ONNX
export(
    preprocessor=processor,
    model=model,
    config=model.config,
    opset=14,
    output_path="model.onnx"
)
```

## Deployment

### Save Fine-tuned Model

```python
# Save model and processor
model.save_pretrained("./finetuned_model")
processor.save_pretrained("./finetuned_model")
```

### Use in Production

Update configuration to use fine-tuned model:

```yaml
# config/config.yaml
model:
  name_or_path: "./finetuned_model"
  device: "cuda"
```

## Tips for Better Results

1. **Quality over quantity**: 500 well-labeled receipts > 5000 poorly labeled
2. **Balanced dataset**: Include diverse vendors, formats, and quality levels
3. **Regular evaluation**: Monitor metrics on validation set to avoid overfitting
4. **Iterative improvement**: Fine-tune → evaluate → collect more data → repeat
5. **Error analysis**: Review misclassifications to identify patterns
6. **Ensemble methods**: Combine multiple models for better accuracy

## Common Issues

### Overfitting

Symptoms:
- High training accuracy, low validation accuracy
- Model memorizes specific receipts

Solutions:
- Add more diverse training data
- Increase data augmentation
- Reduce model complexity
- Add regularization (dropout, weight decay)

### Poor Performance on Specific Vendors

Solutions:
- Collect more examples from underrepresented vendors
- Use targeted data augmentation
- Consider vendor-specific models for high-volume vendors

### Low Confidence Scores

Solutions:
- Calibrate model probabilities with temperature scaling
- Use uncertainty estimation techniques
- Ensemble multiple models

## Resources

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [CORD Dataset](https://github.com/clovaai/cord)
- [SROIE Dataset](https://rrc.cvc.uab.es/?ch=13)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Label Studio](https://labelstud.io/)

## Next Steps

1. Collect and label initial dataset (100-500 receipts)
2. Train baseline model
3. Evaluate and analyze errors
4. Expand dataset based on error patterns
5. Fine-tune iteratively
6. Deploy and monitor in production
