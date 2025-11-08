from runner import SketchTrainer, device
import torch
from tqdm import tqdm


def test(sketch_trainer: SketchTrainer):
    """Evaluate the model on the test set and return average loss and accuracy."""
    model = sketch_trainer.model
    test_loader = sketch_trainer.test_loader
    use_padding_mask = sketch_trainer.use_padding_mask

    model.eval()
    test_token_accuracy = 0

    with torch.no_grad():
        for input_ids, target_ids, class_labels in tqdm(test_loader, desc="Testing"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            class_labels = class_labels.to(device)

            if use_padding_mask:
                mask = input_ids == sketch_trainer.tokenizer.pad_token_id
                logits = model(input_ids, class_labels, src_key_padding_mask=mask)
            else:
                logits = model(input_ids, class_labels)

            # Calculate accuracy of next token predictions
            preds = logits.argmax(dim=-1)
            mask = target_ids != sketch_trainer.tokenizer.pad_token_id
            correct = (preds[mask] == target_ids[mask]).float().sum()
            total = mask.sum()
            acc = (correct / total).detach() if total > 0 else 0.0
            test_token_accuracy += acc

    sketch_trainer.writer.add_scalar(
        "Accuracy/TestNextToken", test_token_accuracy / len(test_loader), 0
    )
    print(f"Test Next Token Accuracy: {test_token_accuracy / len(test_loader):.4f}")

    # TODO FID, Inception Score
    pass
