from runner import SketchTrainer, device
import torch
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from raster_dataset import svg_rasterize
import numpy as np

def to_3ch_tensor(img_pil):
    arr = np.array(img_pil, dtype=np.uint8)
    t = torch.from_numpy(arr).unsqueeze(0)
    return t.repeat(3, 1, 1) 

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
            acc = (correct / total).detach() if total > 0 else torch.tensor(0.0, device=device)
            test_token_accuracy += acc

    print(f"Test Next Token Accuracy: {test_token_accuracy / len(test_loader):.4f}")

    start_id = sketch_trainer.tokenizer.vocab["START"]
    end_id = sketch_trainer.tokenizer.vocab["END"]

    def _trim_at_end(ids):
        if end_id in ids:
            idx = ids.index(end_id)
            return ids[: idx + 1]
        return ids

    fid = FrechetInceptionDistance(input_img_size=(1, 299, 299)).to(device)
    inception = InceptionScore(splits=10, normalize=False).to(device)

    model.eval()
    with torch.no_grad():
        for input_ids, target_ids, class_labels in tqdm(test_loader, desc="FID/IS"):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            class_labels = class_labels.to(device)

            if use_padding_mask:
                mask = input_ids == sketch_trainer.tokenizer.pad_token_id
                logits = model(input_ids, class_labels, src_key_padding_mask=mask)
            else:
                logits = model(input_ids, class_labels)

            preds = logits.argmax(dim=-1).cpu()
            targets_cpu = target_ids.cpu()

            real_batch = []
            fake_batch = []          
            B = preds.size(0)
            for b in range(B):
                # REAL sequence = [START] + gold targets up to END
                real_ids = [start_id] + targets_cpu[b].tolist()
                real_ids = _trim_at_end(real_ids)
                real_svg = sketch_trainer.tokenizer.decode(real_ids)
                real_img = svg_rasterize(real_svg)
                r = to_3ch_tensor(real_img)

                # FAKE sequence = [START] + argmax preds up to END
                fake_ids = [start_id] + preds[b].tolist()
                fake_ids = _trim_at_end(fake_ids)
                fake_svg = sketch_trainer.tokenizer.decode(fake_ids)
                fake_img = svg_rasterize(fake_svg)
                f = to_3ch_tensor(fake_img)
                f2 = to_3ch_tensor(fake_img)

                real_batch.append(r.unsqueeze(0))  # 1x3x299x299
                fake_batch.append(f.unsqueeze(0))

            real_images = torch.cat(real_batch, dim=0).to(device)
            fake_images = torch.cat(fake_batch, dim=0).to(device)

            # Update FID (needs both real and fake)
            fid.update(real_images, real=True)
            fid.update(fake_images, real=False)

            # Update Inception Score (only generated images)
            inception.update(fake_images)

    fid_score = fid.compute().item()
    is_mean, is_std = inception.compute()
    is_mean = is_mean.item()
    is_std = is_std.item()

    sketch_trainer.writer.add_scalar("FID/Test", fid_score, 0)
    sketch_trainer.writer.add_scalar("IS/TestMean", is_mean, 0)
    sketch_trainer.writer.add_scalar("IS/TestStd", is_std, 0)

    print(f"Test FID: {fid_score:.4f}")
    print(f"Test Inception Score: mean={is_mean:.4f}, std={is_std:.4f}")
