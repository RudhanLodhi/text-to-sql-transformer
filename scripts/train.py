from pathlib import Path
from tqdm import tqdm
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
import sentencepiece as spm
from model.transformer import build_transformer
from scripts.dataset import make_loader
from scripts.data_prep import DATASET
from scripts.tokenizer import TOKENIZER, PAD_ID, read_pairs


CHECKPOINTS = Path("artifacts/checkpoints"); CHECKPOINTS.mkdir(exist_ok=True, parents=True)
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True, parents=True)
FIGURES = RESULTS / "figures"; FIGURES.mkdir(exist_ok=True, parents=True)

batch_size = 64

max_len = 512
d_model = 256
h = 4
N = 3
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

clip = 5
tf_start = 0.9
tf_end = 0.3
num_epochs = 20

warmup = 4000


def tf_ratio(epoch):
    """calculate the teacher forcing ratio for a given epoch

    Args:
        epoch (int): the current epoch

    Returns:
        float: the teacher forcing ratio for the given epoch
    """
    return tf_start + (tf_end - tf_start) * (epoch / num_epochs)


def lr_schedular(step):
    """transformer paper lr schedular

    Args:
        step (int): global step number of training
    """
    step = max(1, int(step))
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup ** -1.5))


def run_epoch(model, dataloader, optimizer, criterion, schedular, device, tf, pad_idx, clip, lr_schedule, is_training=True):
    """run one complete training or validation epoch

    Args:
        model (torch.nn.Module): transformer model to train or evaluate
        dataloader (torch.utils.data.DataLoader): batch of padded source and target token ids
        optimizer (torch.optim.Optimizer): optimizer used when training
        criterion (torch.nn.Module): token-level loss function, configured to ignore padding ids
        schedular (torch.optim.lr_scheduler.LambdaLR): learning-rate scheduler stepped after each optimizer update
        device (str or torch.device): device on which tensors and the model are placed
        tf (float): teacher-forcing probability passed to model
        pad_idx (int): padding token id excluded from loss aggregation
        clip (float): maximum gradient norm used during training
        lr_schedule (list[float]): List to which each training-step learning rate is appended for plotting
        is_training (bool, optional): if True perform backpropagation and optimizer updates otherwise run evaluation without updates

    Returns:
        float: average cross-entropy loss per non-padding target token
    """
    model.train(is_training)

    total_loss = 0.0
    total_tokens = 0

    tqdm_bar = tqdm(
        dataloader,
        desc="Training" if is_training else "Validation",
        leave=False
    )

    for idx, (src, tgt) in enumerate(tqdm_bar):
        src = src.to(device)
        tgt = tgt.to(device)

        target = tgt[:, 1:]
        logits = model(src, tgt, tf, pad_idx)
        loss = criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            target.contiguous().view(-1),
        )

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            schedular.step()
            lr_schedule.append(optimizer.param_groups[0]["lr"])

        non_pad_tokens = (target != pad_idx).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

        running_loss = total_loss / total_tokens
        postfix = {
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{running_loss:.4f}",
        }

        if is_training:
            lr = optimizer.param_groups[0]["lr"]

            postfix.update({
                "lr": f"{lr:.2e}",
                "tf": f"{tf:.2f}",
            })

        tqdm_bar.set_postfix(postfix)

    if total_tokens == 0:
        return 0.0

    return total_loss / total_tokens


def dataset_statistics(dataloader, path):
    """return the dataset statistics used in the markdown report
    """
    items = dataloader.dataset.items
    source_lengths = [len(src) for src, _ in items]
    target_lengths = [len(tgt) for _, tgt in items]
    total_pairs = len(read_pairs(path))

    return {
        "pairs": len(items),
        "mean_src": sum(source_lengths) / len(source_lengths),
        "max_src": max(source_lengths),
        "mean_tgt": sum(target_lengths) / len(target_lengths),
        "max_tgt": max(target_lengths),
        "dropped": total_pairs - len(items),
    }


if __name__ == "__main__":
    sp = spm.SentencePieceProcessor(model_file=f"{TOKENIZER}/sql_sp.model")

    train_dl = make_loader(f"{DATASET}/train_pairs.jsonl", sp, train=True, batch_size=batch_size)
    dev_dl = make_loader(f"{DATASET}/dev_pairs.jsonl", sp, train=False, batch_size=batch_size)
    test_dl = make_loader(f"{DATASET}/test_pairs.jsonl", sp, train=False, batch_size=batch_size)

    train_stats = dataset_statistics(train_dl, f"{DATASET}/train_pairs.jsonl")
    dev_stats = dataset_statistics(dev_dl, f"{DATASET}/dev_pairs.jsonl")
    test_stats = dataset_statistics(test_dl, f"{DATASET}/test_pairs.jsonl")

    model = build_transformer(
        vocab_size=sp.get_piece_size(),
        max_len=max_len,
        d_model=d_model,
        h=h,
        N=N,
        d_ff=4*d_model,
        dropout=dropout,
        device=device
    )
    optimizer = Adam(
        model.parameters(),
        lr=1.0,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    criterion = CrossEntropyLoss(
        label_smoothing=0.1,
        ignore_index=PAD_ID,
    )
    schedular = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: lr_schedular(step + 1),
    )

    best_dev_loss = float("inf")
    best_dev_epoch = 0
    best_checkpoint = CHECKPOINTS / "best.pt"

    epoch_train_losses, epoch_valid_losses = [], []
    lr_schedule = []

    training_start = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        train_loss = run_epoch(model, train_dl, optimizer, criterion, schedular, device, 1.0, PAD_ID, clip, lr_schedule, True)
        with torch.no_grad():
            dev_loss = run_epoch(model, dev_dl, optimizer, criterion, schedular, device, 0.0, PAD_ID, clip, lr_schedule, False)
            tf_loss = run_epoch(model, dev_dl, optimizer, criterion, schedular, device, 1.0, PAD_ID, clip, lr_schedule, False)

        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(dev_loss)

        print(
            f"Train Loss: {train_loss:.4f} | Valid Loss: {dev_loss:.4f} | TF Loss: {tf_loss:.4f} | "
            f"TF Ratio: {tf_ratio(epoch):.2f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_dev_epoch = epoch
            torch.save(model.state_dict(), best_checkpoint)
            print(f"Best model saved with loss: {best_dev_loss:.4f}")

    training_end = time.time()
    wall_clock_seconds = training_end - training_start
    wall_clock_str = str(timedelta(seconds=int(wall_clock_seconds)))

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), epoch_valid_losses, label="Valid Loss")
    plt.title("Training and Validation Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(FIGURES / "epochs_train_dev_loss.png")
    plt.close()

    plt.figure()
    plt.plot(range(1, len(lr_schedule) + 1), lr_schedule, label="LR Schedule")
    plt.title("Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.xticks([i * 1000 for i in range(0, 21)])
    plt.legend()
    plt.savefig(FIGURES / "learning_rate_schedule.png")
    plt.close()

    positional_encoding = model.input_layer.pos.pe[0, :100, :].detach().cpu().numpy()
    plt.figure(figsize=(16, 6))
    sns.heatmap(
        positional_encoding,
        cmap="viridis",
        xticklabels=32,
        yticklabels=10,
        cbar_kws={"label": "Encoding value"},
    )
    plt.title("Sinusoidal Positional Encoding (100 positions x 256 dimensions)")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig(FIGURES / "positional_encoding_heatmap.png", dpi=150)
    plt.close()

    with open(RESULTS / "data.md", "w", encoding="utf-8") as f:
        f.write("| | Train | Dev | Test |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| Pairs | {train_stats['pairs']} | {dev_stats['pairs']} | {test_stats['pairs']} |\n")
        f.write(f"| Mean / max source length (tokens) | {train_stats['mean_src']:.2f} / {train_stats['max_src']} | {dev_stats['mean_src']:.2f} / {dev_stats['max_src']} | {test_stats['mean_src']:.2f} / {test_stats['max_src']} |\n")
        f.write(f"| Mean / max target length (tokens) | {train_stats['mean_tgt']:.2f} / {train_stats['max_tgt']} | {dev_stats['mean_tgt']:.2f} / {dev_stats['max_tgt']} | {test_stats['mean_tgt']:.2f} / {test_stats['max_tgt']} |\n")
        f.write(f"| Pairs dropped as too long | {train_stats['dropped']} | - | - |\n")

    with open(RESULTS / "model_training.md", "w", encoding="utf-8") as f:
        f.write("| | |\n")
        f.write("|---|---|\n")
        f.write(f"| Trainable parameters | {num_parameters:,} |\n")
        f.write(f"| Epochs trained / best epoch | {num_epochs} / {best_dev_epoch} |\n")
        f.write(f"| Best dev loss | {best_dev_loss:.4f} |\n")
        f.write(f"| Training time and GPU | {wall_clock_str} on {gpu_name} |\n")