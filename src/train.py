def train(model, loader, optimizer, criterion):
    model.train()

    for batch in loader:
        optimizer.zero_grad()

        logits = model(
            batch["image"],
            batch["input_ids"],
            batch["attention_mask"]
        )

        loss = criterion(logits, batch["label"])
        loss.backward()
        optimizer.step()