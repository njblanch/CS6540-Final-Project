



if __name__=="__main__":
    # Load model
    model.load_state_dict(torch.load('transformer_desync_model.pth'))
    model.eval()

    with torch.no_grad():
        total_loss = 0
        for audio_features, video_features, target in val_loader:  # Replace with your validation DataLoader
            outputs = model(audio_features, video_features)
            loss = criterion(outputs.view(-1), target.view(-1))
            total_loss += loss.item()

        avg_val_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
