### Training Functions




# Function to train the model using batches from a DataLoader
def train_model(model, train_loader, epochs=100, learning_rate=1e-3, temperature=0.5, alpha=0.5, mask_ratio=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    contrastive_loss_fn = NTXentLoss(temperature=temperature).to(device)
    reconstruction_loss_fn = torch.nn.MSELoss().to(device)

    model.train()
    loss_values = []  # List to store combined loss values

    for epoch in range(epochs):
        total_loss = 0
        for data in tqdm(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass: Get latent embeddings and reconstructed features
            latent_embeddings, reconstructed_x, _, _ = model(data)
            
            # Masked graph (student graph)
            masked_data, mask = mask_nodes(data, mask_ratio=mask_ratio)
            masked_latent_embeddings, masked_reconstructed_x, _, _ = model(masked_data)
            
            # Contrastive loss 
            z_teacher = latent_embeddings
            z_student = masked_latent_embeddings
            contrastive_loss = contrastive_loss_fn(z_teacher, z_student)
            
            # Reconstruction loss (between original and reconstructed features)
            reconstruction_loss = reconstruction_loss_fn(reconstructed_x, data.x)
            
            # Combine the losses
            loss = alpha * contrastive_loss + (1 - alpha) * reconstruction_loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        loss_values.append(avg_loss)

        print(f'Epoch {epoch}, Loss: {avg_loss}')

    return loss_values
