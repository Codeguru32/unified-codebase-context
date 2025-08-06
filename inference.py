import os
import torch
import torch.nn as nn

# --- Placeholder Model Definition ---
# IMPORTANT: This is a hypothetical model structure. You will likely need to
# replace this with the actual model class definition from the repohyper
# project to ensure the state dictionary loads correctly. The layer names,
# sizes, and architecture must match the saved file.
class HypotheticalRNN(nn.Module):
    def __init__(self, vocab_size=30000, embedding_dim=128, hidden_size=256, n_layers=2, output_size=1):
        super(HypotheticalRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True # This is a common convention
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)
        # x shape: (batch_size, seq_length, embedding_dim)

        # h_0 (initial hidden state) is initialized to zeros by default
        out, _ = self.rnn(x)
        # out shape: (batch_size, seq_length, hidden_size)

        # We'll take the output from the last time step for classification/regression
        out = self.fc(out[:, -1, :])
        # out shape: (batch_size, output_size)
        return out

def load_trained_model(model_path, model_class, **kwargs):
    """Checks for, loads, and prepares a pre-trained model."""
    if not os.path.exists(model_path):
        print(f"Model file not found at '{model_path}'.")
        print("Please run 'python download_model.py' first.")
        return None, None

    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Instantiate the model with placeholder parameters
    model = model_class(**kwargs)

    try:
        # Load the saved state dictionary.
        # map_location ensures the model loads correctly even if trained on a different device (e.g., GPU -> CPU)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set the model to evaluation mode (disables dropout, etc.)
        print("Model loaded successfully and set to evaluation mode.")
        return model, device
    except RuntimeError as e:
        print("\n--- ERROR ---")
        print(f"Failed to load state_dict: {e}")
        print("This likely means the 'HypotheticalRNN' class does not match the architecture in 'model_10.pt'.")
        print("Please replace it with the correct model definition.")
        print("-------------\n")
        return None, None

if __name__ == "__main__":
    MODEL_PATH = os.path.join("models", "model_10.pt")
    
    # These parameters are guesses and must match the actual trained model.
    model_params = {'vocab_size': 30000, 'embedding_dim': 128, 'hidden_size': 256, 'output_size': 1}
    model, device = load_trained_model(MODEL_PATH, HypotheticalRNN, **model_params)

    if model:
        # --- Example Inference ---
        # This is a dummy input. The actual input should be tokenized code.
        print("\nRunning a dummy inference example...")
        # Create a dummy input tensor: (batch_size=1, sequence_length=50)
        dummy_input = torch.randint(0, model_params['vocab_size'], (1, 50), dtype=torch.long).to(device)
        
        with torch.no_grad(): # Disable gradient calculation for efficiency
            output = model(dummy_input)
        
        print(f"Dummy input shape: {dummy_input.shape}")
        print(f"Model output: {output}")
        print(f"Model output shape: {output.shape}")