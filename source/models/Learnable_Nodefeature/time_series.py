import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, model_type='lstm', timeseries_embedding_type='last'):
        super(TimeSeriesEncoder, self).__init__()
        self.model_type = model_type
        self.embedding_type = timeseries_embedding_type

        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        elif model_type == 'cnn':
            self.cnn = nn.Sequential(
                nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            )
        else:
            raise ValueError("Invalid model type. Choose 'lstm' or 'gru'.")
        
        self.fc = nn.Linear(hidden_size, embedding_size)

    def forward(self, x):
        # print(f'x.shape={x.shape}')
        
        # x shape: (num_samples, num_nodes, time_series_length)
        num_samples, num_nodes, time_series_length = x.shape
        
        # Reshape x to: (num_samples * num_nodes, time_series_length, 1)
        x = x.view(num_samples * num_nodes, time_series_length, -1)
        
        # print(f'after view x.shape={x.shape}')

        if self.model_type in ['lstm', 'gru']:
            # RNN outputs
            output, hidden = self.rnn(x)
            
            if self.embedding_type == 'last':
                # Use the last hidden state
                if self.model_type == 'lstm':
                    time_series_emb = hidden[0][-1]  # LSTM returns (hidden, cell), we take the last hidden state
                else:
                    time_series_emb = hidden[-1]    # GRU returns only hidden state
            elif self.embedding_type == 'mean':
                # average of hidden embeddings of each state
                time_series_emb = output.mean(dim=1)
        elif self.model_type == 'cnn':
            # CNN outputs
            x = x.permute(0, 2, 1)  # Change shape to (num_samples * num_nodes, input_size, time_series_length)
            output = self.cnn(x)
            time_series_emb = output.squeeze(dim=2)  # Remove the last dimension

        # Pass the output of RNN or CNN to a fully connected layer to get the embeddings
        embeddings = self.fc(time_series_emb)
        
        # Reshape embeddings to: (num_samples, num_nodes, embedding_size)
        embeddings = embeddings.view(num_samples, num_nodes, -1)
        
        return embeddings
