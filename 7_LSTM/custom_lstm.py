import torch
import torch.nn as nn

class AudioVideoLSTM(nn.Module):
    def __init__(self, audio_feature_dim=(60, 118), video_feature_dim=(60, 1024), lstm_hidden_dim=128, output_dim=1, num_layers=2, dropout=0.3):
        super(AudioVideoLSTM, self).__init__()

        self.audio_input_size = audio_feature_dim
        self.visual_input_size = video_feature_dim

        # Fully connected layers for the visual input before the LSTM
        self.visual_fc = nn.Sequential(
            nn.Linear(video_feature_dim[-1], 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # LSTM layer for audio
        self.audio_lstm = nn.LSTM(input_size=audio_feature_dim[-1], hidden_size=lstm_hidden_dim, 
                                  num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # LSTM layer for video
        self.video_lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_dim,
                                  num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # Fully connected layers for fusion
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 4, lstm_hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, output_dim)
        )

    def forward(self, audio_input, visual_input):
        # Process visual input through FC layers
        visual_input = self.visual_fc(visual_input)

        # LSTM for audio
        _, (audio_hidden, _) = self.audio_lstm(audio_input)
        audio_out = torch.cat((audio_hidden[-2], audio_hidden[-1]), dim=1)

        # LSTM for processed video
        _, (video_hidden, _) = self.video_lstm(visual_input)
        video_out = torch.cat((video_hidden[-2], video_hidden[-1]), dim=1)
        
        # Combine the outputs from both audio and video
        combined_features = torch.cat((audio_out, video_out), dim=1)

        # Final fully connected layers for prediction
        output = self.fc_layers(combined_features)

        return output
