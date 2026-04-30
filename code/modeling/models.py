import torch
import torch.nn as nn


class StateCentricLSTM_Delta(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, num_layers=2, embed_dim=32, use_projection=True
    ):
        super().__init__()
        self.use_projection = use_projection

        if self.use_projection:
            # Projection: High-Dim Sparse -> Low-Dim Dense
            self.encoder = nn.Linear(input_dim, embed_dim)
            lstm_input_size = embed_dim * 2
        else:
            # No Projection: Use raw input dimensions
            lstm_input_size = input_dim * 2

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Head: Projects back to Input Dim (Delta)
        # No LayerNorm, No ReLU at the end
        # We need to predict negative values (subtractions)
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(self, state_seq, goal_seq, lengths=None, hidden=None):
        """
        Returns:
            pred_delta: [B, T, Input_D] (The predicted CHANGE)
            hidden: LSTM hidden state
        """
        # 1. Encode (or pass through)
        if self.use_projection:
            state_emb = torch.relu(self.encoder(state_seq))
            goal_emb = torch.relu(self.encoder(goal_seq))
        else:
            state_emb = state_seq
            goal_emb = goal_seq

        _, T, _ = state_emb.shape

        # 2. Expand Goal
        goal_expanded = goal_emb.unsqueeze(1).expand(-1, T, -1)

        # 3. Concat
        lstm_input = torch.cat([state_emb, goal_expanded], dim=2)

        # 4. Pack
        if lengths is not None:
            lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # 5. LSTM
        out, hidden = self.lstm(lstm_input, hidden)

        # 6. Unpack
        if lengths is not None:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # 7. Predict Delta
        pred_delta = self.head(out)

        return pred_delta, hidden


class StateCentricLSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, num_layers=2, embed_dim=32, use_projection=True
    ):
        super().__init__()
        self.use_projection = use_projection

        if self.use_projection:
            # 1. Add a Projection Layer (Dimensionality Reduction)
            self.encoder = nn.Linear(input_dim, embed_dim)
            lstm_input_size = embed_dim * 2
        else:
            lstm_input_size = input_dim * 2

        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        # Head projects back to FULL input_dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, state_seq, goal_seq, lengths=None, hidden=None):
        """
        state_seq: [B, T, Input_D]
        goal_seq: [B, D] (will be expanded to [B, T, D])
        """

        # 1. Project High-Dim Sparse -> Low-Dim Dense (Optional)
        if self.use_projection:
            state_emb = torch.relu(self.encoder(state_seq))  # [B, T, embed_dim]
            goal_emb = torch.relu(self.encoder(goal_seq))  # [B, embed_dim]
        else:
            state_emb = state_seq
            goal_emb = goal_seq

        _, T, _ = state_emb.shape

        # Expand goal to match sequence length
        goal_expanded = goal_emb.unsqueeze(1).expand(-1, T, -1)

        # Concatenate: [B, T, 2D]
        lstm_input = torch.cat([state_emb, goal_expanded], dim=2)

        # Pack sequences if lengths are provided
        if lengths is not None:
            lstm_input = torch.nn.utils.rnn.pack_padded_sequence(
                lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        # LSTM Forward
        out, hidden = self.lstm(lstm_input, hidden)

        if lengths is not None:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # Project to next state
        pred_next_state = self.head(out)

        return pred_next_state, hidden
