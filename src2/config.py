class Config:
    def __init__(self):
        # Audio processing
        self.sample_rate       = 22050        # Sample rate für Audio
        self.n_mels            = 256          # Anzahl Mel-Bänder
        self.max_time_steps    = 512          # Max. Zeit-Frames (→ block_size)
        self.use_augmentation  = True         # Data-Augmentation an/aus

        # Training
        self.batch_size        = 4
        self.learning_rate     = 1e-4
        self.epochs            = 50
        self.dropout           = 0.1          # Dropout für Audio-GPT

        # Model Hyperparameters (AudioGPT)
        self.n_embd            = 256
        self.n_layer           = 2
        self.n_head            = 4
        self.block_size        = self.max_time_steps  # Sequenzlänge
        self.pad_token         = 0            # Padding-Index

        # Checkpointing
        self.checkpoint_interval = 5

        # --- NEU: Text-Conditioning ---
        self.vocab_size        = 10000        # Größe deines Text-Vokabulars
        self.n_text_embd       = 256          # Embedding-Dim für Text #128
        self.n_text_layer      = 1            # Anzahl Transformer-Encoder-Layer
        self.text_dropout      = 0.1          # Dropout im Text-Encoder
