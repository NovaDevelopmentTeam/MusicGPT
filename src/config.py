# nova_music2/config.py

"""
Comment lines out you don't need such as the Low-End GPU- or CPU-Version if you have better hardware.
To comment out fastly, you can mark a whole Config Version and press CTRL or CMD and Hashtag on your keyboard.
Very recommended for speeding up modifications of the nova song creator (NSC)
"""

# Low-End GPU- or CPU-Version (e.g. NVIDIA GT 710 or 12th Gen Intel(R) Core(TM) i5-1235U, 1300 MHz, 10 Core(s), 12 logical(r) Proccessor(s))
class Config:
    def __init__(self):
        # Model Hyperparameters
        self.input_vocab_size = 256     # Weniger Mels → geringerer Input
        self.n_embd = 256              # Kleinere Embedding-Dimension
        self.n_layer = 2               # Weniger Transformer-Blöcke
        self.n_head = 4                # Weniger Attention-Heads (128 / 4 = 32 pro Head)
        self.block_size = 1024          # Kürzere Sequenzen
        self.dropout = 0.1             # Kann gleich bleiben
        self.n_codes_total = 1         # Belassen, wenn es 1 Codebuch ist
        
        # Training Hyperparameters
        self.batch_size = 16           # Batch-Größe für das Training
        self.learning_rate = 1e-4      # Lernrate für den Optimierer
        self.epochs = 50               # Anzahl der Epochen
        self.optimizer = "adam"        # Optimierer (z.B. Adam oder SGD)
        self.weight_decay = 1e-5       # Gewichtsabnahme für den Optimierer
        self.early_stopping_patience = 5  # Anzahl der Epochen ohne Verbesserung für Early Stopping
        
        # Augmentation Hyperparameters
        self.pitch_shift_range = (-2, 2)     # Bereich für das Pitch Shifting
        self.time_stretch_range = (0.8, 1.2)  # Bereich für Time Stretching
        self.noise_stddev = 0.01              # Standardabweichung des Rauschens, das hinzugefügt wird

# Medium-End GPU-Version (e.g. NVIDIA GeForce GTX 1660 or higher)
# class Config:
#     def __init__(self):
#         # Model Hyperparameters
#         self.input_vocab_size = 96     # Musik-Codebook (z.B. EnCodec, VQ etc.)
#         self.n_embd = 384               # Embedding-Dimension
#         self.n_layer = 6                # Transformer-Blöcke
#         self.n_head = 6                 # 384 / 6 = 64 Dimensionen/Head
#         self.block_size = 1024
#         self.dropout = 0.1
#         self.n_codes_total = 1

# High-End GPU-Version (e.g. NVIDIA GeForce RTX 3060 or higher)
# class Config:
#     def __init__(self):
#         # Model Hyperparameters
#         self.input_vocab_size = 128 # or higher
#         self.n_embd = 256 # or higher
#         self.n_layer = 4 # or higher
#         self.n_head = 8 # or higher
#         self.block_size = 1024 # or higher, like 2028 or more
#         self.dropout = 0.1 # or higher
#         self.n_codes_total = 1 # or higher
