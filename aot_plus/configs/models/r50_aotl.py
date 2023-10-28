from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.FORMER_MEM_LEN = 1
        self.LATTER_MEM_LEN = 7
        self.USE_MASK = False
        self.NO_LONG_MEMORY = False
        long_mem_text = "_No_long_mem" if self.NO_LONG_MEMORY else ""
        self.NO_MEMORY_GAP = False
        self.MODEL_ATT_HEADS = 2 if self.NO_MEMORY_GAP else self.MODEL_ATT_HEADS
        mem_gap_text = "_No_mem_gap" if self.NO_MEMORY_GAP else ""
        self.REVERSE_INFER = False
        self.REVERSE_LOSS = 0.4
        self.REVERSE_LOSS = self.REVERSE_LOSS / 4 if self.NO_MEMORY_GAP else self.REVERSE_LOSS
        reverse_infer_text = f"_Reverse_infer_detach_short_loss_{self.REVERSE_LOSS}" if self.REVERSE_INFER else ""
        self.MODEL_NAME = f'R50_AOTL{long_mem_text}{mem_gap_text}{reverse_infer_text}'

        self.MODEL_ENCODER = 'resnet50'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnet50-0676ba61.pth'  # https://download.pytorch.org/models/resnet50-0676ba61.pth
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5

        # Temporal Positional Embedding
        self.USE_TEMPORAL_POSITIONAL_EMBEDDING = False