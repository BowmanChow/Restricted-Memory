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
        self.MODEL_ATT_HEADS = 1 if self.NO_MEMORY_GAP else self.MODEL_ATT_HEADS
        mem_gap_text = "_No_mem_gap" if self.NO_MEMORY_GAP else ""
        self.REVERSE_INFER = True
        reverse_infer_text = "_Reverse_infer" if self.REVERSE_INFER else ""
        self.MODEL_NAME = f'R50_AOTL{long_mem_text}{mem_gap_text}{reverse_infer_text}'

        self.MODEL_ENCODER = 'resnet50'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnet50-0676ba61.pth'  # https://download.pytorch.org/models/resnet50-0676ba61.pth
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5