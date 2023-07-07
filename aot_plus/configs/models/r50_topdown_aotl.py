from .default import DefaultModelConfig


class ModelConfig(DefaultModelConfig):
    def __init__(self):
        super().__init__()
        self.TOP_DOWN_FREEZE_ENCODER = False
        freeze_text = "_Freeze" if self.TOP_DOWN_FREEZE_ENCODER else ""
        self.VAR_LOSS_WEIGHT = 0.01
        var_loss_text = f"_var_{self.VAR_LOSS_WEIGHT}"
        self.MODEL_NAME = f'R50_TopDown{freeze_text}{var_loss_text}_AOTL'

        self.MODEL_ENCODER = f'resnet50_topdown{freeze_text}{var_loss_text}'
        self.MODEL_ENCODER_PRETRAIN = './pretrain_models/resnet50-0676ba61.pth'  # https://download.pytorch.org/models/resnet50-0676ba61.pth
        self.MODEL_ENCODER_DIM = [256, 512, 1024, 1024]  # 4x, 8x, 16x, 16x
        self.MODEL_LSTT_NUM = 3

        self.TRAIN_LONG_TERM_MEM_GAP = 2

        self.TEST_LONG_TERM_MEM_GAP = 5