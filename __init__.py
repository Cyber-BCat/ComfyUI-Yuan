from .function.LightorDark import lightdarkjudgment
from .yuan.Yuan import ImageJudgment, YuanBW, YuanTransfer, imageMinusMask


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    # "Yuan": Yuan_node,
    "YuanBW": YuanBW,
    "Light or Dark": lightdarkjudgment,
    "Yuan Transfer": YuanTransfer,
    "Image Judgment": ImageJudgment,
    "ImageMinusMask": imageMinusMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # "Yuan": "Yuan fake Node",
    "YuanBW": "Yuan Black and White Converter",
    "Light or Dark": "Light or Dark",
    "Yuan Transfer": "Yuan Transfer",
    "Image Judgment": "ImageJudgment",
    "ImageMinusMask": "Image Minus Mask"
    # 冒号后是节点上显示的名字,多行对应多个节点
    # After the colon is the name displayed on the node. Multiple lines correspond to multiple nodes
}

