import numpy as np
import torch
from skimage.color import rgb2lab
from skimage.util import img_as_ubyte
from PIL import Image
import matplotlib.pyplot as plt

class lightdarkjudgment:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "threshold": ("FLOAT", {"default": 32.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                "brightness1": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "contrast1": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "saturation1": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "brightness2": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "contrast2": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "saturation2": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("IMAGE", "brightness", "contrast", "saturation", "color diff",)
    FUNCTION = "lightordarkimage"
    CATEGORY = "Yuan😺"




    def calculate_color_difference_in_diff_area(self, img1, img2):
        """
        计算两张图像的色差，忽略透明区域

        参数:
        img1 (numpy array): 第一张图像
        img2 (numpy array): 第二张图像

        返回:
        float: 差异区域的平均色差
        """
        # 提取透明度通道
        alpha1 = img1[:, :, 3] if img1.shape[2] == 4 else np.ones(img1.shape[:2])
        alpha2 = img2[:, :, 3] if img2.shape[2] == 4 else np.ones(img2.shape[:2])
        
        # 创建遮罩，只保留不透明区域
        mask = (alpha1 > 0.1) & (alpha2 > 0.1)

        # 转换图像为Lab颜色空间
        img1_lab = rgb2lab(img1[:, :, :3])
        img2_lab = rgb2lab(img2[:, :, :3])

        # 计算每个像素的绝对差异
        diff_lab = np.abs(img1_lab - img2_lab)

        # 计算整体差异的掩码 (即差异区域的掩码)
        diff_mask = np.any(diff_lab > 1, axis=-1) & mask  # 选择阈值为1，阈值可以根据需要调整

        # 在掩码区域内计算色差
        diff_in_diff_area_lab = np.sqrt(np.sum(diff_lab ** 2, axis=-1))
        diff_in_diff_area_lab[~diff_mask] = 0  # 仅保留差异区域的色差
        
        # 如果是一个极小值，则设置为0值
        if np.isnan(np.mean(diff_in_diff_area_lab[diff_mask])):
            return 0.0
        else:
            return np.mean(diff_in_diff_area_lab[diff_mask])

    def lightordarkimage(self, image1, image2, threshold, brightness1, contrast1, saturation1, brightness2, contrast2, saturation2):
        """
        根据色差阈值判断并返回对应的图像及其属性

        参数:
        image1 (torch.Tensor): 第一张图像
        image2 (torch.Tensor): 第二张图像
        threshold (float): 色差阈值
        brightness1 (float): 第一张图像的亮度
        contrast1 (float): 第一张图像的对比度
        saturation1 (float): 第一张图像的饱和度
        brightness2 (float): 第二张图像的亮度
        contrast2 (float): 第二张图像的对比度
        saturation2 (float): 第二张图像的饱和度

        返回:
        tuple: 根据色差阈值返回的图像及其属性
        """
        B, H, W, C = image1.shape
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image1_tensor = image1.permute(0, 3, 1, 2).clone().to(device)
        image2_tensor = image2.permute(0, 3, 1, 2).clone().to(device)

        if image1.shape[1:] != image2.shape[1:]:
            image2_tensor = torch.nn.functional.interpolate(image2_tensor, size=(H, W), mode="bilinear")

        if image2.shape[0] < B:
            image2_tensor = image2_tensor[0].unsqueeze(0).repeat(B, 1, 1, 1)

        # 转换为numpy格式并计算色差
        image1_np = image1_tensor.permute(0, 2, 3, 1).cpu().numpy()[0]
        image2_np = image2_tensor.permute(0, 2, 3, 1).cpu().numpy()[0]

        # 计算色差
        color_diff = self.calculate_color_difference_in_diff_area(image1_np, image2_np)
        print(f"Color Difference in Difference Area: {color_diff}")
        

        # 根据色差阈值判断并返回对应的图像和属性
        if color_diff < threshold:
            tensor_out = image1_tensor
            brightness = brightness1
            contrast = contrast1
            saturation = saturation1
        else:
            tensor_out = image2_tensor
            brightness = brightness2
            contrast = contrast2
            saturation = saturation2

        tensor_out = torch.clamp(tensor_out, 0, 1)
        tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()

        return tensor_out, brightness, contrast, saturation, color_diff
    
    # A dictionary that contains all nodes you want to export with their names
    # NOTE: names should be globally unique,一行不需要逗号，多行需要逗号
    # 冒号后为函数名
NODE_CLASS_MAPPINGS = {
    "Light or Dark": lightdarkjudgment
}

    # A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Light or Dark": "Light or Dark"
    # After the colon is the name displayed on the node. Multiple lines correspond to multiple nodes
}
