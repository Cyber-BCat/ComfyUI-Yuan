import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import folder_paths
import os
import types
import torch.nn.functional as F
import model_management
import logging
import struct
import comfy.utils
import time
import scipy.ndimage


from PIL import Image, ImageOps
from io import BytesIO
from skimage.color import rgb2lab







class Yuan_node:
    """
    A example node

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Tell the main program input parameters of nodes.
    IS_CHANGED:
        optional method to control when the node is re executed.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tulple.
    RETURN_NAMES (`tuple`):
        Optional: The name of each output in the output tulple.
    FUNCTION (`str`):
        The name of the entry-point method. For example, if `FUNCTION = "execute"` then it will run Example().execute()
    OUTPUT_NODE ([`bool`]):
        If this node is an output node that outputs a result/image from the graph. The SaveImage node is an example.
        The backend iterates on these output nodes and tries to execute all their parents if their parent graph is properly connected.
        Assumed to be False if not present.
    CATEGORY (`str`):
        The category the node should appear in the UI.
    execute(s) -> tuple || None:
        The entry point method. The name of this method must be the same as the value of property `FUNCTION`.
        For example, if `FUNCTION = "execute"` then this method's name must be `execute`, if `FUNCTION = "foo"` then it must be `foo`.
    """
    def __init__(self):
        pass
    # 初始化-不用管init
    
    @classmethod
    def INPUT_TYPES(s):
        # 输入的内容input contents
        """
            Return a dictionary which contains config for all input fields.
            Some types (string): "MODEL", "VAE", "CLIP", "CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "FLOAT".
            Input types "INT", "STRING" or "FLOAT" are special values for fields on the node.
            The type can be a list for selection.

            Returns: `dict`:
                - Key input_fields_group (`string`): Can be either required, hidden or optional. A node class must have property `required`
                - Value input_fields (`dict`): Contains input fields config:
                    * Key field_name (`string`): Name of a entry-point method's argument
                    * Value field_config (`tuple`):
                        + First value is a string indicate the type of field or a list for selection.
                        + Secound value is a config for type "INT", "STRING" or "FLOAT".
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "int_field": ("INT", {
                    "default": 0, 
                    "min": 0, #Minimum value
                    "max": 4096, #Maximum value
                    "step": 64, #Slider's step
                    "display": "number" # Cosmetic only: display as "number" or "slider"
                }),
                "float_field": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001, #The value represeting the precision to round to, will be set to the step value by default. 
                    # Can be set to False to disable rounding.
                    "display": "number"}),
                "print_to_screen": (["enable", "disable"],), # 是否开启滑块的选项
                "string_field": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node字符串-可以作为文本输入，
                    # false则只能输入一行提示词，true可以输入多行
                    "default": "Hello World!"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "test"

    #OUTPUT_NODE = False

    CATEGORY = "Yuan😺"
    # 归类目录名，即右键目录

    def test(self, image, string_field, int_field, float_field, print_to_screen):
        if print_to_screen == "enable":
            print(f"""Your input contains:
                string_field aka input text: {string_field}
                int_field: {int_field}
                float_field: {float_field}
            """)
        #do some processing on the image, in this example I just invert it
        image = 1.0 - image
        return (image,)

    """
        The node will always be re executed if any of the inputs change but
        this method can be used to force the node to execute again even when the inputs don't change.
        You can make this node return a number or a string. This value will be compared to the one returned the last time the node was
        executed, if it is different the node will be executed again.
        This method is used in the core repo for the LoadImage node where they return the image hash as a string, if the image hash
        changes between executions the LoadImage node is executed again.
    """
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"


class YuanBW:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_to_bw"
    CATEGORY = "Yuan😺"

    def convert_to_bw(self, image):
        # 确保图像是Tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor.")

        # 获取图像的批次大小、高度、宽度和通道数
        B, H, W, C = image.shape

        # 检查通道数是否为1，2或3
        if C not in [1, 2, 3]:
            raise ValueError("Input image must have 1, 2, or 3 channels.")

        # 转换为灰度图像
        if C == 3:  # 如果图像是RGB格式
            grayscale_image = (
                0.299 * image[..., 0] +  # 红色通道
                0.587 * image[..., 1] +  # 绿色通道
                0.114 * image[..., 2]   # 蓝色通道
            )
        elif C == 2:  # 如果图像是两个通道，这里我们简单地取平均
            grayscale_image = (image[..., 0] + image[..., 1]) / 2
        elif C == 1:  # 如果图像已经是灰度图，直接使用
            grayscale_image = image[..., 0]  # 去掉最后一维

        # 增加一个通道维度
        grayscale_image = grayscale_image.unsqueeze(-1)

        # 将灰度图像扩展为3个通道
        grayscale_image_3ch = grayscale_image.expand(B, H, W, 3)

        return (grayscale_image_3ch,)



class YuanTransfer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target": ("IMAGE", ),
                "source": ("IMAGE", ),
                "mode": ([
                    "add",
                    "multiply",
                    "screen",
                    "overlay",
                    "soft_light",
                    "hard_light",
                    "color_dodge",
                    "color_burn",
                    "difference",
                    "exclusion",
                    "divide",
                    
                    ], 
                    {"default": "add"}
                    ),
                "blur_sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0, "step": 0.01}),
                "blend_factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.001,  "round": 0.001}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "Yuan😺"

    def adjust_mask(self, mask, target_tensor):
        # Add a channel dimension and repeat to match the channel number of the target tensor
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)  # Add a channel dimension
            target_channels = target_tensor.shape[1]
            mask = mask.expand(-1, target_channels, -1, -1)  # Expand the channel dimension to match the target tensor's channels
    
        return mask


    def process(self, target, source, mode, blur_sigma, blend_factor, mask=None):
        B, H, W, C = target.shape
        device = model_management.get_torch_device()
        target_tensor = target.permute(0, 3, 1, 2).clone().to(device)
        source_tensor = source.permute(0, 3, 1, 2).clone().to(device)

        if target.shape[1:] != source.shape[1:]:
            source_tensor = comfy.utils.common_upscale(source_tensor, W, H, "bilinear", "disabled")

        if source.shape[0] < B:
            source = source[0].unsqueeze(0).repeat(B, 1, 1, 1)

        kernel_size = int(6 * int(blur_sigma) + 1)

        gaussian_blur = transforms.GaussianBlur(kernel_size=(kernel_size, kernel_size), sigma=(blur_sigma, blur_sigma))

        blurred_target = gaussian_blur(target_tensor)
        blurred_source = gaussian_blur(source_tensor)
        
        if mode == "add":
            tensor_out = (source_tensor - blurred_source) + blurred_target
        elif mode == "multiply":
            tensor_out = source_tensor * blurred_target
        elif mode == "screen":
            tensor_out = 1 - (1 - source_tensor) * (1 - blurred_target)
        elif mode == "overlay":
            tensor_out = torch.where(blurred_target < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
        elif mode == "soft_light":
            tensor_out = (1 - 2 * blurred_target) * source_tensor**2 + 2 * blurred_target * source_tensor
        elif mode == "hard_light":
            tensor_out = torch.where(source_tensor < 0.5, 2 * source_tensor * blurred_target, 1 - 2 * (1 - source_tensor) * (1 - blurred_target))
        elif mode == "difference":
            tensor_out = torch.abs(blurred_target - source_tensor)
        elif mode == "exclusion":
            tensor_out = 0.5 - 2 * (blurred_target - 0.5) * (source_tensor - 0.5)
        elif mode == "color_dodge":
            tensor_out = blurred_target / (1 - source_tensor)
        elif mode == "color_burn":
            tensor_out = 1 - (1 - blurred_target) / source_tensor
        elif mode == "divide":
            tensor_out = (source_tensor / blurred_source) * blurred_target
        else:
            tensor_out = source_tensor
        
        tensor_out = torch.lerp(target_tensor, tensor_out, blend_factor)
        if mask is not None:
            # Call the function and pass in mask and target_tensor
            mask = self.adjust_mask(mask, target_tensor)
            mask = mask.to(device)
            tensor_out = torch.lerp(target_tensor, tensor_out, mask)
        tensor_out = torch.clamp(tensor_out, 0, 1)
        tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()
        return (tensor_out,)


class ImageJudgment:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "threshold": ("FLOAT", {"default": 32.0, "min": 0.1, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "judge_images"
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

        if np.isnan(np.mean(diff_in_diff_area_lab[diff_mask])):
            return 0.0
        else:
            return np.mean(diff_in_diff_area_lab[diff_mask])

    def judge_images(self, image1, image2, threshold):
        """
        根据色差阈值判断并返回对应的图像

        参数:
        image1 (torch.Tensor): 第一张图像
        image2 (torch.Tensor): 第二张图像
        threshold (float): 色差阈值

        返回:
        torch.Tensor: 根据色差阈值返回的图像
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

        # 根据色差阈值判断并返回对应的图像,差异值小于threshold则输出图1
        if color_diff < threshold:
            tensor_out = image1_tensor
        else:
            tensor_out = image2_tensor

        tensor_out = torch.clamp(tensor_out, 0, 1)
        tensor_out = tensor_out.permute(0, 2, 3, 1).cpu().float()

        return (tensor_out,)




#You can use this node to save full size images through the websocket, the
#images will be sent in exactly the same format as the image previews: as
#binary images on the websocket with a 8 byte header indicating the type
#of binary message (first 4 bytes) and the image format (next 4 bytes).

#Note that no metadata will be put in the images saved with this node.

class imageMinusMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE", ),
                     "mask": ("IMAGE", ),                    
                     "background_type": (["PNG", "BLACK","WHITE"],)

                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "imageMinusMask"

    OUTPUT_NODE = True

    CATEGORY = "Yuan😺"

    
    def imageMinusMask(self,image,mask,background_type):



        tag = image.numpy()
        tag = tag[0].transpose(0, 1, 2)
        tag= (tag * 255).astype(np.uint8)

        mask = mask.numpy()
        mask = mask[0].transpose(0, 1, 2)
        mask= (mask * 255).astype(np.uint8)



        if background_type == "PNG":

            if (mask.shape[2]) == 4:
                mask = mask[..., :3]
            print(mask.shape)
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            print(mask.shape)

            tag = cv2.cvtColor(tag, cv2.COLOR_RGB2RGBA)
            print(tag.shape)


            img_masked = cv2.bitwise_and(tag, tag, mask=mask)

            # 将OpenCV图像转换为PIL图像
            pil_image = Image.fromarray(img_masked)

            # 将PIL图像的白色背景转换为透明
            # 创建一个数组来存储最终的像素数据
            datas = pil_image.getdata()
            new_data = []
            for item in datas:
                # 检查RGB值是否为白色
                if item[:3] == (255, 255, 255):
                    # 将白色背景转换为完全透明
                    new_data.append((255, 255, 255, 0))
                else:
                    new_data.append(item)
            
            # 使用新的像素数据创建一个带有透明通道的PIL图像
            pil_image = Image.new("RGBA", pil_image.size)
            pil_image.putdata(new_data)
            img1_l = pil_image

      

            

        if background_type == "BLACK":

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            img1_l = cv2.bitwise_and(tag, tag, mask=mask)


        if background_type == "WHITE":
            black_bg = np.uint8(tag*(mask/255.))
            
            # 第二步：将掩码原本0的位置改为255，原本255的位置改为0
            reversed_msk = 255-mask
            
            # 第三步：将黑色背景位置(像素值为0的位置) 加上255
            white_bg = (black_bg + reversed_msk).astype(np.uint8)
            img1_l = white_bg
                



        image_rgb = img1_l
        image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
        image_rgb = torch.from_numpy(image_rgb)[None,]

        
       

        return image_rgb,




# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique,一行不需要逗号，多行需要逗号
# 冒号后为函数名
NODE_CLASS_MAPPINGS = {
    "Yuan": Yuan_node,
    "YuanBW": YuanBW,
    "Yuan Transfer": YuanTransfer,
    "Image Judgment": ImageJudgment,
    "ImageMinusMask": imageMinusMask
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Yuan": "Yuan fake Node",
    "YuanBW": "Yuan Black and White Converter",
    "Yuan Transfer": "Yuan Transfer",
    "Image Judgment": "ImageJudgment",
    "ImageMinusMask": "Image Minus Mask"
    # 冒号后是节点上显示的名字,多行对应多个节点
}
