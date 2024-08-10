from PIL import Image, ImageOps
from io import BytesIO
import numpy as np
import struct
import comfy.utils
import time
import cv2
import numpy as np
import torch
import scipy.ndimage


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

    def apply_mask_exact(self,original_image, target_image, mask_image):



        if original_image.shape != target_image.shape:
            raise ValueError("原图和目标图的尺寸不一致")
        if mask_image.shape != original_image.shape[:2]:
            raise ValueError("mask图的尺寸与原图/目标图不一致")

        # 创建一个空白图像用于保存结果
        result_image = original_image.copy()
        



        # 创建反向mask
        inverse_mask = cv2.bitwise_not(mask_image)
        inverse_mask = inverse_mask.astype(np.uint8)

        # 保留原图的非mask部分
        original_background = cv2.bitwise_and(original_image, original_image, mask=inverse_mask)


        # 提取目标图的mask部分
        target_foreground = cv2.bitwise_and(target_image, target_image, mask=mask_image)

        # 合并两个部分
        result_image = cv2.add(original_background, target_foreground)
        return result_image
    
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

      

            

        if background_type == "black":

            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            img1_l = cv2.bitwise_and(tag, tag, mask=mask)


        if background_type == "white":
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

NODE_CLASS_MAPPINGS = {
    "ImageMinusMask": imageMinusMask
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageMinusMask": "Image Minus Mask"
    # 冒号后是节点上显示的名字,多行对应多个节点
}
