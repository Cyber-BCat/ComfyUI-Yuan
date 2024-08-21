不需要复杂操作，直接用，
节点：black and white 生成灰度图片，
minus mask 通过mask抠出图片剩余部分,
Yuan transfer 实现Photoshop中正片叠底、硬光等图片叠加功能，可以使用这个节点保留原图图像特征，例如文字等。

Some simple&practical ComfyUI image processing nodes




B&W image generate: actural its a Gray level map generator, matches all types of input and output

Mask subtraction: using original image and B&W mask generates the segmented image, as example shows:
![Yuan nodes](https://github.com/user-attachments/assets/873565b0-5c3c-4e2a-96c7-ea03aa85f288)


Transfer like photoshot, such as 'hard light, multiply' and so on...

you can download the default workflow from 'examples' or from this url:
https://github.com/Cyber-Blacat/ComfyUI-Yuan/blob/main/examples/Yuan%20example%20workflow.json

by B_Cat
