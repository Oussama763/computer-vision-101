import cv2 as cv
#import numpy as np


def gray_image(image: str):
    """Shows the black and white version of your image
    Args:
        image (str): name of the image that should be in the directory img_ops/images/  ---- (Example: image_name.png)

    Returns:
        None
    """
    img_gray = cv.imread("images/"+image, cv.IMREAD_GRAYSCALE)
    if img_gray is None:
        print("The image you entered is not found, please read the documentation of gray_image()")
        return
    cv.imshow('Image', img_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()





def resize_image(image: str, width: int, height: int):
    """Shows the resized version of your image
    Args:
        image (str): name of the image that should be in the directory img_ops/images/  ---- (Example: image_name.png)
        width (int): new width of the image
        height (int): new height of the image
    
    Returns:
        None
    """
    img = cv.imread("images/"+image)
    if img is None:
        print("The image you entered is not found, please read the documentation of resize_image()")
        return
    resized_image = cv.resize(img, (width, height))
    cv.imshow('Image', resized_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


resize_image("painting.png", 500, 200)