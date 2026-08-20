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




def crop_image(image: str, i_start: int, j_start: int, i_end: int, j_end: int):
    """Shows the cropped version of your image
    Args:
        image (str): name of the image that should be in the directory img_ops/images/  ---- (Example: image_name.png)
        i_start (int): row from where the cropping starts
        j_start (int): column where the cropping starts
        i_end (int): row where the cropping ends
        j_end (int): column where the cropping ends
    
    Returns:
        None
    """
    img = cv.imread("images/"+image)
    if img is None:
        print("The image you entered is not found, please read the documentation of resize_image()")
        return
    cropped_img = img[i_start:i_end, j_start:j_end]
    cv.imshow('Cropped', cropped_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


crop_image("painting.png", 0, 200, 500, 500)