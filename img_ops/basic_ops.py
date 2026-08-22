import cv2 as cv
import numpy as np


def gray_image(image: str) -> None:
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





def resize_image(image: str, width: int, height: int) -> None:
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




def crop_image(image: str, i_start: int, j_start: int, i_end: int, j_end: int) -> None:
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




def isolate_color(image: str, lower_hsv: np.ndarray, upper_hsv: np.ndarray) -> None:
    """Shows the image after isolating the region between the upper and lower bounds

    Args:
        image (str): name of the image that should be in the directory img_ops/images/    ----    (Example: image_name.png)
        lower_hsv (np.ndarray): lower bound of the region that will be isolated
        upper_hsv (np.ndarray): upper bound of the region that will be isolated
        
    Returns:
        None
    """

    img = cv.imread("images/"+image)
    if img is None:
        print("The image you entered is not found, please read the documentation of resize_image()")
        return

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)                #convert from BGR to HSV
    mask_image = cv.inRange(hsv_img, lower_hsv, upper_hsv)      #create the mask

    isolated_img = cv.bitwise_and(img, img, mask=mask_image)

    cv.imshow("mask", mask_image)
    cv.imshow("isolated", isolated_img)
    cv.waitKey(0)
    cv.destroyAllWindows()




def blur_image(image: str) -> None:
    """Shows the image after blur 
    
    Args:
        image (str): name of the image that should be in the directory img_ops/images/    ----    (Example: image_name.png)

    Returns:
        None
    """
    img = cv.imread("images/"+image)
    if img is None:
        print("The image you entered is not found, please read the documentation of resize_image()")
        return

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    normal_blur = cv.blur(img, (5, 5))
    gauss_blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)

    cv.imshow("original", img)
    cv.imshow("normal", normal_blur)
    cv.imshow("gaussian", gauss_blur)

    cv.waitKey(0)
    cv.destroyAllWindows()




def edges_image(image: str) -> None:
    """Shows the edges of the image 
    
    Args:
        image (str): name of the image that should be in the directory img_ops/images/    ----    (Example: image_name.png)

    Returns:
        None
    """
    img = cv.imread("images/"+image)
    if img is None:
        print("The image you entered is not found, please read the documentation of resize_image()")
        return

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    """normal_blur = cv.blur(img, (5, 5))
    gauss_blur = cv.GaussianBlur(img, (5, 5), sigmaX=0)"""

    edges = cv.Canny(img, threshold1=50, threshold2=150)



    cv.imshow("edges", edges)
    #cv.imshow("normal", normal_blur)
    #cv.imshow("gaussian", gauss_blur)

    cv.waitKey(0)
    cv.destroyAllWindows()

#lower_red = np.array([0, 50, 50])  
#upper_red = np.array([10, 255, 255])

edges_image("painting.png")