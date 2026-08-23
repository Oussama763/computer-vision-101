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

    normal_blur = cv.blur(gray_img, (5, 5))
    gauss_blur = cv.GaussianBlur(gray_img, (5, 5), sigmaX=0)

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

    normal_blur = cv.blur(gray_img, (5, 5))
    gauss_blur = cv.GaussianBlur(gray_img, (5, 5), sigmaX=0)

    edges_original = cv.Canny(img, threshold1=50, threshold2=150)
    edges_normal = cv.Canny(normal_blur, threshold1=50, threshold2=150)
    edges_gaussian = cv.Canny(gauss_blur, threshold1=50, threshold2=150)

    kernel = np.ones((3, 3))
    dilated_gaussian = cv.dilate(edges_gaussian, kernel, iterations=1)

    #cv.imshow("edges_original", edges_original)
    #cv.imshow("edges_normal", edges_normal)
    cv.imshow("dilated_gaussian", dilated_gaussian)
    #cv.imshow("normal", normal_blur)
    #cv.imshow("gaussian", gauss_blur)

    cv.waitKey(0)
    cv.destroyAllWindows()



def detect_objects(image: str, min_area: int) -> None:
    """Shows the objects in the image
        
    Args:
        image (str): name of the image that should be in the directory img_ops/images/    ----    (Example: image_name.png)
        min_area (int): minimum area of the object to be detected

    Returns:
        None
    """
    img = cv.imread("images/"+image)
    if img is None:
        print("The image you entered is not found, please read the documentation of resize_image()")
        return

    output_img = img.copy()

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (5, 5), sigmaX=0)
    edges_img = cv.Canny(blur_img, threshold1=50, threshold2=150)
    kernel = np.ones((3, 3))
    dilated_img = cv.dilate(edges_img, kernel, iterations=1)

    contours, hierarchy = cv.findContours(dilated_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #print(contours[0])
    #print(hierarchy[0])

    count = 0

    for contour in contours:
        area = cv.contourArea(contour)

        if area >= min_area:
            count += 1
            x, y, w, h = cv.boundingRect(contour)      #draw rectangle around object
            cv.rectangle(output_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv.putText(output_img, "text", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    print(f"{count} objects")

    cv.imshow("dialated", dilated_img)
    cv.imshow("objects", output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()



def classify_shapes(image: str, min_area: int) -> None:
    """Shows the objects in the image
            
    Args:
        image (str): name of the image that should be in the directory img_ops/images/    ----    (Example: image_name.png)
        min_area (int): minimum area of the object to be detected

    Returns:
        None
    """
    img = cv.imread("images/"+image)
    if img is None:
        print("The image you entered is not found, please read the documentation of resize_image()")
        return

    output_img = img.copy()
    
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur_img = cv.GaussianBlur(gray_img, (5, 5), sigmaX=0)
    edges_img = cv.Canny(blur_img, threshold1=50, threshold2=150)
    kernel = np.ones((3, 3))
    dilated_img = cv.dilate(edges_img, kernel, iterations=1)

    contours, hierarchy = cv.findContours(dilated_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv.contourArea(contour)

        if area < min_area:
            continue

        perimeter = cv.arcLength(contour, closed=True)
        epsilon = 0.02 * perimeter
        approx_vertices = cv.approxPolyDP(contour, epsilon, closed=True)

        num_vertices: int = len(approx_vertices)
        shape_label: str = "Unknown"

        if num_vertices == 3:
            shape_label = "Triangle"
        elif num_vertices == 4:
            x, y, w, h = cv.boundingRect(approx_vertices)
            aspect_ratio: float = float(w) / h
            shape_label = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif num_vertices == 5:
            shape_label = "Pentagon"
        else:
            shape_label = "Circle"

        
        M = cv.moments(contour)
        if M["m00"] != 0:
            cx: int = int(M["m10"] / M["m00"])
            cy: int = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        cv.drawContours(output_img, [approx_vertices], -1, (255, 0, 0), 2)
        cv.circle(output_img, (cx, cy), 4, (0, 0, 255), -1)

        cv.putText(
            output_img,
            shape_label,
            (cx - 20, cy - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv.imshow("Shape Classification", output_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


    

#lower_red = np.array([0, 50, 50])
#upper_red = np.array([10, 255, 255])


classify_shapes("painting.png", 200)