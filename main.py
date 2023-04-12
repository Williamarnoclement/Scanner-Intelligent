#La librairie de traitement de l'image
import skimage
#Gestion IO, des filtres et des transformations
from skimage import io, filters, transform, data
from scipy.signal import fftconvolve, convolve2d
#Les librairies pour Hough
from skimage.draw import line
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
#Utilitaires de détection des coins
from skimage.feature import corner_moravec, corner_harris, corner_peaks
from skimage.feature import canny

#Les Utilitaires dédiés aux primitives mathématiques
import matplotlib.pyplot as plt
import numpy as np

#Outils de dessins
from skimage.draw import line


## BROUILLONS ##
#tform = transform.ProjectiveTransform()
#tform.estimate(rectangle, keypoints)
#corrected_text = transform.warp(text, tform, output_shape=(50, 300))
#plt.imshow(corrected_text, cmap="gray") and plt.show()

## POINTS ##
#Autre ticket
#326 1368
#2388 516
#705 2332
#2854 1501


fig = plt.figure()
rows = 2
columns = 2

poly_autre_ticket = np.array([ [705,2332], [2854, 1501], [2388, 516], [326, 1368]])

def generate_square():
    square = np.zeros([100, 100], dtype="int")
    square[20:80, 20:80] = 1
    return square

def get_image():
    my_image = io.imread("images/autre-ticket.jpeg")
    if (my_image.shape[2] == 1):
        return my_image
    if (my_image.shape[2] == 3):
        return skimage.color.rgb2gray(my_image)
    return skimage.color.rgb2gray(skimage.color.rgba2rgb(my_image))


## BINARISATION ##
def apply_otsu(image):
    otsu_threshold = filters.threshold_otsu(image)
    image_binarized = image > otsu_threshold
    return image_binarized

def apply_sauvola(image):
    sauvola_threshold = filters.threshold_sauvola(image)
    image_binarized = image > sauvola_threshold
    return image_binarized

current_image = get_image()

fig.add_subplot(rows, columns, 1)
plt.imshow(current_image)


## RECUPERATION DYNAMIQUE DES COINS VIA HARRIS ET CONVOLUTION 2D ##
def get_auto_corner_harris_convolve2d(image):
    moravec = corner_moravec(image)

    k = 10
    box_filter = np.ones((k, k)) * 1/k * 1/k
    image = convolve2d(image, box_filter, mode="same")

    harris = corner_harris(image)
    corners = corner_peaks(harris, threshold_rel=0.92)
    return corners

## RECUPERATION DYNAMIQUE DES COINS VIA HOUGH ##
def get_auto_lines(image):
    angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, d = hough_line(image, theta=angles)
    incr=0
    lines_tables = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        #plt.axline((x0, y0), slope=np.tan(angle + np.pi/2))
        lines_tables[incr] = (x0, y0)
        incr = incr + 1
    return lines_tables

current_image = canny(current_image, sigma=3, low_threshold=0.2, high_threshold=0.3)
#HoughLinesP from opencv
all_lines = probabilistic_hough_line(current_image, line_gap=3)

image_width = current_image.shape[0]
image_height= current_image.shape[1]

inbetween_img = np.zeros((image_width, image_height), dtype=np.uint8)

for (x1, y1), (x2, y2) in all_lines:
    rr, cc = line(x1, y1, x2, y2)
    inbetween_img[cc, rr] = 1

## AFFICHAGE INTERMEDIAIRE ##
fig.add_subplot(rows, columns, 2)
plt.imshow(inbetween_img)

## PROCESSUS DE TRANSFORMATION DE L'IMAGE A PARTIR DES COINS FOURNIS DANS LA VARIABLE "CORNERS" ##
corners = get_auto_corner_harris_convolve2d(inbetween_img)

for (x,y) in corners:
    plt.plot(x, y, marker="+", markersize=20)

height = 300
width = 150

final_rectangle = np.array([[0, 0], [0, height], [width, height], [width, 0]])

tform = transform.ProjectiveTransform()
tform.estimate(final_rectangle, corners)
corrected_image = transform.warp(current_image, tform, output_shape=(height, width))

fig.add_subplot(rows, columns, 4)
plt.imshow(corrected_image)

plt.show()
