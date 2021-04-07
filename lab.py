import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from skimage.morphology import binary_opening, binary_closing
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.transform import warp, AffineTransform, rotate
from skimage.feature import canny

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams["figure.figsize"] = [10, 5]

def rotate_pic(img, center):
    h, theta, d = hough_line(canny(img_gray))
    average_angle = 0
    size = 0
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        if angle > angle_eps:
            average_angle += angle
            size += 1
            print('angle on iter:', np.degrees(angle))
    if average_angle > size * angle_eps:
        img = rotate(img, np.rad2deg(average_angle))
    print('avgangle: ' + str(average_angle))


def find_largest_components(mask, img, obj_name, ax):
    # ищем максимальную компоненту
    labels = label(mask)
    props = regionprops(labels)
    areas = [prop.area for prop in props]
    largest_comp_id = np.array(areas).argmax()
    # вычисляем размеры "коробки" вокруг аудиогида
    minr, minc, maxr, maxc = props[largest_comp_id].bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    if obj_name == 'box':
        ax.plot(bx, by, '-b', linewidth=1)
    else:
        ax.plot(bx, by, '-r', linewidth=1)
    
    # ширина прямоугольника вокруг аудиогида
    box_width = maxr - minr     
    
    # высота прямоугольника вокруг аудиогида
    box_height = maxc - minc      
    
    # возвращаем характерные размеры аудиогида (коробка в которую он помещается)
    return box_width, box_height


def find_box(img, ax):
    edge_map = binary_closing(canny(img_gray, sigma=1), selem=np.ones((9, 9)))
    areas = binary_fill_holes(edge_map)
    return find_largest_components(areas, img, 'box', ax)


def find_audioguide(img, ax):
    black_mask = img_gray < 0.25
    #audioguide_mask = binary_opening(black_mask, selem=np.ones((30, 30)))
    return find_largest_components(black_mask, img, 'audio', ax)


def eval_tests(img, img_gray):
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 6))
    w_box, h_box = find_box(img, ax)
    w_audio, h_audio = find_audioguide(img, ax)
    test_res = w_box > w_audio and h_box > h_audio
    ax.imshow(img)
    plt.show()
    print(f'Test:{i} result:{test_res}')
    if test_res is True:
        return True
    return False

angle_eps = np.radians(3) # угол при котором мы еще не поворачиваем картинку


path_to_true = 'dataSet/True/'
path_to_false = 'dataSet/False/'
size_of_neg = 27
size_of_pos = 22
right_neg = 0
right_pos = 0

for i in range(1, size_of_neg+1):
    if i < 10:
        pic_num_str = '0'+str(i)
    else:
        pic_num_str = str(i)
    path = path_to_false + pic_num_str + '.jpg'
    img = imread(path)
    img_gray = rgb2gray(img)
    res = eval_tests(img, img_gray)
    if not res:
        right_neg += 1

for i in range(1, size_of_pos+1):
    if i < 10:
        pic_num_str = '0'+str(i)
    else:
        pic_num_str = str(i)
    path = path_to_true + pic_num_str + '.jpg'
    img = imread(path)
    img_gray = rgb2gray(img)
    res = eval_tests(img, img_gray)
    if res:
        right_pos += 1

print(f'Right negatives: {right_neg}')
print(f'Right positives: {right_pos}')