# CSC420 A4
# Author: Victor Wu
#

#####
# Need to clean up this file, imports, 
# make it work for generic images
# rework the folder structure
# maybe use updated tools?
# get rid of the pseudocode for the robot soccer ball thing
#####


# Imports
import numpy as np
import math
import cv2
import ast
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import gaussian_laplace
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt


# Relevant to Q2f
def show_descriptions(filename):

    # filename passed in is "004945"
    image = read_image("data/test/left/" + filename + ".jpg")
    if image is None:
        return -1

    depth_image = read_image("data/test/results/" + filename + "_depth.png", 0)
    if depth_image is None:
        return -1

    [f, px, py, baseline] = camera_params(filename)
    label_map = {1: "person", 2: "bicycle", 3: "car", 10: "traffic light"}

    f3_dboxes = open("data/test/results/" + filename + "_detection_boxes.txt", "r")
    f3_dclasses = open("data/test/results/" + filename + "_detection_classes.txt", "r")
    f3_dcoms = open("data/test/results/" + filename + "_detection_CoMs.txt", "r")
    dboxes_lines = f3_dboxes.readlines()
    dclasses_lines = f3_dclasses.readlines()
    dcoms_lines = f3_dcoms.readlines()
    f3_dboxes.close()
    f3_dclasses.close()
    f3_dcoms.close()

    # Format the data from strings into numbers
    dboxes = ast.literal_eval(dboxes_lines[0]) # [xleft, ytop, xright, ybottom]
    dclasses =  ast.literal_eval(dclasses_lines[0])
    dcoms =  ast.literal_eval(dcoms_lines[0])

    watch_out = []

    for i in range(len(dcoms)):
        top, left, bottom, right = dboxes[i]
        left, right = int(left * image.shape[1]), int(right * image.shape[1])
        top, bottom = int(top * image.shape[0]), int(bottom * image.shape[0])
        com_y, com_x, total_weight = dcoms[i]
        
        # Coordinates of detected object relative to your camera/car
        x_world = (com_x - px) * depth_image[com_y, com_x] / f
        y_world = (com_y - py) * depth_image[com_y, com_x] / f
        z_world = depth_image[com_y, com_x]

        diag_dist = np.linalg.norm((x_world, y_world, z_world))
        left_right = "left" if x_world < 0 else "right"
        obj = label_map[dclasses[i]]

        st = "There's a {} approximately {} meters away from you to the {}.".format(obj, diag_dist, left_right)
        watch_out.append((diag_dist, st))

    print("In image {}--------------------------------------------".format(filename))
    print("There are {} people in your field of view.".format(dclasses.count(1)))
    print("There are {} bicycles in your field of view.".format(dclasses.count(2)))
    print("There are {} cars in your field of view.".format(dclasses.count(3)))
    print("There are {} traffic lights in your field of view.".format(dclasses.count(10)))

    watch_out.sort(key=lambda tup: tup[0])
    for i in range(len(watch_out)):
        if i == 0:
            print(watch_out[i][1].upper() + "!!!!!")
        else:
            print(watch_out[i][1])

    return 1


# Relevant to Q2e
def segment(filename):

    # filename passed in is "004945"
    image = read_image("data/test/left/" + filename + ".jpg")
    if image is None:
        return -1

    depth_image = read_image("data/test/results/" + filename + "_depth.png", 0)
    if depth_image is None:
        return -1

    f3_dboxes = open("data/test/results/" + filename + "_detection_boxes.txt", "r")
    f3_dcoms = open("data/test/results/" + filename + "_detection_CoMs.txt", "r")
    dboxes_lines = f3_dboxes.readlines()
    dcoms_lines = f3_dcoms.readlines()
    f3_dboxes.close()
    f3_dcoms.close()
    

    # Format the data from strings into numbers
    dboxes = ast.literal_eval(dboxes_lines[0]) # [xleft, ytop, xright, ybottom]  might have to unnormalize these again
    dcoms = ast.literal_eval(dcoms_lines[0])

    seg_image = np.zeros(depth_image.shape)
    for i in range(len(dboxes)):
        top, left, bottom, right = dboxes[i]
        left, right = int(left * image.shape[1]), int(right * image.shape[1])
        top, bottom = int(top * image.shape[0]), int(bottom * image.shape[0])
        com_y, com_x, total_weight = dcoms[i]
        depth_at_CoM = depth_image[com_y, com_x]
        for y in range(top, bottom):
            for x in range(left, right):
                depth = depth_image[y, x]
                if (abs(depth - depth_at_CoM) <= 3):
                    if i < 15:
                        seg_image[y, x] = 255 - i * 10
                    else:
                        seg_image[y, x] = 255 - (i % 15) * 10
    write_image(seg_image, "data/test/results/" + filename + "_segmented.png")

    return 1


# Relevant to Q2d
def get_center_of_mass(depth_image, top, bottom, left, right):

    weights = np.array([0, 0])
    total_weight = 0
    for y in range(top, bottom):
        for x in range(left, right):
            weights += np.array([y, x]) * depth_image[y, x]
            total_weight += depth_image[y, x]
    
    center_of_mass = weights / total_weight
    # print(int(center_of_mass[0]), int(center_of_mass[1]))

    return (int(center_of_mass[0]), int(center_of_mass[1]), total_weight)


# Relevant to Q2d
def get_3D_locations(filename):

    # filename passed in is "004945"
    image = read_image("data/test/left/" + filename + ".jpg")
    if image is None:
        return -1

    boundary_image = read_image("data/test/results/" + filename + "_2c.png")
    if image is None:
        return -1

    depth_image = read_image("data/test/results/" + filename + "_depth.png", 0)
    if depth_image is None:
        return -1

    f3_dboxes = open("data/test/results/" + filename + "_detection_boxes.txt", "r")
    f3_dclasses = open("data/test/results/" + filename + "_detection_classes.txt", "r")
    f3_dscores = open("data/test/results/" + filename + "_detection_scores.txt", "r")
    dboxes_lines = f3_dboxes.readlines()
    dclasses_lines = f3_dclasses.readlines()
    dscores_lines = f3_dscores.readlines()
    f3_dboxes.close()
    f3_dclasses.close()
    f3_dscores.close()

    # Format the data from strings into numbers
    dboxes = []
    dclasses = []
    dscores = []
    dboxes = ast.literal_eval(dboxes_lines[0]) # [xleft, ytop, xright, ybottom]  might have to unnormalize these again
    dclasses =  ast.literal_eval(dclasses_lines[0])
    dscores = ast.literal_eval(dscores_lines[0])

    dcoms = []
    for i in range(len(dboxes)):
        # its actually [Y, X, Y, X] not [X, Y, X, Y]
        top, left, bottom, right = dboxes[i]
        left, right = int(left * image.shape[1]), int(right * image.shape[1])
        top, bottom = int(top * image.shape[0]), int(bottom * image.shape[0])
        (y, x, total_weight) = get_center_of_mass(depth_image, top, bottom, left, right)
        dcoms.append((y, x, total_weight))
        # cv2.circle(boundary_image, (x,y), 5, (0, 255, 0), -1)
    # cv2.imshow("CoMs", boundary_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    file = open("data/test/results/" + filename + "_detection_CoMs.txt", "w")
    file.write(str(dcoms))
    file.close()

    return 1


# Relevant to Q2c
def visualize_detections(filename):

    # filename passed in is "004945"
    image = read_image("data/test/left/" + filename + ".jpg")
    if image is None:
        return -1
    
    f3_dboxes = open("data/test/results/" + filename + "_detection_boxes.txt", "r")
    f3_dclasses = open("data/test/results/" + filename + "_detection_classes.txt", "r")
    f3_dscores = open("data/test/results/" + filename + "_detection_scores.txt", "r")
    dboxes_lines = f3_dboxes.readlines()
    dclasses_lines = f3_dclasses.readlines()
    dscores_lines = f3_dscores.readlines()
    f3_dboxes.close()
    f3_dclasses.close()
    f3_dscores.close()

    # Mark the car detections with red, person with blue, cyclist with green and traffic light with cyan rectangles
    # cv2.rectangle(image, top_left, bottom_right, red, 4)

    # Format the data from strings into numbers
    dboxes = []
    dclasses = []
    dscores = []
    dboxes = ast.literal_eval(dboxes_lines[0]) # [xleft, ytop, xright, ybottom]
    dclasses =  ast.literal_eval(dclasses_lines[0])
    dscores = ast.literal_eval(dscores_lines[0])
    # print(type(dboxes[0][0]))
    # print(dclasses)
    # WHY IS R AND B SWAPPING VALUES
    colour_map = {1: (0,0,255), 2: (0,255,0), 3:(255,0,0), 10: (0,255,255)}
    label_map = {1: "person", 2: "bicycle", 3: "car", 10: "traffic light"}
    for i in range(len(dclasses)):
        # print(type(dboxes[0]))
        # its actually [Y, X, Y, X] not [X, Y, X, Y]
        top_left = (int(dboxes[i][1] * image.shape[1]), int(dboxes[i][0] * image.shape[0]))
        below_top_left = (int(dboxes[i][1] * image.shape[1]) + 5, int(dboxes[i][0] * image.shape[0]) + 15)
        bottom_right = (int(dboxes[i][3] * image.shape[1]), int(dboxes[i][2] * image.shape[0]))
        cv2.rectangle(image, top_left, bottom_right, colour_map[dclasses[i]], 4)
        cv2.putText(image, label_map[dclasses[i]], below_top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_map[dclasses[i]], 2)
        # print(i)

    write_image(image, "data/test/results/" + filename + "_2c.png")
    # print(image.shape)

    return 1



# Relevant to Q2b
def store_strong_bounds(outfile, infile):

    threshold = 0.33
    # outfile passed in is "004945"
    # infile passed in is "image3"
    f3_dboxes = open("data/test/results/" + infile + "_detection_boxes.txt", "r")
    f3_dclasses = open("data/test/results/" + infile + "_detection_classes.txt", "r")
    f3_dscores = open("data/test/results/" + infile + "_detection_scores.txt", "r")
    dboxes_lines = f3_dboxes.readlines()
    dclasses_lines = f3_dclasses.readlines()
    dscores_lines = f3_dscores.readlines()
    f3_dboxes.close()
    f3_dclasses.close()
    f3_dscores.close()

    dboxes = []
    dclasses = []
    dscores = []

    # Format the data from strings into numbers
    dboxes_lines[0] = dboxes_lines[0][1:]
    dboxes_lines[len(dboxes_lines) - 1] = dboxes_lines[len(dboxes_lines) - 1][:-1]
    for l in dboxes_lines:
        l = l.strip()[1:-1].split()
        dbox = []
        for i in l:
            dbox.append(float(i))
        dboxes.append(dbox)
    # print(len(dboxes[0]))

    dclasses_lines[0] = dclasses_lines[0][1:]
    dclasses_lines[len(dclasses_lines) - 1] = dclasses_lines[len(dclasses_lines) - 1][:-1]
    for l in dclasses_lines:
        l = l.split()
        for i in l:
            dclasses.append(int(i))
    # print(dclasses)

    dscores_lines[0] = dscores_lines[0][1:]
    dscores_lines[len(dscores_lines) - 1] = dscores_lines[len(dscores_lines) - 1][:-1]
    for l in dscores_lines:
        l = l.split()
        for i in l:
            dscores.append(float(i))
    # print(dscores[:30])

    final_dboxes = []
    final_dclasses = []
    final_dscores = []

    for i in range(len(dscores)):
        if dclasses[i] in [1, 2, 3, 10]: # In this assignment, you only need to use the following classes: 1 (person), 2 (bicycle) ,3 (car) and 10 (traffic_light).
            if dscores[i] > threshold:
                final_dboxes.append(dboxes[i])
                final_dclasses.append(dclasses[i])
                final_dscores.append(dscores[i])
    # print(len(final_dboxes), len(final_dclasses), len(final_dscores))

    file = open("data/test/results/" + outfile + "_detection_boxes.txt", "w")
    file.write(str(final_dboxes))
    file.close()
    file = open("data/test/results/" + outfile + "_detection_classes.txt", "w")
    file.write(str(final_dclasses))
    file.close()
    file = open("data/test/results/" + outfile + "_detection_scores.txt", "w")
    file.write(str(final_dscores))
    file.close()

    return 1


# Relevant to Q2a
def camera_params(filename):

    # filename passed in is "004945"
    full_path = "data/test/calib/" + filename + "_allcalib.txt"
    fi = open(full_path, "r")
    lines = fi.readlines()
    fi.close()

    f = float(lines[0].split(":")[1].strip())
    px = float(lines[1].split(":")[1].strip())
    py = float(lines[2].split(":")[1].strip())
    baseline = float(lines[3].split(":")[1].strip())
    # print(f, px, py, baseline)

    return [f, px, py, baseline]


# Relevant to Q2a
def compute_depth(filename):

    # filename passed in is "004945"

    #depth is a n nxm matrix where n is height and m is width of original image
    #in pdf include a visualization of the depth matrices

    #camera parameters are in test/calib named xxx_ALLCALIB.txt  (baseline given in meters)

    disparity_image = read_image("data/test/results/" + filename + "_left_disparity.png", 0)
    if disparity_image is None:
        return -1

    [f, px, py, baseline] = camera_params(filename)
    Z = (f * baseline) / (disparity_image if not 0 else math.inf)

    write_image(Z, "data/test/results/" + filename + "_depth.png")
    result = read_image("data/test/results/" + filename + "_depth.png", 0)

    return result


# Relevant to Q1c
def q1c():

    while True:
        if game_is_paused:
            balls = scan_court_for_balls() # using object detection similar to question 2 of this assignment
            if len(balls) == 0:
                move_to_opposite_side_of_arena()
            balls = scan_court_for_balls()
            while len(balls) > 0:
                while cur_position != nearest_ball(balls):
                    go_to_nearest_ball(balls) # moves a couple inches in the best route
                grab_ball_and_put_into_bag(balls) # grabs the one at the current position
                if bag_is_full:
                    drop_off_balls_at_bin()
                # do another check for balls
                if len(balls) == 0:
                    move_to_opposite_side_of_arena()
                balls = scan_court_for_balls()
            # no more balls left
            return_to_resting_spot()
            victory_dance_at_location()

    return 1


def read_image(filename, t=None):

    if t == 0:
        img = cv2.imread(filename, 0)
    else:
        img = cv2.imread(filename)

    if img is not None:
        print(filename + " has been read")
    return img


def write_image(img, filename):

    if img is None:
        print(filename + " could not be written")
        return -1
    cv2.imwrite(filename, img)
    return 1


def main():

    images = ["004945", "004964", "005002"]
    # for img in images:
    #     compute_depth(img)

    input_txt = ["image3", "image4", "image5"]
    # store_strong_bounds(images[0], input_txt[0])
    # store_strong_bounds(images[1], input_txt[1])
    # store_strong_bounds(images[2], input_txt[2])
    # visualize_detections(images[0])
    # visualize_detections(images[1])
    # visualize_detections(images[2])

    # get_3D_locations(images[0])
    # get_3D_locations(images[1])
    # get_3D_locations(images[2])

    # segment(images[0])
    # segment(images[1])
    # segment(images[2])

    # show_descriptions(images[0])
    # show_descriptions(images[1])
    # show_descriptions(images[2])

if __name__ == '__main__':
    main()