import cv2


# Crop to obtain a single person image
def cut_person(location, img, output):
    # img = Image.open(input)
    height = len(img)
    width = len(img[0])
    # width, height = img.size
    person_center_x = location[0]
    person_center_y = location[1]
    person_width = location[2] / 2
    person_height = location[3] / 2
    x1 = int((person_center_x - person_width) * width)
    x2 = int((person_center_x + person_width) * width)
    y1 = int((person_center_y - person_height) * height)
    y2 = int((person_center_y + person_height) * height)
    img = img[y1:y2, x1:x2]
    if output != '':
        cv2.imwrite(output, img)
    return int((x1 + x2) / 2), y1, img
