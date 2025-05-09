import cv2
import numpy as np
import hashlib

def dialogue_box_code(img, turn):


    dialogue_start = [255, 255, 0, 255, 255, 0, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 255, 255, 0, 255, 255]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)

    # Find all matching rows
    matching_rows = np.all(img == dialogue_start, axis=1)

    # Get the index of the first matching row
    start_row_indices = np.where(matching_rows)[0]

    # Return the first matching row index or -1 if not found
    start_row = start_row_indices[0] if len(start_row_indices) > 0 else -1

    if start_row == -1:
        return 0
    
    if turn != 0:
        return 1
    
    dialogue_region = img[start_row:]

    flattened = dialogue_region.flatten().astype(np.uint8).tobytes()
    text = hashlib.md5(flattened).hexdigest()

    return text