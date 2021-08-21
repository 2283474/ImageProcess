from os import listdir
import cv2
import numpy as np
from collections import deque
from tqdm import trange


def star_tails(folder_path, file_extension, output_path, filter_func=None, stack_func=np.median, stack_num=3):

    files = sorted([folder_path + f for f in listdir(folder_path) if file_extension in f])
    result = np.zeros(cv2.imread(files[0]).shape)
    img_que = deque([])
    for i in trange(len(files)):
        new_img = cv2.imread(files[i])

        if filter_func:
            new_img = filter_func(new_img)

        if stack_func:
            if len(img_que) < stack_num:
                img_que.append(new_img)

            if len(img_que) == stack_num:
                stack_result = stack_func(np.array(img_que), axis=0)
                result = np.max([result, stack_result], axis=0)
                img_que.popleft()
        else:
            result = np.max([result, new_img], axis=0)

    cv2.imwrite(output_path, result)


if __name__ == "__main__":
    star_tails('./color_star_tails/', 'jpg', './output_demo.jpg')
