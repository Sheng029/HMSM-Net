import os
import cv2
import threading
from PIL import Image


def crop_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    image_lu = image[0:512, 0:512, :]
    image_ru = image[0:512, 512:1024, :]
    image_lb = image[512:1024, 0:512, :]
    image_rb = image[512:1024, 512:1024, :]
    return image_lu, image_ru, image_lb, image_rb


def crop_label(filename):
    label = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    label_lu = label[0:512, 0:512]
    label_ru = label[0:512, 512:1024]
    label_lb = label[512:1024, 0:512]
    label_rb = label[512:1024, 512:1024]
    return label_lu, label_ru, label_lb, label_rb


def write(src_dir, des_dir, flag):
    src_files = os.listdir(src_dir)
    for src_file in src_files:
        # image
        if flag == 0:
            image_lu, image_ru, image_lb, image_rb = crop_image(os.path.join(src_dir, src_file))
            lu = src_file.replace('RGB', 'LU_RGB')
            ru = src_file.replace('RGB', 'RU_RGB')
            lb = src_file.replace('RGB', 'LB_RGB')
            rb = src_file.replace('RGB', 'RB_RGB')
            image_lu = Image.fromarray(image_lu)
            image_ru = Image.fromarray(image_ru)
            image_lb = Image.fromarray(image_lb)
            image_rb = Image.fromarray(image_rb)
            image_lu.save(os.path.join(des_dir, lu))
            image_ru.save(os.path.join(des_dir, ru))
            image_lb.save(os.path.join(des_dir, lb))
            image_rb.save(os.path.join(des_dir, rb))
        # cls
        elif flag == 1:
            label_lu, label_ru, label_lb, label_rb = crop_label(os.path.join(src_dir, src_file))
            lu = src_file.replace('CLS', 'LU_CLS')
            ru = src_file.replace('CLS', 'RU_CLS')
            lb = src_file.replace('CLS', 'LB_CLS')
            rb = src_file.replace('CLS', 'RB_CLS')
            label_lu = Image.fromarray(label_lu)
            label_ru = Image.fromarray(label_ru)
            label_lb = Image.fromarray(label_lb)
            label_rb = Image.fromarray(label_rb)
            label_lu.save(os.path.join(des_dir, lu))
            label_ru.save(os.path.join(des_dir, ru))
            label_lb.save(os.path.join(des_dir, lb))
            label_rb.save(os.path.join(des_dir, rb))
        # disp
        elif flag == 2:
            label_lu, label_ru, label_lb, label_rb = crop_label(os.path.join(src_dir, src_file))
            lu = src_file.replace('DSP', 'LU_DSP')
            ru = src_file.replace('DSP', 'RU_DSP')
            lb = src_file.replace('DSP', 'LB_DSP')
            rb = src_file.replace('DSP', 'RB_DSP')
            label_lu = Image.fromarray(label_lu)
            label_ru = Image.fromarray(label_ru)
            label_lb = Image.fromarray(label_lb)
            label_rb = Image.fromarray(label_rb)
            label_lu.save(os.path.join(des_dir, lu))
            label_ru.save(os.path.join(des_dir, ru))
            label_lb.save(os.path.join(des_dir, lb))
            label_rb.save(os.path.join(des_dir, rb))
        else:
            raise TypeError("flag must be 0, 1, or 2!")
        print(src_file, ' done.')


if __name__ == '__main__':
    # left_src_dir = 'G:/Disparity Estimation/Datasets/US3D/JAX/train/left'
    # left_des_dir = 'G:/Disparity Estimation/Datasets/US3D/crop_JAX/train/left'
    # t1 = threading.Thread(target=write, args=(left_src_dir, left_des_dir, 0))
    #
    # right_src_dir = 'G:/Disparity Estimation/Datasets/US3D/JAX/train/right'
    # right_des_dir = 'G:/Disparity Estimation/Datasets/US3D/crop_JAX/train/right'
    # t2 = threading.Thread(target=write, args=(right_src_dir, right_des_dir, 0))
    #
    # cls_src_dir = 'G:/Disparity Estimation/Datasets/US3D/JAX/train/cls'
    # cls_des_dir = 'G:/Disparity Estimation/Datasets/US3D/crop_JAX/train/cls'
    # t3 = threading.Thread(target=write, args=(cls_src_dir, cls_des_dir, 1))
    #
    # dsp_src_dir = 'G:/Disparity Estimation/Datasets/US3D/JAX/train/disp'
    # dsp_des_dir = 'G:/Disparity Estimation/Datasets/US3D/crop_JAX/train/disp'
    # t4 = threading.Thread(target=write, args=(dsp_src_dir, dsp_des_dir, 2))
    #
    # t1.start()
    # t2.start()
    # t3.start()
    # t4.start()

    pass
