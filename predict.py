from email.mime import image
from itertools import islice
import logging
import os
from time import sleep
from typing import List
import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from datasets.nparray import NumpyArray
from model import generate_model
from opts import parse_opts
import transforms

logger = logging.getLogger()
opt = parse_opts()


def getCamImg():
    for file in os.listdir("test_imgs"):
        if file.endswith(".jpg"):
            yield cv2.imread(os.path.join("test_imgs", file))
    # cap = cv2.VideoCapture()
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))

    # while True:
    #     try:
    #         ret, frame = cap.read()

    #         if not ret:
    #             logger.warning("Can't get image from camera")
    #             sleep(0.5)
    #             continue

    #         yield frame
    #     except:
    #         cap.release()
    #         break


def preprocess(pictures: List[npt.NDArray]):
    # 假設輸入的圖片是numpy array，並且是 (H x W x C)，並且範圍在0~255
    video_transform = transforms.Compose(
        [
            lambda img: cv2.resize(img, (224, 224)),
            transforms.ToTensor(opt.video_norm_value),
        ]
    )
    return [video_transform(img) for img in pictures]


def main():
    logger.info("Launching with opts: ", opt)

    if opt.device != "cpu":
        opt.device = "cuda" if torch.cuda.is_available() else "cpu"

    opt.store_name = "_".join([opt.dataset, opt.model, str(opt.sample_duration)])

    model, parameters = generate_model(opt)

    best_state = torch.load(
        "%s/%s_best" % (opt.result_path, opt.store_name) + "0" + ".pth"
    )
    model.load_state_dict(best_state["state_dict"])

    camera = getCamImg()

    video_transform = transforms.Compose(
        [
            lambda img: cv2.resize(img, (224, 224)),
            transforms.ToTensor(opt.video_norm_value),
        ]
    )
    # 每個dataset一定只能15張
    images = [*islice(camera, 15)]
    images = NumpyArray(images, video_transform)
    image_loader = DataLoader(
        images, opt.batch_size, shuffle=False, num_workers=opt.n_threads
    )
    for inputs_audio, inputs_visual in image_loader:
        inputs_visual = inputs_visual.reshape(
            inputs_visual.shape[0] * inputs_visual.shape[1],
            inputs_visual.shape[2],
            inputs_visual.shape[3],
            inputs_visual.shape[4],
        )
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)
            inputs_audio = Variable(inputs_audio)
        outputs = model(inputs_audio, inputs_visual)
        print(torch.asarray(outputs))
        print(f"{type(outputs)=}")
        print(f"{outputs.shape=}")


main()
