import sys
sys.path.append('model')
sys.path.append('utils')

from utils_SH import *
import numpy as np

from torch.autograd import Variable
import torch
import cv2
import runway
from runway.data_types import *
from defineHourglass_512_gray_skip import *

@runway.setup(options={"checkpoint" : runway.file(extension=".t7")})
def setup(opts):
    model = HourglassNet()
    model.load_state_dict(torch.load(opts["checkpoint"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device {}".format(device))
    model.to(device)
    model.train(False)
    return model


preset_parameters = category(choices=["left", "top_left", "top", "top_right", "right",
                                    "bottom", "bottom_left"], default="left")

preset_inputs = {"input_image" : image, "preset_parameters" : preset_parameters}
preset_outputs = {"output_image" : image}

custom_inputs = {"input_image" : image, "Intensity" : number(step=0.001, min=-5, max=5), "Distance" : number(step=0.001, min=-5, max=5), "Y" : number(step=0.001, min=-5, max=5),
                 "X" : number(step=0.001, min=-5, max=5), "L2-2" : number(step=0.001, min=-5, max=5), "L2-1" : number(step=0.001, min=-5, max=5), "L20" : number(step=0.001, min=-5, max=5),
                 "L21" : number(step=0.001, min=-5, max=5), "L22" : number(step=0.001, min=-5, max=5)}
custom_outputs = {"output_image" : image}


@runway.command("relight_using_preset", inputs=preset_inputs, outputs=preset_outputs, description="Relight images from preset lighting")
def relight_image(model, inputs):
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = np.array(inputs["input_image"])
    row, col, _ = img.shape

    img = cv2.resize(img, (512, 512))

    Lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    inputL = Lab[:, :, 0]
    inputL = inputL.astype(np.float32) / 255.0
    inputL = inputL.transpose((0, 1))
    inputL = inputL[None, None, ...]
    inputL = Variable(torch.from_numpy(inputL).to(device))

    user_choice = inputs["preset_parameters"]

    if user_choice == "left":
        sh = np.array([ 1.0841255, -0.46426763, 0.02837847, 0.67652927, -0.35940677,  0.04790996,
                       -0.22800546, -0.08125983, 0.2881082 ])
    elif user_choice == "top_left":
        sh = np.array([ 1.0841255, -0.46426763, 0.54662557,  0.39962192, -0.26154398, -0.25112416,
                        0.06495695,  0.3510322, 0.11896627])
    elif user_choice == "top":
        sh = np.array([ 1.0841255,  -0.46426763,  0.65325247, -0.17820889,  0.03326677, -0.36105666,
                        0.36475618, -0.0749642,  -0.05412289])
    elif user_choice == "top_right":
        sh = np.array([ 1.0841255, -0.46426763, 0.26796693, -0.62184477, 0.30302696, -0.19910614,
                       -0.06162944, -0.31767, 0.19205096])
    elif user_choice == "right":
        sh = np.array([ 1.0841255, -0.46426763, -0.31910317, -0.59721886, 0.34460167, 0.11277537,
                       -0.17166922, 0.21634065, 0.25558246])
    elif user_choice == "bottom":
        sh = np.array([1.0841255, -0.46426763, -0.66588208, -0.12287497, 0.12668429, 0.33973472,
                       0.30368871, 0.22138935, -0.01886557])
    elif user_choice == "bottom_left":
        sh = np.array([ 1.0841255, -0.46426763, -0.5112382, 0.44399628, -0.18662894, 0.3108669,
                        0.2021743, -0.31486818, 0.0397438 ])
    sh = sh[0:9]
    sh = sh * 0.7

    sh = np.squeeze(sh)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).to(device))
    outputImg, outputSH = model(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg * 255.0).astype(np.uint8)
    Lab[:, :, 0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB)
    resultLab = cv2.resize(resultLab, (col, row))

    return { "output_image" : resultLab}


@runway.command("relight_using_coefficients", inputs=custom_inputs, outputs=custom_outputs, description="Relight images using custom coefficients.")
def relight_image(model, inputs):
    img_size = 256
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x ** 2 + z ** 2)
    valid = mag <= 1
    y = -np.sqrt(1 - (x * valid) ** 2 - (z * valid) ** 2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[..., None], y[..., None], z[..., None]), axis=2)
    normal = np.reshape(normal, (-1, 3))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = np.array(inputs["input_image"])
    row, col, _ = img.shape

    img = cv2.resize(img, (512, 512))

    Lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    inputL = Lab[:, :, 0]
    inputL = inputL.astype(np.float32) / 255.0
    inputL = inputL.transpose((0, 1))
    inputL = inputL[None, None, ...]
    inputL = Variable(torch.from_numpy(inputL).to(device))


    sh = np.array([inputs["Intensity"], inputs["Distance"], inputs["Y"], inputs["X"],
                   inputs["L2-2"], inputs["L2-1"], inputs["L20"], inputs["L21"],
                   inputs["L22"]])
    sh = sh[0:9]
    sh = sh * 0.7

    sh = np.squeeze(sh)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading)) / (np.max(shading) - np.min(shading))
    shading = (shading * 255.0).astype(np.uint8)
    shading = np.reshape(shading, (256, 256))
    shading = shading * valid

    sh = np.reshape(sh, (1, 9, 1, 1)).astype(np.float32)
    sh = Variable(torch.from_numpy(sh).to(device))
    outputImg, outputSH = model(inputL, sh, 0)
    outputImg = outputImg[0].cpu().data.numpy()
    outputImg = outputImg.transpose((1, 2, 0))
    outputImg = np.squeeze(outputImg)
    outputImg = (outputImg * 255.0).astype(np.uint8)
    Lab[:, :, 0] = outputImg
    resultLab = cv2.cvtColor(Lab, cv2.COLOR_LAB2RGB)
    resultLab = cv2.resize(resultLab, (col, row))

    return {"output_image": resultLab}




if __name__ == "__main__":
    runway.run(model_options={"checkpoint" : "trained_model/trained_model_03.t7" })
