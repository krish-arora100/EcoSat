import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import datetime
import os
import rasterio as rio
import torch
from PIL import Image
import tifffile

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    time_utils
)

# The following is not a package. It is a file utils.py which should be in the same folder as this notebook.
from utils import plot_image, save_image
from model_unet import *

from sentinelhub import SHConfig

def get_available_timestamps(start, end, coords, path):
    print("getting timestamps")
    config = SHConfig()
    
    timestamps = []

    if os.path.isdir(path) == False:
        os.mkdir(path)

    path = path + "rgb_color_imgs/"
    if os.path.isdir(path) == False:
        os.mkdir(path)

    resolution = 15 #small is better
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    
    evalscript_true_color = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    while (start <= end):
        request_true_color = SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=(start, start + datetime.timedelta(hours=24)),
                    mosaicking_order=MosaickingOrder.LEAST_CC, #filter by cloud-coverage
                    maxcc=0.01 #max cloud coverage in image can be x*100 percent
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.JPG)],
            bbox=bbox,
            size=bbox_size,
            config=config,
        )

        true_color_imgs = request_true_color.get_data()

        #print(f"Returned data is of type = {type(true_color_imgs)} and length {len(true_color_imgs)}.")
        #print(f"Single element in the list is of type {type(true_color_imgs[-1])} and has shape {true_color_imgs[-1].shape}")

        image = true_color_imgs[0]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if np.mean(image) > 5 and np.mean(image) < 34: #no all-black or all-white images(not actual photos/all clouds maybe)
            #save image into folder, df
            date = start.date()

            print(date)
            print("success" + str(np.mean(image)))
            img_path = str(path + str(date) + ".jpg")
            #cv2.imwrite(img_path, image)
            save_image(img_path, image, factor=3.5 / 255, clip_range=(0, 1))
            timestamps.append(date)
            #increment start-time by 5 days
            start += datetime.timedelta(days=5)
        
        else:
            #increment start-time by 1 day
            start += datetime.timedelta(days=1)
        
    return timestamps



def get_images(timestamp, coords, path):
    print("getting 12 band images")
    config = SHConfig()

    if os.path.isdir(path + "all_bands_imgs")  == False:
        os.mkdir(path + "all_bands_imgs")
    
    if os.path.isdir(path + "rgb_zoomed_imgs")  == False:
        os.mkdir(path + "rgb_zoomed_imgs")

    resolution = 8 #small is better
    bbox = BBox(bbox=coords, crs=CRS.WGS84)
    bbox_size = bbox_to_dimensions(bbox, resolution=resolution)
    
    evalscript_all_bands = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08", "B8A", "B09", "B10", "B11","B12"],
                    units: "DN"
                }],
                output: {
                    bands: 13,
                    sampleType: "FLOAT32"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01,
                    sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B06,
                    sample.B07,
                    sample.B08,
                    sample.B8A,
                    sample.B09,
                    sample.B10,
                    sample.B11,
                    sample.B12];
        }
    """
    #date_format = '%Y-%m-%d' #for converting string to datetime
    #timestamp = datetime.datetime.strptime(timestamp, date_format).date()

    request_all_bands = SentinelHubRequest(
        data_folder = path + "all_bands_imgs/" + str(timestamp), 
        evalscript=evalscript_all_bands,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(timestamp, timestamp+datetime.timedelta(hours=24)),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_size,
        config=config,
    )

    all_bands_imgs = request_all_bands.get_data(save_data=True)

    image = all_bands_imgs[0][:, :, 12]

    rgb_image = all_bands_imgs[0][:, :, [4, 3, 2]]
    #plot_image(all_bands_imgs[0][:, :, [4, 3, 2]], factor=3.5 / 1e4, clip_range=(0,1))
    rgb_path = path + "rgb_zoomed_imgs/" + str(timestamp) + ".jpg"
    save_image(rgb_path, rgb_image, factor=3.5 / 255, clip_range=(0, 1))
    
    #get path of image
    image_path = ""
    for folder, _, filenames in os.walk(request_all_bands.data_folder):
        for filename in filenames:
            path = os.path.join(folder, filename)
            path = os.path.relpath(path)

            # this will return a tuple of root and extension
            split_tup = os.path.splitext(path)

            if str(split_tup[1]) == ".tiff": #get just the images
                image_path = path

    return image_path

def segment_smoke_plumes(timestamp, img_path, path):
    print("segmenting smoke plumes")
    # read in image data
    imgfile = rio.open(img_path)
    imgdata = np.array([imgfile.read(i) for i in
                        [1,2,3,4,5,6,7,8,9,10,12,13]])


    imgdata = np.float32(imgdata)

    model.load_state_dict(torch.load('/Users/26krisha/Desktop/SentinelSatelliteDataCollection/IndustrialSmokePlumeDetection/segmentation/ep150_lr7e-01_bs60_mo0.7_134.model'))
    model.eval()  # Set the model to evaluation mode


    input_image = torch.from_numpy(imgdata).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)
        # Process the output to get the final segmentation result



    # obtain binary prediction map
    pred = np.zeros(output.shape)
    pred[output >= 0] = 1

    prediction = np.array(np.sum(pred,
                            axis=(1,2,3)) != 0).astype(int)


    # derive binary segmentation map from prediction
    output_binary = np.zeros(output.shape)
    output_binary[output.cpu().detach() >= 0] = 1

    # derive smoke areas
    area_pred = np.sum(output_binary, axis=(1,2,3))


    #get data in correct shape
    output_array = output.numpy()


    # segmentation ground-truth and prediction
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    ax.set_axis_off()
    ax.axis('off')

    ax.imshow(output_array[0][0], alpha=0.3, cmap="Greys")

    seg_path = path + "smoke_seg_imgs/"
    rgb_path = path + "rgb_zoomed_imgs/"

    if os.path.isdir(seg_path) == False:
        os.mkdir(seg_path)
    
    fig.savefig(seg_path + str(timestamp) + ".jpg", bbox_inches='tight')

    f, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(1, 3))
    # RGB plot
    ax1.imshow(0.2+1.5*(np.dstack([input_image[0][3], input_image[0][2], input_image[0][1]])-
                np.min([input_image[0][3].numpy(),
                        input_image[0][2].numpy(),
                        input_image[0][1].numpy()]))/
            (np.max([input_image[0][3].numpy(),
                        input_image[0][2].numpy(),
                        input_image[0][1].numpy()])-
                np.min([input_image[0][3].numpy(),
                        input_image[0][2].numpy(),
                        input_image[0][1].numpy()])),
            origin='upper')
    
    f.savefig(rgb_path + str(timestamp) + ".jpg", bbox_inches='tight')

    return seg_path + str(timestamp) + ".jpg"

def get_area_of_smoke_plumes(timestamp, img_path, path):
    print("getting area of smoke plumes")
    img = cv2.imread(img_path)
    
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 195, 255, cv2.THRESH_BINARY) #first number is the threshold value for pixels, 
    #second number is what we should set any pixel that is more than the threshold to(255 is black)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    contourAreas = []

    for i, value in enumerate(contours):
        contourAreas.append(cv2.contourArea(value))

    contourAreas.remove(max(contourAreas)) #removes border contour

    total_area = 0
    for i, value in enumerate(contourAreas):
        total_area += value


    contours_path = path + "smoke_contours_imgs/"

    if os.path.isdir(contours_path) == False:
        os.mkdir(contours_path)
    
    cv2.imwrite(contours_path + str(timestamp) + ".jpg", img)


    return contours_path + str(timestamp) + ".jpg", total_area