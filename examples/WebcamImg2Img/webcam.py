# import the opencv library 
import cv2 

#streamdiffusion dependencies?
import os
import sys
import time
import threading
from multiprocessing import Process, Queue, get_context
from typing import List, Literal, Dict, Optional
import PIL.Image
from streamdiffusion.image_utils import pil2tensor
import mss
import fire
import tkinter as tk

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

#StreamDiffusion
import torch
from diffusers import AutoencoderTiny, StableDiffusionPipeline
from diffusers.utils import load_image

from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image

#converters
import numpy as np
from PIL import Image


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

  
# You can load any models using diffuser's StableDiffusionPipeline
#KBlueLeaf/kohaku-v2.1
#stabilityai/sdxl-turbo
#KBlueLeaf/kohaku-v3-rev2
pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v3-rev2").to(
    device=torch.device("cuda"),
    dtype=torch.float16,
)

width = 960 #640
height = 540 #360

# Wrap the pipeline in StreamDiffusion
stream = StreamDiffusion(
    pipe,
    t_index_list=[32, 45], #32, 45  35/45 sems better..
    torch_dtype=torch.float16,
    width=width,
    height=height,
    use_denoising_batch=True,
    cfg_type="full", ## "none", "full", "self", "initialize"
)

# If the loaded model is not LCM, merge LCM
stream.load_lcm_lora()
stream.fuse_lora()
# Use Tiny VAE for further acceleration
stream.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd").to(device=pipe.device, dtype=pipe.dtype)
# Enable acceleration
pipe.enable_xformers_memory_efficient_attention()

prompt = "1girl, breasts, green eyes, brown hair, lipstick, makeup, blush, long hair, green shirt, nails, indoors, looking at viewer, solo, confetti, happy, jewelry, sparkle, gliter, flower petals, big eyes"
negative_prompt = "low quality, bad quality, blurry, low resolution"
guidance_scale= 1.4
delta = 0.5


# Prepare the stream
#stream.prepare(prompt)
stream.prepare(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=guidance_scale,
        delta=delta,
        seed=-1,
)


# Prepare image
# init_image = load_image("assets/img2img_example.png").resize((512, 512))

  
# define a video capture object 
vid = cv2.VideoCapture(0) 

# Warmup >= len(t_index_list) x frame_buffer_size
for _ in range(2):
    ret, frame = vid.read()
    #stream(init_image)
    stream(convert_from_cv2_to_image(frame)) 


# Run infinitely
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
    #resize frame
    frame = cv2.resize(frame, (width, height)) #16:9\

    #split RGB
    #b,g,r = cv2.split(frame)
    #rgb_img1 = cv2.merge([r,g,b]) # switch it to r, g, b

    #ret, th1 = cv2.threshold(frame,160, 255, cv2.THRESH_BINARY) 
    #th2 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    #th3 = cv2.adaptiveThreshold( frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


    #edge
    #edges = cv2.Canny(frame,200,300,True)

    #to stablediffustion
    PIL_output = stream(convert_from_cv2_to_image(frame))
    #PIL_output = stream( convert_from_cv2_to_image( cv2.blur(frame, (5,5)) ) )
    #PIL_output = stream( convert_from_cv2_to_image( cv2.medianBlur(frame, 5) ) )
    #PIL_output = stream( convert_from_cv2_to_image( cv2.GaussianBlur(frame, (5,5), cv2.BORDER_DEFAULT) ) )
    #PIL_output = stream( convert_from_cv2_to_image( cv2.bilateralFilter(frame,9,75,75) ) )
    #PIL_output = stream(convert_from_cv2_to_image(th3))
    #PIL_output = stream(convert_from_cv2_to_image(b))
    #PIL_output = stream(convert_from_cv2_to_image(edges))

    # Display the resulting frame streamdiffution way
    PIL_prostprocess = postprocess_image(PIL_output)[0]

    # Display the resulting frame with cv2
    cv2_output = convert_from_image_to_cv2(PIL_prostprocess)

    cv2.imshow('camera', cv2_output)
    #cv2.imshow('cameraRaw', frame)
    #cv2.imshow('edge',edges)
    #cv2.imshow('th3',th3)
    #cv2.imshow('camera', rgb_img1)
    #plt.subplot(121),plt.imshow(rgb_img1),plt.title('TIGER_COLOR')
      
    # the 'q' button is set as the 
    # 1 - 6 switch between prompts
    key = cv2.waitKey(1)
    if (key == ord('q') ):
        break
    if (key == ord('1') ):
        prompt = "1girl, breasts, green eyes, brown hair, lipstick, makeup, blush, long hair, green shirt, nails, indoors, looking at viewer, solo, confetti, happy, jewelry, sparkle, gliter, flower petals, big eyes"
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
    elif (key == ord('2') ):
        prompt = "gundam, emblem, machinery, mecha, mobile suit, no humans, robot, thrusters, zeon, science fiction, armor, helmet, wepon, armored core, armored core 6, steel haze, full body, mecha focus, missile pod, red eyes, shoulder cannon, iron_man, metalic, red metal, red body,"
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
    elif (key == ord('3') ):
        prompt = "grim_reaper, solo, no humans, scythe, skeleton, skull, cloak, black cloak, hood, hood up, hooded cloak, looking at viewer, mask, teeth, red glowing eyes, glasses, chibi"
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
    elif (key == ord('4') ):
        prompt = "astronaut, helmet, solo, space, space helmet, gloves, looking at viewer, spacesuit"
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
    elif (key == ord('5') ):
        prompt = "1girl, solo, maid, dress, apron, falling petals, short hair, black hair, breasts, ribbon, ponytail, black dress, white bow, white apron"
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
    elif (key == ord('6') ):
        prompt = "1boy, smile, solo, green hair, short hair, serious, scar, black eyes, glasses"
        stream.prepare(
            prompt=prompt,
            negative_prompt=negative_prompt,
    )

  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows
cv2.destroyAllWindows() 