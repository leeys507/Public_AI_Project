import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import platform
import numpy as np
import uuid
import datetime

# font settings - for korean
if platform.system() == 'Darwin': # MacOS
        plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows
        plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# setting api parameters
subscription_key = 'bb2cc1a868344e73a31aa6bbf88636b5'
vision_base_url = 'https://ai4scattervision.cognitiveservices.azure.com/vision/v3.2/'
ocr_url = vision_base_url + 'ocr'

headers = {'Ocp-Apim-Subscription-Key': subscription_key,
            "Content-Type": "application/octet-stream"
    }
params = {
    'language': 'ko',
    'detectOrientation': 'true'
}

# test image links array
image_urls = [
    'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR7swvSJQSA75ONiK16girn4JLgkifkc1QuBQ&usqp=CAU',
    'https://koreajoongangdaily.joins.com/jmnet/koreajoongangdaily/_data/photo/2018/03/11204030.jpg',
    'https://www.licenseplates.tv/images/intkore.gif'
]

def add_padding(pil_img, top, bottom, left, right, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_text(imgs, image_format, img_save_path="."):
    if imgs == None or len(imgs) == 0: return

    color = (0, 0, 0)
    print_text_list = []
    for img in imgs:
        if image_format.lower() == "jpg": img_format = "JPEG"
        else: img_format = image_format

        if img.width < 50:  # OCR API image larger than 50 x 50
            padding = int((50 - img.width) / 2)
            if not (img.width % 2 == 0):
                padding += 1
            img = add_padding(img, 0, 0, padding, padding, color) # img.resize((50, img.height))
        if img.height < 50:
            padding = int((50 - img.height) / 2)
            if not (img.height % 2 == 0):
                padding += 1
            img = add_padding(img, padding, padding, 0, 0, color) # img.resize((img.width, 50))

        img_data = img

        img_byte_arr = BytesIO()
        img_data.save(img_byte_arr, format=img_format)
        img_byte_arr = img_byte_arr.getvalue()

        response = requests.post(ocr_url, headers=headers, params=params, data=img_byte_arr)
        result = response.json()
        # print(result)

        if 'regions' in result:
            line_infos = [region['lines'] for region in result['regions']]
            print(line_infos)
            word_infos = []
            for line in line_infos:
                for word_metadata in line:
                    for word_info in word_metadata['words']:
                        word_infos.append(word_info)

            plt.figure(figsize=(5, 5))
            # image = Image.open(BytesIO(requests.get(url).content))

            ax = plt.imshow(img, alpha=0.5)

            print_text = ""
            for word in word_infos:
                bbox = [int(num) for num in word["boundingBox"].split(",")]
                if print_text != "": print_text += " "
                print_text += word["text"]
                text = word["text"]
                # print(text) # concat
                origin = (bbox[0], bbox[1]) # 0: x, 1: y, 2: w, 3: h
                patch = plt.Rectangle(origin, bbox[2], bbox[3], fill=False, linewidth=2, color='y')
                ax.axes.add_patch(patch)
                plt.text(origin[0], origin[1], text, fontsize=17, weight="bold", va="top")

            time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            if print_text != "":
                bad_char = '\/:*?"<>|'
                for c in bad_char:
                    if c in print_text:
                        print_text = print_text.replace(c, "")
                plt.savefig(img_save_path + "/" + print_text + "-" + time_now + "." + img_format)
                print_text_list.append(print_text)
            else:
                plt.savefig(img_save_path + "/" + time_now + "-" + str(uuid.uuid4()) + "." + img_format)
            plt.axis("off")
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

        else:
            error_code = result['code']
            print("OCR API Error --> ", error_code)
            return ""
    
    return print_text_list