import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import platform

# font settings - for korean
if platform.system() == 'Darwin': # MacOS
        plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows': # Windows
        plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False



# setting api parameters
subscription_key = 'bb2cc1a868344e73a31aa6bbf88636b5'
vision_base_url = 'https://ai4scattervision.cognitiveservices.azure.com/vision/v2.0/'
ocr_url = vision_base_url + 'ocr'

headers = {'Ocp-Apim-Subscription-Key': subscription_key}
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


for url in image_urls:
    data = {'url': url}
    response = requests.post(ocr_url, headers=headers, params=params, json=data)
    result = response.json()

    if 'regions' in result:
        line_infos = [region['lines'] for region in result['regions']]
        print(line_infos)
        word_infos = []
        for line in line_infos:
            for word_metadata in line:
                for word_info in word_metadata['words']:
                    word_infos.append(word_info)

        plt.figure(figsize=(5, 5))
        image = Image.open(BytesIO(requests.get(url).content))

        ax = plt.imshow(image, alpha=0.5)

        for word in word_infos:
            bbox = [int(num) for num in word["boundingBox"].split(",")]
            text = word["text"]
            origin = (bbox[0], bbox[1]) # 0: x, 1: y, 2: w, 3: h
            patch = plt.Rectangle(origin, bbox[2], bbox[3], fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=17, weight="bold", va="top")

        plt.axis("off")
        plt.show()
    else:
        error_code = result['code']
        print("OCR API Error --> ", error_code)




