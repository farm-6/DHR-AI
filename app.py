# app.py
import torch
from flask import Flask, render_template, request, redirect, url_for
import base64
from model_loader import load_model
from image_preprocessor import preprocess_image

app = Flask(__name__)

class_names = {
    'main': ['식물', '양서파충류', '어류', '육상곤충', '조류', '포유류'],
    'reptile': ['누룩뱀', '도롱뇽', '두꺼비', '맹꽁이', '유혈목이', '자라', '줄장지뱀', '청개구리', '표범장지뱀'],
    'fish': ['가시납지리', '각시붕어', '갈문망둑', '긴몰개', '꺽정이', '누치', '됭경모치', '몰개', '미꾸리', '블루길',
             '잉어', '줄납자루', '참갈겨니', '참붕어', '피라미', '황복'],
    'insect': ['검은실다리베짱이', '꼬마꽃등에', '꼬마남생이무당벌레', '꽃매미', '네발나비', '물자라', '방울실잠자리', '섬서구메뚜기',
               '애긴노린재', '왕잠자리', '왕파리매', '호랑나비'],
    'plant': ['가시박', '갈풀', '개나리', '갯버들', '긴병풀꽃', '달뿌리풀', '말냉이', '수선화', '아까시나무', '애기똥풀', '자주개불주머니',
              '큰개불알풀', '환삼덩굴', '흰제비꽃'],
    'mammal': ['고라니', '너구리', '삵', '수달', '청설모'],
    'bird': ['개개비', '괭이갈매기', '붉은머리오목눈이', '비오리', '오색딱다구리', '원앙', '저어새', '중대백로', '직박구리', '참수리',
             '큰기러기', '황조롱이', '흰눈썹황금새', '흰목물떼새', '흰뺨검둥오리']
}

model_paths = {
    'main': 'model/ALL_efficientnet_b0_model.pt',
    'reptile': 'model/REPTILE_efficientnet_b0_model.pt',
    'fish': 'model/FISH_efficientnet_b0_model.pt',
    'insect': 'model/INSECT_efficientnet_b0_model.pt',
    'plant': 'model/PLANT_efficientnet_b0_model.pt',
    'mammal': 'model/MAMMAL_efficientnet_b0_model.pt',
    'bird': 'model/BIRDS_efficientnet_b0_model.pt'
}

# Define the number of classes for each model
num_classes = {
    'main': 14,
    'reptile': 14,
    'fish': 16,
    'insect': 14,
    'plant': 14,
    'mammal': 14,
    'bird': 15
}

models = {category: load_model(
    path, num_classes=num_classes[category]) for category, path in model_paths.items()}


def classify_image(model, class_names, image_bytes):
    preprocessed_image = preprocess_image(image_bytes=image_bytes)
    with torch.no_grad():
        input_tensor = preprocessed_image.unsqueeze(0)
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
        class_name = class_names[preds[0].item()]
    return class_name


@app.route("/")
def main():
    return render_template('app.html')


@app.route("/<category>", methods=['GET', 'POST'])
@app.route("/<category>/upload", methods=['POST'])
def classify(category):
    if request.method == 'POST':
        file = request.files['image']
        image_bytes = file.read()
        class_name = classify_image(
            models[category], class_names[category], image_bytes)

        result = {
            'class': class_name,
            'image_base64': base64.b64encode(image_bytes).decode('utf-8')
        }

        return render_template(f'{category}.html', result=result)
    else:
        return render_template(f'{category}.html', result=None)


if __name__ == '__main__':
    app.run()
