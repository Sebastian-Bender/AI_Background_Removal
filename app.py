import os
import numpy as np
import torch
from PIL import Image
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.image
import uuid

from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(3, 1).to(device)
net.load_state_dict(torch.load('Model/net.pt', map_location=device))
net.eval()


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_model(filename):

    original_img = Image.open(filename).convert("RGB")
    shape = original_img.size
    img = original_img.resize((350, 350))
    img = np.array(img)
    img = torch.as_tensor(np.expand_dims(img, axis=0), dtype=torch.uint8)
    img_in = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
    out = net(img_in)[0].permute(1, 2, 0).reshape(img[0].shape[0], img[0].shape[1]).detach().cpu()
    mask_pred = np.zeros(out.shape)
    mask_pred[out>0.7] = 1
    
    mask_pred = Image.fromarray(mask_pred).resize(shape)
    cut = (np.array(original_img) * np.array(mask_pred).reshape(shape[1], shape[0], 1)).astype(np.uint8)
    
    cut = Image.fromarray(cut).convert('RGBA')
    arr = np.array(np.asarray(cut))
    r,g,b,a=np.rollaxis(arr,axis=-1)    
    mask=((r==0)&(g==0)&(b==0))
    arr[mask,3]=0
    cut = arr
    
    matplotlib.image.imsave(filename, cut)

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = str(uuid.uuid4()) + '.png'
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        run_model(UPLOAD_FOLDER+filename)
        #print('upload_image filename: ' + filename)
        flash('Image background successfully removed')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()