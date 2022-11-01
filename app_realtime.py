import cv2
import numpy as np
import torch
from PIL import Image

from model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
net = UNet(3, 1).to(device)
net.load_state_dict(torch.load('Model/net.pt', map_location=device))
net.eval()


cap = cv2.VideoCapture(0)
#transform = T.Resize((1000, 1000))
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[:1000, 500:1500, :]

    cv2.imshow('Background Remover', frame)

    img = np.array(cv2.resize(frame, (250, 250)))

    img = torch.as_tensor(np.expand_dims(img, axis=0), dtype=torch.uint8)
    img_in = img.permute(0, 3, 1, 2).type(torch.FloatTensor).to(device)
    out = net(img_in)[0].permute(1, 2, 0).reshape(img[0].shape[0], img[0].shape[1]).detach().cpu()
    mask_pred = np.zeros(out.shape)
    mask_pred[out>0.5] = 1
    #cut = (img[0].detach().cpu() * mask_pred.reshape(mask_pred.shape[0], mask_pred.shape[1], 1)).type(torch.IntTensor)

    #cut = cut.detach().cpu().numpy().astype(np.uint8)
    mask_pred = Image.fromarray(mask_pred).resize((1000, 1000))
    cut = (np.array(frame) * np.array(mask_pred).reshape(1000, 1000, 1)).astype(np.uint8)
    #cut[cut==0] = 255
    
    cut = Image.fromarray(cut).convert('RGBA')
    # np.asarray(img) is read only. Wrap it in np.array to make it modifiable.
    arr = np.array(np.asarray(cut))
    r,g,b,a=np.rollaxis(arr,axis=-1)    
    mask=((r==0)&(g==0)&(b==0)).T
    arr[mask,3]=0
    cut = arr
    #cut=Image.fromarray(arr,mode='RGBA')

    cv2.imshow('Background Remover', cut)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        #cut=Image.fromarray(cut,mode='RGBA')
        cv2.imwrite('test_out.png', cut)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()