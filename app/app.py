import io, torch, numpy as np
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from models.srcnn import SRCNN

app = FastAPI()
app.mount('/static', StaticFiles(directory='app/static'), name='static')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = SRCNN().to(device)
net.load_state_dict(torch.load('weights/srcnn.pth', map_location=device)['state_dict'])
net.eval()

@app.get('/')
def index():
    with open('app/static/index.html','r', encoding='utf-8') as f:
        return HTMLResponse(f.read())

@app.post('/enhance')
@torch.inference_mode()
async def enhance(file: UploadFile):
    img = Image.open(io.BytesIO(await file.read())).convert('RGB')
    arr = np.array(img)
    ten = torch.from_numpy(arr).permute(2,0,1).float().unsqueeze(0)/255.
    ten = ten.to(device)
    out = net(ten).clamp(0,1).cpu().squeeze(0)
    out_img = Image.fromarray((out.permute(1,2,0).numpy()*255).astype('uint8'))
    buf = io.BytesIO()
    out_img.save(buf, format='PNG'); buf.seek(0)
    headers = {"Content-Disposition": "attachment; filename=enhanced.png"}
    return StreamingResponse(buf, media_type='image/png', headers=headers)
