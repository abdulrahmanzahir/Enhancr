import torch
from PIL import Image
import numpy as np
from models.srcnn import SRCNN

@torch.no_grad()
def enhance_image(input_path, output_path, ckpt='weights/srcnn_best.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = SRCNN().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device)['state_dict'])
    net.eval()
    img = Image.open(input_path).convert('RGB')
    arr = np.array(img)
    ten = torch.from_numpy(arr).permute(2,0,1).float().unsqueeze(0)/255.
    ten = ten.to(device)
    out = net(ten).clamp(0,1).cpu().squeeze(0)
    out_img = Image.fromarray((out.permute(1,2,0).numpy()*255).astype('uint8'))
    out_img.save(output_path, format='PNG')

if __name__ == '__main__':
    enhance_image('sample_input.jpg', 'enhanced.png')
