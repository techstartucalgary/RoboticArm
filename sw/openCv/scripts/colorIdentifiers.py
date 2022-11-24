import shutil
import os
from tkinter import filedialog
from datetime import datetime

# only reliable on svgs that I made
def alterSvg(imgPath, exportPath):
    itemIn = open(imgPath, 'rt')
    data = itemIn.read()
    color_hex = int("FFFFFF", 16)
    hex_adjustment = "00F0A2"

    data = data.replace('fill="white"', 'fill="#FFFFFF"')

    for i in range(0, 30):
        new_color = color_hex - int(hex_adjustment, 16)
        data = data.replace('fill="#' + str(hex(color_hex)[2:]).upper()  + '"', 'fill="#' + str(hex(new_color)[2:]).upper() + '"')
        color_hex = new_color
        with open(
            os.path.join(
                exportPath, 
                os.path.basename(imgPath).replace('.svg', '') + '_' + str(hex(color_hex)) + '.svg'
            ),
            'w+'
        ) as itemOut:
            itemOut.write(data)
        
        itemOut.close()
    itemIn.close()

def getDir():
    dir = filedialog.askdirectory()
    date = datetime.now().strftime("%d %m %Y %H:%M:%S").replace(' ', '_').replace(':', '-')
    
    new_dir = os.path.join(dir, 'working_' +  date + '/').replace('\\', '/')
    export_dir = os.path.join(new_dir, 'output').replace('\\', '/')
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    
    os.mkdir(new_dir)
    os.mkdir(export_dir)

    for f in os.listdir(dir):
        if(f.endswith('.svg')):
            shutil.copy(
                os.path.join(dir, f), 
                new_dir
            )
    return new_dir, export_dir

def colorImgs(imageDir, output):
    for f in os.listdir(imageDir):
        img = os.path.join(imageDir, f).replace('\\', '/')
        if img.endswith('.svg'):
            alterSvg(img, output)

def main():
    imageDir, output = getDir()
    colorImgs(imageDir, output)

if __name__ in '__main__':
    main()