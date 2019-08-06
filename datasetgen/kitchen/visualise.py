import cv2
from PIL import Image, ImageDraw
import numpy as np

def show_predictions(scene, correct, incorrect):
    pil_image = Image.new('RGB', (512,512), (255, 255, 255))
    draw = ImageDraw.Draw(pil_image)  
    scene.render(draw, withindexes=True, highlight=correct, fails=incorrect) #, withindexes=self.show_idx_item.IsChecked())
    img = np.array(pil_image)

    cv2.namedWindow('Kitchen Scene')

    next_ex = False
    cv2.imshow('Kitchen Scene',img)
    k = cv2.waitKey() & 0xFF            
    # key bindings
    if k == 9:         # tab to next image
        next_ex = True

    # end of while loop
    cv2.destroyAllWindows()

    return next_ex
