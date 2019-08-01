import argparse
from PIL import Image, ImageDraw
import json

from .scene import Scene, SceneObject
from .cfggenerator import CFGGenerator, ProductionRule

parser = argparse.ArgumentParser()
parser.add_argument('--image_width', default=800,
    help="image width in pixels")
parser.add_argument('--image_height', default=600,
    help="image height in pixels")
parser.add_argument('--object_count', default=10,
    help="number of objects to place in the scene")
parser.add_argument('--output_path', default='./output',
    help="path to the output directory")
parser.add_argument('--template_path', default='./datasetgen/object_templates',
    help="path to the scene object templates")
parser.add_argument('--rotate_objects', default=True,
    help="ramdomly rotate objects when placing them in the scene")
parser.add_argument('--near_distance', default=50,
    help="ramdomly rotate objects when placing them in the scene")
args = parser.parse_args()

config = {
    'template_path' : args.template_path,
    'near_distance' : args.near_distance
}
SceneObject.class_init(config)

scene = Scene((args.image_width, args.image_height))
rotation=0
if args.rotate_objects:
    rotation=None
for i in range(args.object_count):
    so = scene.add_object(rotation=rotation)
    if so is None:
        raise ValueError("Couldn't place an object (too crowded?).")

print(scene.mobile_objects)

# size of image
canvas = (scene.resolution[0], scene.resolution[1])
# draw the image
im = Image.new('RGBA', canvas, (255, 255, 255, 255))
draw = ImageDraw.Draw(im)    
scene.render(draw, withindexes=True)

fname = 'img_test'
im.save('{}/{}.png'.format(args.output_path, fname))

# save the scene as a json file
scene_json = Scene.JSONEncoder(indent=4).encode(scene)
with open('{}/{}.json'.format(args.output_path, fname), 'w') as outfile:
    outfile.write(scene_json)

# demo highlight objects
# im2 = Image.new('RGBA', canvas)
# im3 = Image.blend(im, im2, 0.5)
# draw = ImageDraw.Draw(im3) 
# for o in scene.mobile_objects:
#     if o.scene_object_type == 'cup':
#         o.draw(draw)
# im3.save('{}/{}.png'.format(args.output_path, fname+'_select'))

with open('datasetgen/production_rules.json', 'r') as prfile:
    production_rules = json.load(fp=prfile, object_hook=ProductionRule.from_dict)
gen = CFGGenerator(scene, production_rules)
selections = gen.expand_symbol('_Select')
for s in selections:
    print(s.to_string())


