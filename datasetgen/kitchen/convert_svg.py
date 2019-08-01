# Converts an Inkscape SVG to a PIL path
# there are lots of wild assumptions about the SVG file structure

from lxml import etree
import json
import argparse
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--scene_object', default='cup',
    help="name of the table setting object")
parser.add_argument('--output_path', default='./scene_generation/object_templates',
    help="path to the output directory")
args = parser.parse_args()

print(args)

# get the polygon path from the SVG file
parser = etree.XMLParser(ns_clean=True)
tree = etree.parse("{}/{}.svg".format(args.output_path, args.scene_object))
root = tree.getroot()
find_g = etree.XPath("//svg:svg/svg:g", namespaces={'svg': 'http://www.w3.org/2000/svg'})
find_paths = etree.XPath("//svg:svg/svg:g/svg:path", namespaces={'svg': 'http://www.w3.org/2000/svg'})
g = find_g(root)[0]
p = find_paths(root)[0]
path_str = p.get('d')
transform = g.get('transform')
translate = list(map(float, transform[10:-1].split(',')))
pl = path_str.split(' ')[1:-1]

# convert from relative to absolute coordinates
path = []
for i in range(len(pl)):
    node = list(map(float, pl[i].split(',')))
    if i==0:
        node[0] += translate[0]
        node[1] += translate[1]
    else:
        node[0] += path[i-1][0]
        node[1] += path[i-1][1]    
    path.append(tuple(node))

# output the path list as a json file
with open('{}/{}.json'.format(args.output_path, args.scene_object), 'w') as outfile:
    json.dump(path, outfile)

# size of image
canvas = (100, 100)
# draw the image
im = Image.new('RGBA', canvas, (255, 255, 255, 255))
draw = ImageDraw.Draw(im)    
draw.polygon(path, fill='grey')
im.save('{}/{}.png'.format(args.output_path, args.scene_object))