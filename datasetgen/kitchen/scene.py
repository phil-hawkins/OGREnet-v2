import random
from PIL import Image, ImageDraw, ImageFont
import json
import os
import copy
import math
import numpy as np
from shapely.geometry import Polygon, MultiPoint, LineString, Point
import shapely.affinity
import shapely.ops
import shapely.wkt


class SceneObject(object):
    colours = {'cup':'aqua', 'plate':'white', 'fork':'grey', 'knife':'grey', 'spoon':'grey'}
    polygons = {}
    template_path = ''
    near_distance = None
    EXT_X1 = 0
    EXT_Y1 = 1
    EXT_X2 = 2
    EXT_Y2 = 3

    class JSONEncoder(json.JSONEncoder):
        def default(self, scene_object):
            assert(type(scene_object) is SceneObject)

            scene_object_dict = {
                'colour' : scene_object.colour,
                'rotation' : scene_object.rotation,
                'scene_object_type' : scene_object.scene_object_type,
                'shape' : shapely.wkt.dumps(scene_object.shape)
            }

            return scene_object_dict

    @classmethod
    def from_dict(cls, dict):
        scene_object = SceneObject(scene_object_type=dict['scene_object_type'], colour=dict['colour'])
        scene_object.rotation = dict['rotation']
        scene_object.shape = shapely.wkt.loads(dict['shape'])
        scene_object.extents = SceneObject.calculate_extents(scene_object.shape)

        return scene_object

    @classmethod
    def class_init(cls, config):
        template_path = config.TEMPLATE_PATH
        cls.near_distance = config.NEAR_DISTANCE

        _, _, filenames = next(os.walk(template_path))
        for f in filenames:
            filename, file_extension = os.path.splitext(f)
            if file_extension == '.json':
                with open('{}/{}'.format(template_path, f), 'r') as polyfile:
                    path = json.load(polyfile)
                cls.polygons[filename] = Polygon([tuple(c) for c in path])
  
    @classmethod
    def calculate_extents(cls, shape):
        return shape.bounds

    def __init__(self, scene_object_type, colour='grey', table_extents=None, scale=1.):
        self.colour = colour
        self.rotation = 0       
        self.rotation_is_meaningful = scene_object_type not in ['plate', 'cup', 'center']
        self.scene_object_type = scene_object_type

        if scene_object_type == 'left edge':
            assert(table_extents is not None)
            self.shape = LineString([(table_extents[self.EXT_X1], table_extents[self.EXT_Y1]), (table_extents[self.EXT_X1], table_extents[self.EXT_Y2])])
        elif scene_object_type == 'right edge':
            assert(table_extents is not None)
            self.shape = LineString([(table_extents[self.EXT_X2], table_extents[self.EXT_Y1]), (table_extents[self.EXT_X2], table_extents[self.EXT_Y2])])
        elif scene_object_type == 'top edge':
            assert(table_extents is not None)
            self.rotation = 90
            self.shape = LineString([(table_extents[self.EXT_X1], table_extents[self.EXT_Y1]), (table_extents[self.EXT_X2], table_extents[self.EXT_Y1])])
        elif scene_object_type == 'bottom edge':
            assert(table_extents is not None)
            self.rotation = 90
            self.shape = LineString([(table_extents[self.EXT_X1], table_extents[self.EXT_Y2]), (table_extents[self.EXT_X2], table_extents[self.EXT_Y2])])
        elif scene_object_type == 'center':
            assert(table_extents is not None)
            self.shape = Point((table_extents[self.EXT_X1]+table_extents[self.EXT_X2])/2, (table_extents[self.EXT_Y1]+table_extents[self.EXT_Y2])/2)
        elif scene_object_type == 'cup':
            self.shape = shapely.affinity.scale(SceneObject.polygons[scene_object_type], xfact=1.8*scale, yfact=1.8*scale)
        else:
            self.shape = shapely.affinity.scale(SceneObject.polygons[scene_object_type], xfact=scale, yfact=scale)
        self.extents = SceneObject.calculate_extents(self.shape)



    def move(self, offset, absolute=False):
        if absolute:           
            offset = (offset[0] - self.extents[self.EXT_X1], offset[1] - self.extents[self.EXT_Y1])
        self.shape = shapely.affinity.translate(self.shape, offset[0], offset[1])
        self.extents = SceneObject.calculate_extents(self.shape)

    def rotate(self, rotation):
        self.shape = shapely.affinity.rotate(self.shape, rotation, origin='center')
        self.extents = SceneObject.calculate_extents(self.shape)
        self.rotation = (self.rotation + rotation) % 360

    # gets the ceter of the bounding box
    def get_center(self):
        return tuple(((self.extents[0]+self.extents[2])//2, (self.extents[1]+self.extents[3])//2))

    # gets the centroid of the shape as a numpy array
    def centroid(self):
        return self.shape.centroid

    def distance_to(self, other_so):
        return self.shape.distance(other_so.shape)

    def draw(self, draw, outline="black", fill=None):
        if fill is None:
            fill = self.colour
        draw.polygon(self.shape.exterior.coords, fill=fill, outline=outline)


    # Predicates
    @classmethod
    def is_scene_object_type(cls, scene_object, args):
        return scene_object.scene_object_type == args[0]

    @classmethod
    def is_related(cls, scene_object, args):      
        relationship = args[1]
        other = args[0]

        if relationship == 'left':
            return cls.is_left(scene_object, args)
        if relationship == 'right':
            return cls.is_right(scene_object, args)
        if relationship == 'above':
            return cls.is_above(scene_object, args)
        if relationship == 'below':
            return cls.is_below(scene_object, args)
        if relationship == 'intersects':
            return cls.intersects_pred(scene_object, args)
        if relationship == 'parallel':
            return scene_object.is_parallel(other)
        if relationship == 'orthogonal':
            return scene_object.is_orthogonal(other)
        if relationship == 'near':
            return scene_object.is_withindistance(other, cls.near_distance)

        return False

    @classmethod
    def is_left(cls, scene_object, args):
        other_so = args[0]
        return scene_object.extents[cls.EXT_X2] <= other_so.extents[cls.EXT_X1]

    @classmethod
    def is_right(cls, scene_object, args):
        other_so = args[0]
        return scene_object.extents[cls.EXT_X1] >= other_so.extents[cls.EXT_X2]

    @classmethod
    def is_above(cls, scene_object, args):
        other_so = args[0]
        return scene_object.extents[cls.EXT_Y2] <= other_so.extents[cls.EXT_Y1]

    @classmethod
    def is_below(cls, scene_object, args):
        other_so = args[0]
        return scene_object.extents[cls.EXT_Y1] >= other_so.extents[cls.EXT_Y2]

    @classmethod
    def scene_compare_pred(cls, scene_object, args):
        comparison_scene_object = args[0]
        rel = args[1]
        other_scene_objects = args[2]

        if rel == 'nearest':
            return scene_object.is_nearest(comparison_scene_object, other_scene_objects)
        elif rel == 'furthest':
            return scene_object.is_farthest(comparison_scene_object, other_scene_objects)
        else:
            # should be one of the above
            assert(False) 

    def is_nearest(self, other_so, other_so_set):
        dist = self.distance_to(other_so) 
        for other_so in other_so_set:
            if (self is not other_so) and (self.distance_to(other_so) < dist):
                return False
        return True

    def is_farthest(self, other_so, other_so_set):
        dist = self.distance_to(other_so) 
        for other_so in other_so_set:
            if (self is not other_so) and (self.distance_to(other_so) > dist):
                return False
        return True

    @classmethod
    def intersects_pred(cls, scene_object, args):
        other_so = args[0]
        return scene_object.intersects(other_so)

    def intersects(self, other):
        return self.shape.intersects(other.shape)

    @classmethod
    def is_aligned_pred(cls, scene_object, args):
        other_so = args[0]
        alignment_type = args[1]
        return scene_object.is_aligned(other_so, alignment_type)

    def is_aligned(self, other_so, alignment_type):
        alignment_margin = 1.0

        if alignment_type == 'left':
            return abs(self.extents[self.EXT_X1] - other_so.extents[self.EXT_X1]) < alignment_margin
        elif alignment_type == 'right':
            return abs(self.extents[self.EXT_X2] - other_so.extents[self.EXT_X2]) < alignment_margin
        elif alignment_type == 'top':
            return abs(self.extents[self.EXT_Y1] - other_so.extents[self.EXT_Y1]) < alignment_margin
        elif alignment_type == 'bottom':
            return abs(self.extents[self.EXT_Y2] - other_so.extents[self.EXT_Y2]) < alignment_margin
        elif alignment_type == 'horizontal center':
            c1 = self.get_center()
            c2 = other_so.get_center()
            return abs(c1[1] - c2[1]) < alignment_margin
        elif alignment_type == 'vertical center':
            c1 = self.get_center()
            c2 = other_so.get_center()
            return abs(c1[0] - c2[0]) < alignment_margin

        return False

    @classmethod
    def is_orthogonal_pred(cls, scene_object, args):
        other_so = args[0]
        return scene_object.is_orthogonal(other_so)

    def is_orthogonal(self, other_so):
        if (self.rotation_is_meaningful and other_so.rotation_is_meaningful):
            rotational_diffence = abs((self.rotation - other_so.rotation) % 180)
            return (rotational_diffence == 90)
        else:
            return False
        
    @classmethod
    def is_parallel_pred(cls, scene_object, args):
        other_so = args[0]
        return scene_object.is_parallel(other_so)

    def is_parallel(self, other_so):
        if (self.rotation_is_meaningful and other_so.rotation_is_meaningful):
            rotational_diffence = abs((self.rotation - other_so.rotation) % 180)
            return (rotational_diffence == 0)
        else:
            return False

    @classmethod
    def is_between_pred(cls, scene_object, args):
        other_so1 = args[0]
        other_so2 = args[1]
        return scene_object.is_between(other_so1, other_so2)

    def is_between(self, other_so1, other_so2):
        # First attempt gives some unexpected results
        # union = shapely.ops.unary_union([other_so1.shape, other_so2.shape])
        # return self.shape.intersects(union.convex_hull)

        union = shapely.ops.unary_union([other_so1.shape, other_so2.shape])
        return self.shape.centroid.within(union.convex_hull)

    @classmethod
    def is_touching_pred(cls, scene_object, args):
        other_so = args[0]
        return scene_object.is_touching(other_so)

    def is_touching(self, other_so):
        return self.shape.touches(other_so.shape)

    @classmethod
    def is_withindistance_pred(cls, scene_object, args):
        other_so = args[0]
        distance = args[1]
        return scene_object.is_touching(other_so, distance)

    def is_withindistance(self, other_so, distance):
        d = self.shape.distance(other_so.shape)
        return d <= distance

class Scene(object):

    class JSONEncoder(json.JSONEncoder):
        def default(self, scene):
            assert(type(scene) is Scene)

            so_encoder = SceneObject.JSONEncoder(indent=self.indent)
            mobile_objects = [so_encoder.default(so) for so in scene.mobile_objects]
            scene_dict = {
                'resolution' : scene.resolution,
                'table_extents' : scene.table_extents,
                'mobile_objects' : mobile_objects
            }

            return scene_dict

    class JSONDecoder(json.JSONDecoder):
        def __init__(self, *args, **kwargs):
            kwargs['object_hook'] = self.object_hook
            super(Scene.JSONDecoder, self).__init__(*args, **kwargs)

        def object_hook(self, obj):
            if isinstance(obj, dict) and 'resolution' in obj:
                scene = Scene(obj['resolution'])
                scene.table_extents = obj['table_extents']
                scene.mobile_objects = [SceneObject.from_dict(sod) for sod in obj['mobile_objects']]
                return scene
            else:
                return obj

    def __init__(self, resolution, table_margin=10, placement_retrys=5):
        self.resolution = resolution
        self.table_extents = (table_margin, table_margin, resolution[0] - table_margin, resolution[1] - table_margin)
        self.table_locations = {
            'left edge' : SceneObject('left edge', table_extents=self.table_extents),
            'right edge' : SceneObject('right edge', table_extents=self.table_extents),
            'top edge' : SceneObject('top edge', table_extents=self.table_extents),
            'bottom edge' : SceneObject('bottom edge', table_extents=self.table_extents),
            'center' : SceneObject('center', table_extents=self.table_extents)
        }
        self.mobile_objects = []
        self.placement_retrys = placement_retrys

    def all_objects(self):
        return self.mobile_objects + list(self.table_locations.values())

    def add_object(self, scene_object_type=None, position=None, colour=None, rotation=None, scale=1.):
        if scene_object_type == None:
            scene_object_type = random.choice(list(SceneObject.polygons))

        if colour == None:
            colour = SceneObject.colours[scene_object_type]

        if rotation == None:
            rotation = random.randint(0, 24) * 15

        place_attempt = 0
        if position == None:
            so = SceneObject(scene_object_type=scene_object_type, colour=colour, scale=scale)
            so.rotate(rotation)
            #so.move((so.extents[0]*-1, so.extents[1]*-1))
            intersects = True
            position = [0,0]
            width = int(math.ceil(so.extents[2]))
            height = int(math.ceil(so.extents[3]))
            
            while intersects and (place_attempt <= self.placement_retrys):
                place_attempt += 1
                x = random.randint(self.table_extents[0], self.table_extents[2]-width)
                y = random.randint(self.table_extents[1], self.table_extents[3]-height)
                so.move((x, y), absolute=True)
                intersects = self.intersects_any(so)
        else:
            so = SceneObject(scene_object_type, colour, scale=scale)
            so.rotate(rotation)
            so.move(position, absolute=True)

        self.mobile_objects.append(so)

        if place_attempt > self.placement_retrys:
            return None
        else:
            return so

    def render(self, draw, withindexes=False, highlight=[]):
        draw.rectangle(self.table_extents, fill='sienna')
        for idx, o in enumerate(self.mobile_objects):
            outline = 'green' if idx in highlight else 'black'
            fill = 'green' if idx in highlight else None
            o.draw(draw, outline=outline, fill=fill)
            if withindexes:
                draw.text(o.get_center(),str(idx),fill='black')

    def intersects_any(self, scene_object):
        intersects = False
        for other in self.mobile_objects:
            if scene_object.intersects(other):
                intersects = True
        return intersects

    # get the indexes of the objects that match the predicate
    # input_set: a list of self.mobile_objects indexes to filter
    def get_selection(self, input_set, predicate, args):
        return [i for i in input_set if predicate(self.all_objects()[i], args)]

    def mask(self, object_indexes):
        # render a mask of the selected objects
        im = Image.new('1', tuple(self.resolution))
        draw = ImageDraw.Draw(im)
        for idx in object_indexes:
            self.mobile_objects[idx].draw(draw, outline='White')
        mask = np.array(im)

        return mask




