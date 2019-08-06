from absl import app, flags, logging
from absl.flags import FLAGS
import torch

from nlp.InferSent.models import InferSent
from datasetgen.kitchen.data import generate_scene, Config, gen_scene_example
from datasetgen.kitchen.visualise import show_predictions
from datasetgen.kitchen.scene import SceneObject
from ogrenetv2.model import OGRENet

flags.DEFINE_string('weights', './checkpoints/99999.chk', 'Model parameters to use')

def main(_argv):
    # initialise configs
    config = Config()
    SceneObject.class_init(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OGRENet().to(device)
    #with open(FLAGS.weights, "r") as f:
    checkpoint = torch.load(FLAGS.weights, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # setup sentence embedding model
    infersent = InferSent(config.params_model)
    infersent.load_state_dict(torch.load(config.MODEL_PATH))
    infersent.set_w2v_path(config.W2V_PATH)
    infersent.build_vocab(config.vocab, tokenize=True)

    next_ex = True
    while next_ex:
        # get an example
        scene = generate_scene(config)
        data, selection_text = gen_scene_example(config=config, scene=scene, encode_fn=infersent.encode)

        # run inference
        out = model(data)
        sig = out.sigmoid()
        pred = sig.clone()
        pred[sig >= .5] = 1.
        pred[sig < .5] = 0.
        tp = (pred == 1.) & (data.y == 1.)
        correct = [i for i in range(tp.numel()) if tp[i]]
        fp = (pred == 1.) & (data.y == 0.)
        incorrect = [i for i in range(fp.numel()) if fp[i]]

        # display result
        print(selection_text)
        print("   Class       T    P    S      R")
        for i in range(out.numel()):
            
            print("{0: <2} {1: <10} {2: .1f} {3: .1f} {4: .4f} {5: .4f}".format(
                i, 
                scene.mobile_objects[i].scene_object_type, 
                data.y[i],
                pred[i],
                sig[i],
                out[i]
            ))
        next_ex = show_predictions(scene, correct, incorrect)

if __name__ == '__main__':
    app.run(main)