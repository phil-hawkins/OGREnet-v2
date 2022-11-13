# Object Graph Relation Encoding Network 
# (ORGE-net) - version 2

Consider a domestic robot being asked to pick up ``the cup nearest to the plate''. Natural language is an intuitive way for humans to interact with robots. However, enabling robots to comprehend natural language spatial instructions such as this is challenging because objects in the scene must be identified by a combination of their type, cup, and their relation to other objects, 'nearest to the plate'.

Specifically, we define a task involving an image of a table setting and a phrase that contains a spatial relationship, such as ``the cup nearest to the plate''. The system should respond with a reference to the selected object in a object model built from the image. 

<img src="images/table_objects.png"
     alt="Search heuristic network"
     style="float: margin-bottom: 100px;" />

This problem is a complex multi-modal task. It requires a Neural Network model to ground relational language jointly in text and image data. There are no cues such as texture and colour to help the model detect a spatial relationship like 'nearest'. Furthermore, in contrast to the target labels of object detection datasets, spatial relationships are abstract concepts that do not readily fit the paradigm of supervised training targets.

A challenge we address the in the model is that an image may contain an arbitrary number of objects. The model must therefore scale dynamically with no fixed upper bound on the number of objects.

The model is trained on scenes with a small number of items, however it is capable of performing inference on more complex scenes than those encountered during training. This is a key benefit in learning relationship features. They enable a complex structure, such as a scene of a table setting, to be processed as a composite of related components. Rather than select a fixed number of features to represent a scene model, the graph representation treats the scene as an arbitrary sized collection of binary object relationships.

## Setting up OGRE-net

### Python dependencies


`pip install -r requirements.txt`

### InferSent

See https://github.com/phil-hawkins/InferSent for InferSent setup instructions and pretrained weights.

### Generating synthetic training data

`python datasetgen/kitchen/generate.py`

### Training OGER-net

`python train.py`

### Running evaluations

`python int_eval.py`