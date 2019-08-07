from absl import app, flags, logging
from absl.flags import FLAGS
import torch
import torch_geometric.utils as tg_utils
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import subprocess

from nlp.InferSent.models import InferSent
from datasetgen.kitchen.data import gen_data_batch, Config
from datasetgen.kitchen.scene import SceneObject
from ogrenetv2.model import OGRENet
from ogrenetv2.meter import Meter


flags.DEFINE_integer('log_step', 10, 'number of training batches')
flags.DEFINE_string('job_id', 'testrun', 'job identifier from the batch system.  Used in tagging the logs')
flags.DEFINE_string('notes', '', 'experiment notes to log')
# Hyper-paramteres
flags.DEFINE_integer('train_batches', 200, 'number of training batches')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_integer('selection_sz', 4096, 'size of selection sentence encoding vector = u_attr_sz')
flags.DEFINE_integer('u_attr_reduced_sz', 256, 'size of u_attr_sz after dimention reduction')
flags.DEFINE_integer('edge_h_sz', 1024, 'size of hidden layers in the edge model')
flags.DEFINE_integer('edge_attr_sz1', 512, 'edge attribute output')
flags.DEFINE_integer('node_h_sz', 512, 'size of hidden layers in the node model')
flags.DEFINE_integer('edge_hidden_layers', 3, 'number of hidden layers in the edge model')
flags.DEFINE_integer('node_hidden_layers', 1, 'number of hidden layers in the node model')

def git_record():
    git_msg = subprocess.check_output(['git', 'log', '-n 1 --pretty=format:%s HEAD'])
    return git_msg.decode("utf-8")

def flags_text():
    ignore_flags = [
        '?',
        'alsologtostderr',
        'help',
        'helpfull',
        'helpshort',
        'helpxml',
        'logtostderr',
        'log_dir',
        'only_check_args',
        'pdb_post_mortem',
        'profile_file',
        'run_with_pdb',
        'run_with_profiling',
        'showprefixforinfo',
        'stderrthreshold',
        'use_cprofile_for_profiling',
        'v',
        'verbosity'
    ]
    flag_lst = ["{} : {}".format(flag, FLAGS[flag].value) for flag in FLAGS if flag not in ignore_flags]
    
    return "\n".join(flag_lst)

def run_settings():
    return "{}\n{}".format(flags_text(), git_record())


def main(_argv):
    writer = SummaryWriter(log_dir="./logs/{}".format(FLAGS.job_id))
    writer.add_text(tag="Run settings", text_string=run_settings())
    config = Config()
    SceneObject.class_init(config)

    # setup sentence embedding
    infersent = InferSent(config.params_model)
    infersent.load_state_dict(torch.load(config.MODEL_PATH))
    infersent.set_w2v_path(config.W2V_PATH)
    infersent.build_vocab(config.vocab, tokenize=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OGRENet(
        u_attr_sz=FLAGS.selection_sz, 
        u_attr_reduced_sz=FLAGS.u_attr_reduced_sz, 
        edge_h_sz=FLAGS.edge_h_sz, 
        edge_attr_sz1=FLAGS.edge_attr_sz1, 
        node_h_sz=FLAGS.node_h_sz, 
        edge_hidden_layers=FLAGS.edge_hidden_layers, 
        node_hidden_layers=FLAGS.node_hidden_layers
    ).to(device)
    #data = dataset[0].to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate, weight_decay=5e-4)
    meter = Meter()
    pos_weight = (config.MAX_OBJECTS + config.MIN_OBJECTS) / 2.

    # training
    model.train()
    for batch in range(FLAGS.train_batches):
        optimizer.zero_grad()
        data = gen_data_batch(config=config, encode_fn=infersent.encode).to(device)
        out = model(data)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.ones_like(out) * pos_weight)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        meter.add_batch_results(out, data.y, loss, data.batch)

        # log metrics to tensorboard
        if (batch > 0) and (batch % FLAGS.log_step == 0):
            logging.info("Batch: {}/{} loss:{}, acc:{}, prec:{}, rec:{}, top1:{}".format(
                batch, FLAGS.train_batches, 
                meter.loss(), meter.accuracy(), meter.precision(), meter.recall(), meter.top1_precision()))
            writer.add_scalar("loss", meter.loss(), global_step=batch)
            writer.add_scalar("accuracy", meter.accuracy(), global_step=batch)
            writer.add_scalar("precision", meter.precision(), global_step=batch)
            writer.add_scalar("recall", meter.recall(), global_step=batch)
            writer.add_scalar("top1_precision", meter.top1_precision(), global_step=batch)
            meter = Meter()


    torch.save({
        'epoch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }, "./logs/{}/{}.chk".format(FLAGS.job_id, batch))





if __name__ == '__main__':
    app.run(main)