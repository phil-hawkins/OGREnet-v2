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

flags.DEFINE_integer('train_batches', 200, 'number of training batches')
flags.DEFINE_integer('eval_batches', 20, 'number of evaluation batches')
flags.DEFINE_integer('log_step', 10, 'number of training batches')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_string('job_id', 'testrun', 'job identifier from the batch system.  Used in tagging the logs')


def git_record():
    git_msg = subprocess.check_output(['git', 'log', '-n 1 --pretty=format:%s HEAD'])
    return git_msg.decode("utf-8")


def main(_argv):
    writer = SummaryWriter(log_dir="./logs/{}".format(FLAGS.job_id))
    writer.add_text(tag="git record", text_string=git_record())
    config = Config()
    SceneObject.class_init(config)

    # setup sentence embedding
    infersent = InferSent(config.params_model)
    infersent.load_state_dict(torch.load(config.MODEL_PATH))
    infersent.set_w2v_path(config.W2V_PATH)
    infersent.build_vocab(config.vocab, tokenize=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OGRENet().to(device)
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
        meter.add_batch_results(out, data.y, loss)

        # log metrics to tensorboard
        if (batch > 0) and (batch % FLAGS.log_step == 0):
            logging.info("Batch: {}/{} loss:{}, acc:{}, prec:{}, rec:{}".format(
                batch, FLAGS.train_batches, 
                meter.loss(), meter.accuracy(), meter.precision(), meter.recall()))
            writer.add_scalar("loss", meter.loss(), global_step=batch)
            writer.add_scalar("accuracy", meter.accuracy(), global_step=batch)
            writer.add_scalar("precision", meter.precision(), global_step=batch)
            writer.add_scalar("recall", meter.recall(), global_step=batch)
            meter = Meter()


    # evaluation
    # model.eval()
    # acc = 0.
    # for batch in range(FLAGS.eval_batches):
    #     data = gen_data_batch(config=config, encode_fn=infersent.encode).to(device)
    #     pred = model(data)
    #     pred[pred >= .5] = 1.
    #     pred[pred < .5] = 0.
    #     acc += (accuracy(pred, data.y) / FLAGS.eval_batches)
    # logging.info('Accuracy: {:.4f}'.format(acc))






if __name__ == '__main__':
    app.run(main)