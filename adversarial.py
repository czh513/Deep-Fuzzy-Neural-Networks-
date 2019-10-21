from cleverhans.attacks import SaliencyMapMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.train import train
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from cleverhans.model import CallableModelWrapper
from cleverhans.attacks import CarliniWagnerL2
import numpy as np
import tensorflow as tf
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import seaborn as sns

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: %s" % device)

class AdversarialExperiment(object):

    def __init__(self, attack, params, test_data, batch_size=100, verbose=False):
        self.attack = attack
        self.params = params
        self.test_data = test_data
        self.batch_size = batch_size
        self.verbose = verbose

    def evaluate_model(self, model_path, num_batches=-1):
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            print('Evaluating model: ' + model_path)

            torch_model_orig = torch.load(model_path).to(device)
            torch_model = lambda x: torch_model_orig(x)[0] # to standard format
            tf_model_fn = convert_pytorch_model_to_tf(torch_model, out_dims=10)    
            cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')

            test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

            x_test_sample, _ = next(iter(test_loader)) # to get the shape of the input
            nchannels, img_rows, img_cols = x_test_sample.shape[1:]
            x = tf.placeholder(tf.float32, shape=(None, nchannels, img_rows, img_cols))
            attack_model = self.attack(cleverhans_model, sess=sess)
            preds_op = tf_model_fn(x)

            # Run an evaluation of our model against fgsm
            self.saved_xs, self.saved_advs, self.saved_ys, \
                self.saved_adv_preds, self.saved_clean_preds = [], [], [], [], []
            accuracies = []
            try:
                for batch_no, (xs, ys) in enumerate(test_loader):
                    ys_one_hot = torch.nn.functional.one_hot(ys, 10)
                    if self.attack == CarliniWagnerL2:
                        self.params['y'] = ys_one_hot.numpy()
                    # using generate_np() or generate() leads to similar performance
                    # not sure if the GPU is fully utilized...
                    advs = attack_model.generate_np(xs.numpy(), **self.params)

                    adv_preds = sess.run(preds_op, feed_dict={x: advs})
                    clean_preds = sess.run(preds_op, feed_dict={x: xs})
                    correct = (np.argmax(adv_preds, axis=1) == ys.numpy()).sum()
                    total = test_loader.batch_size

                    self.saved_xs.append(xs)
                    self.saved_ys.append(ys)
                    self.saved_advs.append(advs)
                    self.saved_adv_preds.append(adv_preds)
                    self.saved_clean_preds.append(clean_preds)
                    accuracies.append(correct / total)
                    if self.verbose:
                        print('Batch: #%d, accuracy: %.2f, std: %.2f' %(batch_no, np.mean(accuracies), np.std(accuracies)))

                    if num_batches > 0 and batch_no+1 >= num_batches: break
            except KeyboardInterrupt:
                print('Evaluation aborted')
            self._process_saved_info()
            print('Accuracy under attack: %.2f (std=%.2f)' 
                  %(np.mean(accuracies), np.std(accuracies)))
            return accuracies

    def _process_saved_info(self):
        self.saved_xs = np.vstack(self.saved_xs)
        self.saved_advs = np.vstack(self.saved_advs)
        self.saved_ys = np.concatenate(self.saved_ys)
        self.saved_adv_preds = np.vstack(self.saved_adv_preds)
        self.saved_clean_preds = np.vstack(self.saved_clean_preds)
        self.saved_diff = (self.saved_advs-self.saved_xs)
        xs, d = self.saved_xs, self.saved_diff
        self.saved_diff_norm = np.linalg.norm(d.reshape(xs.shape[0], xs.size//xs.shape[0]), ord=2, axis=1)

    def plot_example2(self, i = None):
        nonzero_diff_indices, = self.saved_diff_norm.nonzero()
        i = i or np.random.choice(len(nonzero_diff_indices))
        idx = nonzero_diff_indices[i]
        fig, ax = plt.subplots(ncols=5, facecolor=(1, 1, 1), figsize=(15, 3))
        ax[0].title.set_text('Perturbation')
        ax[0].imshow(self.saved_diff[idx,0], vmin=-0.5, vmax=0.5)
        ax[1].title.set_text('Clean image')
        ax[1].imshow(self.saved_xs[idx,0], vmin=0, vmax=1)
        ax[2].title.set_text('Prediction on clean')
        ax[2].bar(np.arange(10), self.saved_clean_preds[idx])
        ax[3].title.set_text('Adversarial image')
        ax[3].imshow(self.saved_advs[idx,0], vmin=0, vmax=1)
        ax[4].title.set_text('Prediction on adv')
        ax[4].bar(np.arange(10), self.saved_adv_preds[idx])
        print('Perturbation strength (L2): %.1f' % self.saved_diff_norm[idx])
