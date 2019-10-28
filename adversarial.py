from cleverhans.attacks import CarliniWagnerL2, SPSA, FastGradientMethod, BasicIterativeMethod, MaxConfidence
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils_pytorch import convert_pytorch_model_to_tf
from cleverhans.model import CallableModelWrapper
import numpy as np
import tensorflow as tf
import torch
from torchvision import datasets, transforms
from scipy.special import softmax
import json
import sys
from time import time
from train import cifar_stats

# if you have more than one GPU on the same machine, it's important to specify 
# the device number to force pytorch and tensorflow to the same GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device: %s" % device, file=sys.stderr)

class AdversarialExperiment(object):

    def __init__(self, attack, params, test_data, batch_size, report_interval=20):
        self.attack = attack
        self.params = {**params} # avoid modifying the dict
        self.test_data = test_data
        self.batch_size = batch_size
        self.report_interval = report_interval

    def evaluate_model(self, model_path, num_batches=-1, model_device=None):
        model_device = model_device or device
        start_sec = time()
        with tf.Session() as sess:
            torch_model_orig = torch.load(model_path, map_location=torch.device('cpu')).to(model_device)
            torch_model = lambda x: torch_model_orig(x.to(model_device))[0].to(device) # [0]: convert to standard format
            tf_model_fn = convert_pytorch_model_to_tf(torch_model, out_dims=10)    
            cleverhans_model = CallableModelWrapper(tf_model_fn, output_layer='logits')
            # fix error with SPSA: "ValueError: Tried to convert 'depth' to a tensor and 
            # failed. Error: None values not supported."
            cleverhans_model.nb_classes = 10 

            # important to shuffle the data since we'll measure standard deviation
            test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)

            x_test_sample, _ = next(iter(test_loader)) # to get the shape of the input
            nchannels, img_rows, img_cols = x_test_sample.shape[1:]
            x = tf.placeholder(tf.float32, shape=(None, nchannels, img_rows, img_cols))
            y = tf.placeholder(tf.int32, shape=(None,))
            attack_model = self.attack(cleverhans_model, sess=sess)
            clean_preds_op = tf_model_fn(x)
            preds_op = tf_model_fn(x)
            # # to use generate() instead of generate_np()
            # self.params['y'] = y
            # advs = attack_model.generate(x, **self.params)
            # adv_preds_op = tf_model_fn(advs)

            # Run an evaluation of our model against fgsm
            self.saved_xs, self.saved_advs, self.saved_ys, \
                self.saved_adv_preds, self.saved_clean_preds = [], [], [], [], []
            accuracies = []
            try:
                for batch_no, (xs, ys) in enumerate(test_loader):
                    if self.attack == SPSA:
                        self.params['y'] = ys.numpy().astype(np.int32)
                    else:
                        ys_one_hot = torch.nn.functional.one_hot(ys, 10).numpy()
                        if self.attack == MaxConfidence:
                            self.params['y'] = ys_one_hot.astype(np.float32)
                        else:
                            self.params['y'] = ys_one_hot.astype(np.int32)
                    # using generate_np() or generate() leads to similar performance
                    # not sure if the GPU is fully utilized...
                    advs = attack_model.generate_np(xs.numpy(), **self.params)

                    adv_preds = sess.run(preds_op, feed_dict={x: advs})
                    clean_preds = sess.run(preds_op, feed_dict={x: xs})
                    # clean_preds, adv_preds = sess.run([clean_preds_op, adv_preds_op],
                    #                                   feed_dict={x: xs.numpy(), y: ys.numpy()})
                    correct = (np.argmax(adv_preds, axis=1) == ys.numpy()).sum()
                    total = test_loader.batch_size

                    self.saved_xs.append(xs)
                    self.saved_ys.append(ys)
                    self.saved_advs.append(advs)
                    self.saved_adv_preds.append(adv_preds)
                    self.saved_clean_preds.append(clean_preds)
                    accuracies.append(correct / total)
                    if self.report_interval > 0 and batch_no % self.report_interval == 0:
                        elapsed_sec = time() - start_sec
                        print('Batch: #%d, accuracy: %.2f, std: %.2f, %.1f secs/batch' 
                              %(batch_no, np.mean(accuracies), np.std(accuracies),
                                elapsed_sec / (batch_no+1)), file=sys.stderr)

                    if num_batches > 0 and batch_no+1 >= num_batches: break
            except KeyboardInterrupt:
                print('Evaluation aborted', file=sys.stderr)
            self._process_saved_info()
            print('Accuracy under attack: %.2f (std=%.2f)' 
                  %(np.mean(accuracies), np.std(accuracies)), file=sys.stderr)
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

    def extract_max_probs_on_perturbed_examples(self):
        perturbed_examples = np.logical_not(np.isclose(self.saved_diff_norm, np.zeros_like(self.saved_diff_norm)))
        print('%d examples of %d were perturbed' % (perturbed_examples.sum(), len(perturbed_examples)), file=sys.stderr)
        logits = self.saved_adv_preds[perturbed_examples]
        return np.max(softmax(logits, axis=1), axis=1)

    def plot_example(self, i = None):
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
        print('Perturbation strength (L2): %.1f' % self.saved_diff_norm[idx], file=sys.stderr)

def run(attack=None, model_path=None, model_device=None, dataset=None, batch_size=100, 
                   normalize_data=True, json_out_path=True, num_batches=-1, report_interval=20, **kwargs):
    '''
    Default params are set based on this paper as much as possible:
    Taghanaki, S. A., Abhishek, K., Azizi, S., & Hamarneh, G. (2019). A Kernelized Manifold Mapping 
    to Diminish the Effect of Adversarial Perturbations, 11340–11349. 
    Retrieved from http://arxiv.org/abs/1903.01015
    '''
    if attack == 'BIM':
        attack_func = BasicIterativeMethod
        default_attack_params = {
            'ord': np.inf, 'eps': 0.3, 
            'nb_iter': 5, 'eps_iter': .1 
        }
    elif attack == 'CW':
        attack_func = CarliniWagnerL2
        default_attack_params = {
            'binary_search_steps': 1, # tried to put 5 here but it runs too slowly
            'max_iterations': 50, 'learning_rate': .5,
            'batch_size': batch_size, 'initial_const': 1}
    elif attack == "FGM_inf":
        attack_func = FastGradientMethod
        default_attack_params = {'ord': np.inf, 'eps': 0.3}
    elif attack == "FGM_L2":
        attack_func = FastGradientMethod
        default_attack_params = {
            'ord': 2, 
            'eps': 2 # setting this to 3 leads to internal CUDA error
        }
    elif attack == "SPSA":
        attack_func = SPSA
        default_attack_params = {
            'eps': 0.3, 
            'nb_iter': 5 # tried 50 but it runs super slowly
        }
    elif attack == 'MaxConf':
        attack_func = MaxConfidence
        default_attack_params = {
            'ord': np.inf, 'eps': 0.3, 
            'nb_iter': 5, 'eps_iter': .1 
        }
    else:
        raise ValueError("Unsupported attack: " + str(attack))

    attack_params = {**default_attack_params, **kwargs}
    if dataset == 'mnist':
        attack_params.update({
            'clip_min': 0., 'clip_max': 1.,
        })
        test_data = datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor())
    elif dataset == 'cifar10':
        attack_params.update({
            # without clipping, would get "NotImplementedError: _project_perturbation currently has clipping hard-coded in."
            # input is normalized to normal distribution, 3 sigmas are enough for clipping
            'clip_min': -3., 'clip_max': 3.,
        })
        transform_test = transforms.Compose([transforms.ToTensor(),
                ] + ([transforms.Normalize(*cifar_stats)] if normalize_data else []))
        # important to shuffle the data since we'll measure standard deviation
        test_data = datasets.CIFAR10(root='./cifar10', train=False, transform=transform_test)
    else: 
        raise ValueError("Unsupported dataset: " + str(dataset))

    ex = AdversarialExperiment(attack_func, attack_params, test_data, batch_size, report_interval=report_interval)
    print('Evaluating model %s on attack %s' % (model_path, str(attack)), file=sys.stderr)
    accuracies = ex.evaluate_model(model_path, model_device=model_device, num_batches=num_batches)
    results = {
        'model_path': model_path,
        'accuracies': accuracies,
        'max_probs': ex.extract_max_probs_on_perturbed_examples().tolist(),
        'attack_type': str(attack),
        **attack_params
    }
    if json_out_path:
        json_str = json.dumps(results)
        with open(json_out_path, 'at') as f:
            f.write(json_str) # close it as soon as possible
            f.write('\n')

if __name__ == "__main__":
    import fire
    fire.Fire(run)
