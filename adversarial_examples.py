from foolbox import PyTorchModel
from settings import *
from sklearn.utils import shuffle
import numpy as np
import foolbox.attacks as fa
import torch
from data_loader import read_train_data, get_adv_loaders, get_transform
from PIL import Image


def save_images(clipped_advs, i):
    for j, eps in enumerate(EPSILONS):
        for k, adv in enumerate(clipped_advs):
            for u, image in enumerate(adv):
                image = image.cpu().reshape(32, 32, 3)
                image = image.numpy()
                # image = np.squeeze(image, axis=2)
                image = Image.fromarray(np.uint8(image*255))
                # gray scale
                image = image.convert('L')
                image.save(f"./adv_images/{u}_{j}_{k}.png")
                if u > 50:
                    break


class AdversarialExamplesBaseClass:
    def __init__(self):

        self.attacks = [fa.FGSM(), fa.InversionAttack(), fa.LinfPGD(), fa.LinfBasicIterativeAttack(),
                        fa.LinfDeepFoolAttack(), fa.LinfAdditiveUniformNoiseAttack()]

    def generate_adversarial_examples(self, model, images, labels, t):
        attack = self.attacks[t-1]
        return_images = []
        return_labels = []
        images = images.cuda()
        labels = labels.cuda()
        fmodel = PyTorchModel(model, bounds=(-1, 1))
        number_of_examples_per_iteration = 1000
        iterations = int(number_of_adversarial_examples_pr_attack/number_of_examples_per_iteration)
        for i in range(iterations):
            raw_advs, clipped_advs, success = attack(fmodel, images[i*number_of_examples_per_iteration:(i+1)*number_of_examples_per_iteration],
                                                     labels[i*number_of_examples_per_iteration:(i+1)*number_of_examples_per_iteration], epsilons=EPSILONS)

            for adv in clipped_advs:
                adv = adv.cpu().numpy()
                np.random.shuffle(adv)
                for image in adv[:MAX_ADV_PER_EPSILON]:
                    return_images.append(image)
                    return_labels.append(t+config['number_of_classes']-1)
        return return_images, return_labels

    def prepare_adv_dataset(self, model, images, labels, task):
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        labels = labels.long()
        images = torch.stack(images)
        return self.generate_adversarial_examples(model, images, labels, task)

    def get_loaders_with_adv_examples(self, net, tst_loader, t):
        trn_transform, tst_transform = get_transform()
        images, labels, validation_images, validation_labels = read_train_data(trn_transform=trn_transform,
                                                                               tst_transform=tst_transform)
        # shuffle
        images, labels = shuffle(images, labels)
        raw_advs, labels = self.prepare_adv_dataset(net.model, images[:number_of_adversarial_examples_pr_attack],
                                                    labels[:number_of_adversarial_examples_pr_attack], t)
        trn_loader, val_loader, tst = get_adv_loaders(raw_advs, labels, t)
        tst_loader.append(tst)
        return trn_loader, val_loader, tst_loader