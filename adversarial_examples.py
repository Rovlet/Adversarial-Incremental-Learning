from foolbox import PyTorchModel
from settings import *
from sklearn.utils import shuffle
import numpy as np
import foolbox.attacks as fa
import torch
from settings import MAX_ADV_PER_EPSILON
from data_loader import read_train_data, get_adv_loaders, get_transform
import pandas as pd
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
                image.save(f"./adv_images/{i}/{u}_{j}_{k}.png")
                if u > 50:
                    break


class AdversarialExamplesBaseClass:
    def __init__(self):

        self.attacks = [fa.FGSM(), fa.InversionAttack(), fa.LinfPGD(), fa.LinfBasicIterativeAttack(),
                        fa.LinfDeepFoolAttack(), fa.LinfAdditiveUniformNoiseAttack()]

    def generate_adversarial_examples(self, model, images, labels, t):
        result_df = pd.DataFrame(columns=['dataset', 'attack', 'accuracy', 'epsilon'])
        fmodel = PyTorchModel(model, bounds=(0, 1))
        attacks = self.attacks
        images = images.cuda()
        labels = labels.cuda()

        attack_success = np.zeros((len(attacks), len(EPSILONS), len(images)), dtype=bool)
        for i, attack in enumerate(attacks):
            raw_advs, clipped_advs, success = attack(fmodel, images[:2000], labels[:2000], epsilons=EPSILONS)
            success = np.array(success.cpu())
            success = np.where(success == True, 1, 0)
            attack_success[i] = success
            for j, result in enumerate(success):
                values = {'dataset': "USTC-TFC2016", 'attack': attack, 'accuracy': 1.0 - result.mean(axis=-1).round(3),
                          'epsilon': EPSILONS[j]}
                result_df = result_df.append(values, ignore_index=True)
            save_images(clipped_advs, i)


        # SAVE RESULTS
        result_df.to_csv("USTC-TFC2016_adversarial_examples.csv")

        # attack = self.attacks[t-1]
        # return_images = []
        # return_labels = []
        # images = images.cuda()
        # labels = labels.cuda()
        # # for i, attack in enumerate(attacks):
        # fmodel = PyTorchModel(model, bounds=(-1, 1))
        # for i in range(9):
        #     raw_advs, clipped_advs, success = attack(fmodel, images[i*1000:(i+1)*1000], labels[i*1000:(i+1)*1000], epsilons=EPSILONS)
        #
        #     for adv in clipped_advs:
        #         adv = adv.cpu().numpy()
        #         np.random.shuffle(adv)
        #         for image in adv[:MAX_ADV_PER_EPSILON]:
        #             return_images.append(image)
        #             return_labels.append(t+9)
        #
        # raw_advs, clipped_advs, success = attack(fmodel, images[3000:6000], labels[3000:6000],
        #                                          epsilons=EPSILONS)
        # for adv in clipped_advs:
        #     adv = adv.cpu().numpy()
        #     np.random.shuffle(adv)
        #     for image in adv[:MAX_ADV_PER_EPSILON]:
        #         return_images.append(image)
        #         return_labels.append(t + 9)
        #
        # raw_advs, clipped_advs, success = attack(fmodel, images[6000:], labels[6000:],
        #                                          epsilons=EPSILONS)
        # for adv in clipped_advs:
        #     adv = adv.cpu().numpy()
        #     np.random.shuffle(adv)
        #     for image in adv[:MAX_ADV_PER_EPSILON]:
        #         return_images.append(image)
        #         return_labels.append(t + 9)
        # #
        # raw_advs, clipped_advs, success = attack(fmodel, images[6000:], labels[6000:],
        #                                          epsilons=EPSILONS)
        # for adv in clipped_advs:
        #     adv = adv.cpu().numpy()
        #     np.random.shuffle(adv)
        #     for image in adv[:MAX_ADV_PER_EPSILON]:
        #         return_images.append(image)
        #         return_labels.append(t + 9)
        # return_images, return_labels = shuffle(return_images, return_labels)
        # return return_images, return_labels

    def prepare_adv_dataset(self, model, images, labels, task):
        labels = np.array(labels)
        labels = torch.from_numpy(labels)
        labels = labels.long()
        images = torch.stack(images)
        self.generate_adversarial_examples(model, images, labels, task)
        # return raw_advs, labels


    def get_loaders_with_adv_examples(self, net, tst_loader, t):

        trn_transform, tst_transform = get_transform()
        images, labels, validation_images, validation_labels = read_train_data(trn_transform=trn_transform,
                                                                               tst_transform=tst_transform)
        # shuffle
        images, labels = shuffle(images, labels)
        raw_advs, labels = self.prepare_adv_dataset(net.model, images[:2000], labels[:2000], t)
        trn_loader, val_loader, tst = get_adv_loaders(raw_advs, labels, t)
        tst_loader.append(tst)
        return trn_loader, val_loader, tst_loader



# from foolbox import PyTorchModel
# from settings import *
# from sklearn.utils import shuffle
# import numpy as np
# import foolbox.attacks as fa
# import torch
# from settings import MAX_ADV_PER_EPSILON
# from data_loader import read_train_data, get_adv_loaders, get_transform
#
#
# class AdversarialExamplesBaseClass:
#     def __init__(self):
#         self.attacks = {10: fa.FGSM(), 11: fa.LinfPGD(), 12: fa.LinfBasicIterativeAttack(),
#                         13: fa.LinfAdditiveUniformNoiseAttack(), 14: fa.LinfDeepFoolAttack()}
#
#     def generate_adversarial_examples(self, model, images, labels, task):
#         attack = self.attacks[task]
#         images = images.cuda()
#         labels = labels.cuda()
#         fmodel = PyTorchModel(model, bounds=(-1, 1))
#         raw_advs, clipped_advs, success = attack(fmodel, images[:3000], labels[:3000], epsilons=EPSILONS)
#         return_images = []
#         for adv in clipped_advs:
#             adv = adv.cpu().numpy()
#             np.random.shuffle(adv)
#             for image in adv[:MAX_ADV_PER_EPSILON]:
#                 return_images.append(image)
#
#         raw_advs, clipped_advs, success = attack(fmodel, images[3000:], labels[3000:],
#                                                  epsilons=EPSILONS)
#         for adv in clipped_advs:
#             adv = adv.cpu().numpy()
#             np.random.shuffle(adv)
#             for image in adv[:MAX_ADV_PER_EPSILON]:
#                 return_images.append(image)
#
#         result = shuffle(return_images)
#         return result
#
#     def prepare_adv_dataset(self, model, images, labels, task):
#         labels = np.array(labels)
#         labels = torch.from_numpy(labels)
#         labels = labels.long()
#         images = torch.stack(images)
#         raw_advs = self.generate_adversarial_examples(model, images, labels, task)
#         labels = [task] * len(raw_advs)
#         return raw_advs, labels
#
#     def get_loaders_with_adv_examples(self, net, tst_loader, t):
#
#         trn_transform, tst_transform = get_transform()
#         images, labels, validation_images, validation_labels = read_train_data(trn_transform=trn_transform,
#                                                                                tst_transform=tst_transform)
#         # shuffle
#         images, labels = shuffle(images, labels)
#         raw_advs, labels = self.prepare_adv_dataset(net.model, images[:6000], labels[:6000], t + 9)
#         trn_loader, val_loader, tst = get_adv_loaders(raw_advs, labels, t + 9)
#         tst_loader.append(tst)
#         return trn_loader, val_loader, tst_loader

