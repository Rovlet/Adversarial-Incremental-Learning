import os
import time
import torch
import utils

from adversarial_examples import AdversarialExamplesBaseClass
from configuration import *
from settings import *


if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
    device = 'cuda'
    print("Current device name: ", torch.cuda.get_device_name(0))
else:
    print('WARNING: [CUDA unavailable] Using CPU instead!')
    device = 'cpu'

utils.seed_everything(seed=41)

if __name__ == '__main__':
    for method_name, method_paramethers in methods.items():
        tstart = time.time()
        full_exp_name = f'{DATABASE}_{method_name}'
        base_kwargs, approach_kwargs, appr_exemplars_dataset_args = get_method_kwargs(method_paramethers, method_name)
        adversarial_examples = AdversarialExamplesBaseClass()
        net = get_model(method_paramethers)
        logger = get_logger(full_exp_name)
        trn_loader, val_loader, tst_loader, taskcla = get_data_loaders(method_paramethers)
        approach = get_approach(method_name, trn_loader, base_kwargs, logger, appr_exemplars_dataset_args, net, device)

        max_task = len(taskcla)
        acc_taw, acc_tag, forg_taw, forg_tag = prepare_results_numpy(max_task)
        logger.print_appr_args(approach_kwargs)
        for t in range(len(adversarial_examples.attacks)):
            number_of_classes = t + 10
            if t >= max_task:
                continue

            if t > 0:
                print("adverserial examples creation")
                trn_loader, val_loader, tst_loader = adversarial_examples.get_loaders_with_adv_examples(net, tst_loader, t)
                number_of_classes = 1

            print('*' * 108)
            print('Task {:2d}'.format(t))
            print('*' * 108)

            # Add head for current task
            net.add_head(number_of_classes)
            net.to(device)

            # Train
            approach.train(t, trn_loader, val_loader)
            print('-' * 108)

            # Test
            predicted = []
            true = []
            for u in range(t + 1):
                test_loss, acc_taw[t, u], acc_tag[t, u], p, tar = approach.eval(u, tst_loader[u])
                predicted += p
                true += tar
                if u < t:
                    forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                    forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
                logger.print_task_results(acc_taw, acc_tag, t, u, test_loss, forg_taw, forg_tag)
            logger.save_conf_matrix(predicted, true, t)

            print('Save at ' + os.path.join(RESULT_PATH, full_exp_name))
            logger.print_final_results(acc_taw, acc_tag, forg_taw, forg_tag, net, t, taskcla, max_task)

            if LAST_LAYER_ANALYSIS:
                logger.print_last_layer_result(net, t, taskcla)
        # Print Summary
        utils.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
        print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
        print('Done!')
