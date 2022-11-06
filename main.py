from adversarial_examples import AdversarialExamplesBaseClass
from utils import *
from settings import *

if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
    device = 'cuda'
    print("Current device name: ", torch.cuda.get_device_name(0))
else:
    print('WARNING: [CUDA unavailable] Using CPU instead!')
    device = 'cpu'

seed_everything(seed=41)


class Config:
    def __init__(self, method_name, method_parameters):
        self.method_name = method_name
        self.full_name = f'{DATABASE}_{self.method_name}'
        self.method_parameters = method_parameters

        self.base_kwargs, self.approach_kwargs, self.appr_exemplars_dataset_args = get_method_kwargs(
            self.method_parameters,
            self.method_name)
        self.taskcla = config['taskcla']
        self.max_task = len(self.taskcla)


class ResultsStorage:
    def __init__(self):
        self.taskcla = config['taskcla']
        max_task = len(self.taskcla)
        self.acc_taw = np.zeros((max_task, max_task))
        self.acc_tag = np.zeros((max_task, max_task))
        self.forg_taw = np.zeros((max_task, max_task))
        self.forg_tag = np.zeros((max_task, max_task))


def run():
    for method_name, method_parameters in methods.items():
        config = Config(method_name, method_parameters)
        results = ResultsStorage()

        logger = get_logger(config.full_name)
        logger.print_appr_args(config.approach_kwargs)

        adversarial_examples = AdversarialExamplesBaseClass()

        net = get_model(method_parameters)

        trn_loader, val_loader, tst_loader = get_data_loaders(config.method_parameters)
        incremental_learning_method = get_incremental_learning_class(config.method_name, trn_loader, config.base_kwargs,
                                                                     logger, config.appr_exemplars_dataset_args, net,
                                                                     device)

        number_of_classes_in_this_task = base_classes_number
        for task in range(len(adversarial_examples.attacks)):
            if task > 0:
                trn_loader, val_loader, tst_loader = adversarial_examples.get_loaders_with_adv_examples(
                    net, tst_loader, task)
                number_of_classes_in_this_task = 1

            # Add head for current task
            net.add_head(number_of_classes_in_this_task)
            net.to(device)

            incremental_learning_method.train(task, trn_loader, val_loader)
            logger.save_results(results, task, tst_loader, incremental_learning_method, net, config.max_task)

        logger.print_summary(results.acc_taw, results.acc_tag, results.forg_taw, results.forg_tag)


if __name__ == '__main__':
    run()
