import numpy as np
from last_layer_analysis import last_layer_analysis
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from utils import get_test_metrics
from settings import LAST_LAYER_ANALYSIS


class Logger:
    def __init__(self, logger):
        self.logger = logger

    def save_conf_matrix(self, predicted, true, t, normalize=False):
        title = f'Confusion matrix {t} malware detection'
        cm = confusion_matrix(true, predicted)
        accuracy = np.trace(cm) / np.sum(cm).astype('float')
        misclass = 1 - accuracy

        plt.figure(figsize=(8, 6))
        cmap = plt.cm.Blues
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        target_names = sorted(list(set(true)))
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.1f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.1f}; misclass={:0.1f}'.format(accuracy, misclass))
        self.logger.log_figure(name='confusion_matrix', iter=t, figure=plt.gcf())

    def print_task_results(self, acc_taw, acc_tag, t, u, test_loss, forg_taw, forg_tag):
        print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
              '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                             100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                             100 * acc_tag[t, u], 100 * forg_tag[t, u]))
        self.logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
        self.logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
        self.logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
        self.logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
        self.logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])

    def log_results(self, acc_taw, acc_tag, forg_taw, forg_tag, net, t, taskcla, max_task):
        self.logger.log_result(acc_taw, name="acc_taw", step=t)
        self.logger.log_result(acc_tag, name="acc_tag", step=t)
        self.logger.log_result(forg_taw, name="forg_taw", step=t)
        self.logger.log_result(forg_tag, name="forg_tag", step=t)
        self.logger.save_model(net.state_dict(), task=t)
        self.logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
        self.logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        self.logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
        self.logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)

    def print_last_layer_result(self, net, t, taskcla):
        weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
        self.logger.log_figure(name='weights', iter=t, figure=weights)
        self.logger.log_figure(name='bias', iter=t, figure=biases)

        # Output sorted weights and biases
        weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
        self.logger.log_figure(name='weights', iter=t, figure=weights)
        self.logger.log_figure(name='bias', iter=t, figure=biases)

    @staticmethod
    def print_appr_args(approach_kwargs):
        print('Approach arguments =')
        for arg in approach_kwargs.keys():
            print('\t' + arg + ':', approach_kwargs[arg])
        print('=' * 108)

    @staticmethod
    def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
        """Print summary of results"""
        for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
            print('*' * 108)
            print(name)
            for i in range(metric.shape[0]):
                print('\t', end='')
                for j in range(metric.shape[1]):
                    print('{:5.1f}% '.format(100 * metric[i, j]), end='')
                if np.trace(metric) == 0.0:
                    if i > 0:
                        print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
                else:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
                print()
        print('*' * 108)

    @staticmethod
    def save_results(results, current_task_number, tst_loader, approach, logger, net):
        all_predicted, all_true = get_test_metrics(results, current_task_number, tst_loader, approach)
        logger.save_conf_matrix(all_predicted, all_true, current_task_number)
        logger.save_results(results.acc_taw, results.acc_tag, results.forg_taw, results.forg_tag, net, current_task_number,
                            results.taskcla, results.max_task)
        if LAST_LAYER_ANALYSIS:
            logger.print_last_layer_result(net, current_task_number, results.taskcla)