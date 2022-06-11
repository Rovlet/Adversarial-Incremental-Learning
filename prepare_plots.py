import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_rel
import numpy as np

if __name__ == '__main__':
    dirs = [d for d in os.listdir('log/results')]
    accuracy_on_first_task = {}
    for d in dirs:
        files = [f for f in os.listdir('results/' + d + '/results/') if f.startswith('acc_tag-2022')][0]
        df = pd.read_csv('results/' + d + '/results/' + files, header=None, sep='	')
        df = df[0]
        accuracy_on_first_task[d] = list(df)[:6]
        accuracy_on_first_task[d] = [round(x, 3) for x in accuracy_on_first_task[d]]

        plt.scatter(list(range(len(accuracy_on_first_task[d]))),accuracy_on_first_task[d] , label=' '.join((d.split('_')[1:])))
        plt.plot(list(range(len(accuracy_on_first_task[d]))), accuracy_on_first_task[d])
        for i, txt in enumerate(accuracy_on_first_task[d]):
            plt.annotate(txt, (i, accuracy_on_first_task[d][i]))
    plt.legend()
    plt.xlabel('Task number')
    plt.ylabel('Accuracy')
    plt.title('Accuracy on the first task')

    plt.show()

    for d in dirs:
        files = [f for f in os.listdir('results/' + d + '/results/') if f.startswith('avg_accs_tag')][0]
        df = pd.read_csv('results/' + d + '/results/' + files, header=None, sep='	')
        list_of_values = df.values.tolist()[0][:6]
        list_of_values = [round(x, 3) for x in list_of_values]
        plt.scatter(list(range(len(list_of_values))), list_of_values , label=' '.join((d.split('_')[1:])), )
        for i, txt in enumerate(list_of_values):
            plt.annotate(txt, (i, list_of_values[i]))
        plt.plot(list_of_values)
    plt.legend()
    plt.xlabel('Task number')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy on all tasks')
    plt.show()

