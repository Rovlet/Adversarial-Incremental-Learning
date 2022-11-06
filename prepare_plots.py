import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_rel
import numpy as np

if __name__ == '__main__':
    dirs = [d for d in os.listdir('log/results')]
    accuracy_on_first_task = {}
    # for d in dirs:
    #     files = [f for f in os.listdir('results/' + d + '/results/') if f.startswith('acc_tag-2022')][0]
    #     df = pd.read_csv('results/' + d + '/results/' + files, header=None, sep='	')
    #     df = df[0]
    #     accuracy_on_first_task[d] = list(df)[:6]
    #     accuracy_on_first_task[d] = [round(x, 3) for x in accuracy_on_first_task[d]]
    #
    #     plt.scatter(list(range(len(accuracy_on_first_task[d]))),accuracy_on_first_task[d] , label=' '.join((d.split('_')[1:])))
    #     plt.plot(list(range(len(accuracy_on_first_task[d]))), accuracy_on_first_task[d])
    #     for i, txt in enumerate(accuracy_on_first_task[d]):
    #         plt.annotate(txt, (i, accuracy_on_first_task[d][i]))
    # plt.legend()
    # plt.xlabel('Task number')
    # plt.ylabel('Accuracy')
    # plt.title('Accuracy on the first task')
    #
    # plt.show()
    #
    # for d in dirs:
    #     files = [f for f in os.listdir('results/' + d + '/results/') if f.startswith('avg_accs_tag')][0]
    #     df = pd.read_csv('results/' + d + '/results/' + files, header=None, sep='	')
    #     list_of_values = df.values.tolist()[0][:6]
    #     list_of_values = [round(x, 3) for x in list_of_values]
    #     plt.scatter(list(range(len(list_of_values))), list_of_values , label=' '.join((d.split('_')[1:])), )
    #     for i, txt in enumerate(list_of_values):
    #         plt.annotate(txt, (i, list_of_values[i]))
    #     plt.plot(list_of_values)
    # plt.legend()
    # plt.xlabel('Task number')
    # plt.ylabel('Accuracy')
    # plt.title(f'Accuracy on all tasks')
    # plt.show()


    # ttest_rel
    list_of_values = []
    print(dirs)
    for d in dirs:
        files = [f for f in os.listdir('log/results/' + d + '/results/') if f.startswith('avg_accs_tag')][0]
        df = pd.read_csv('log/results/' + d + '/results/' + files, header=None, sep='	')
        list_of_values.append(df.values.tolist()[0][:6])

    list_of_values = np.array(list_of_values)
    alfa = .05
    t_statistic = np.zeros((len(dirs), len(dirs)))
    p_value = np.zeros((len(dirs), len(dirs)))

    for i in range(len(dirs)):
        for j in range(len(dirs)):
            if i == j:
                t_statistic[i][j] = 0
            else:
                t_statistic[i, j], p_value[i, j] = ttest_rel(list_of_values[i], list_of_values[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    from tabulate import tabulate

    names_column = np.array([[d.split('_')[1]] for d in dirs])
    print(names_column)
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, dirs, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, dirs, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((len(dirs), len(dirs)))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), dirs)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((len(dirs), len(dirs)))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), dirs)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), dirs)
    print("Statistically significantly better:\n", stat_better_table)


    #
    # print(list_of_values)
    # t_statistic = np.zeros((len(list_of_values[d]), len(list_of_values[d])))
    # p_value = np.zeros((len(list_of_values[d]), len(list_of_values[d])))
    # headers = list(list_of_values.keys())
    # names_column = np.array([[d.split('_')[1]]]for d in dirs)
    # for i in range(len(list_of_values[d])):
    #     for j in range(len(list_of_values[d])):
    #         t_statistic[i][j], p_value[i][j] = ttest_rel(list_of_values[d][i], list_of_values[d][j])
    # print(t_statistic)
    # from tabulate import tabulate
    #
    # # p_value_table = np.concatenate((names_column, p_value), axis=1)
    # # p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    # advantage = np.zeros((len(headers), len(headers)))
    # advantage[t_statistic > 0] = 1
    # advantage_table = tabulate(np.concatenate(
    #     (names_column, advantage), axis=1), headers)
    # print("Advantage:\n", advantage_table)
