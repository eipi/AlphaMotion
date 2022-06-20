from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

perplexity = [2, 5, 10]


def plot_and_save_confusion_matrix(cm, detail, results_folder, name):
    plt.figure(figsize=(15, 10), dpi=100)
    plt.title(name)
    if detail == 'normalized':
        s = sns.heatmap(cm, annot=True, fmt=".2f")
    else:
        s = sns.heatmap(cm, annot=True, fmt="d")
    s.set_xlabel("Predicted")
    s.set_ylabel("Actual")
    plt.savefig(results_folder + '/' + name + '_' + detail + '.png')
    plt.close()


def plotTsne(X, y):
    # performing dim reduction
    X_reduce = TSNE(verbose=2, perplexity=perplexity).fit_transform(X)

    print('Creating plot for this t-sne visualization..')
    data = {'x': X_reduce[:, 0],
            'y': X_reduce[:, 1],
            'label': y}
    # preparing dataframe from reduced data
    df = pd.DataFrame(data)
    # draw the plot
    sns.lmplot(data=df, x='x', y='y', hue='label', fit_reg=False, height=8, \
               palette="Set1", markers=['^', 'v', 's', 'o'])
    plt.title("perplexity : {}".format(perplexity))
    img_name = 'TSNE_perp_{}.png'.format(perplexity)
    print('saving this plot as image in present working directory...')
    plt.savefig(img_name)
    plt.show()
    print('Done')
