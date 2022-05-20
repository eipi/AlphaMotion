from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

perplexity=[2,5,10]

def plotTsne(X, y, perplexity):
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
