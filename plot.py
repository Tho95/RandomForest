#module to plot dataframe
import seaborn as sns
import matplotlib.pyplot as plt

def target(X):
    '''function to plot how often the target variable has a particular value'''
    sns.countplot(x='Type',data= X)
    plt.title('Type of glass')
    plt.show()

def paramOptimization(scores,ns):
    sns.lineplot(x=ns, y=scores)
    plt.show()

