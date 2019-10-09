import seaborn as sns
import matplotlib.pyplot as plt

def visualize_conv_weights(conv, channel=0, rows=4, columns=4, figsize=(20, 15)):
    fig = plt.figure(figsize=figsize)
    for i in range(1, columns*rows+1):
        fig.add_subplot(rows, columns, i)
        weights = conv.weight[i-1, channel].detach().numpy()
        sns.heatmap(weights, cmap='coolwarm', center=0)
    plt.show()