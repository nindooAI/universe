
from sklearn.manifold import TSNE
# import plotly.express as px
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from stellargraph.mapper import GraphSAGENodeGenerator


def plot_emb(Universe):
    ids = list(Universe.graph.nodes())
    logger.debug(len(ids))
    emb = Universe.emb_model.predict(GraphSAGENodeGenerator(
        Universe.graph, batch_size=500, num_samples=[5, 5]).flow(ids))
    logger.debug(emb.shape)
    transform = TSNE  # PCA
    trans = transform(n_components=2)
    emb_transformed = pd.DataFrame(trans.fit_transform(emb), index=ids)
    alpha = 0.7
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        emb_transformed[0],
        emb_transformed[1],
        cmap="jet",
        alpha=alpha,
    )
    ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
    plt.title(
        "{} visualization of GraphSAGE embeddings of hold out nodes for dataset".format(
            transform.__name__
        )
    )
    plt.show()
