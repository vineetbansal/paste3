from src.paste import pairwise_align, center_align


# TODO: at this stage I just want to have test functions that provide start points to different functionalities of the code base
# TODO: add more to the test cases


def test_pairwise_alignment(slices):
    pairwise_align(
        slices[0],
        slices[1],
        alpha=0.1,
        dissimilarity="kl",
        a_distribution=slices[0].obsm["weights"],
        b_distribution=slices[1].obsm["weights"],
        G_init=None,
    )


def test_center_alignment(slices):
    n_slices = len(slices)
    center_align(
        slices[0],
        slices,
        lmbda=n_slices * [1.0 / n_slices],
        alpha=0.1,
        n_components=15,
        threshold=0.001,
        dissimilarity="kl",
        distributions=[slices[i].obsm["weights"] for i in range(len(slices))],
    )
