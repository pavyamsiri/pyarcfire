# Internal libraries
from typing import Sequence

# External libraries
from matplotlib import pyplot as plt

# Internal libraries
from .definitions import ImageArrayUnion


def _debug_plot_image(image: ImageArrayUnion | Sequence[ImageArrayUnion]) -> None:
    is_sequence = isinstance(image, list) or isinstance(image, tuple)
    fig = plt.figure()
    axis = fig.add_subplot(111)
    if is_sequence:
        axis.imshow(image[0])
        for current_image in image[:1]:
            axis.imshow(current_image, alpha=0.5)
    else:
        axis.imshow(image)
    plt.show()
    plt.close()
