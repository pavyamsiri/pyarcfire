# PyArcFiRe

PyArcFiRe is a python port of [SpArcFiRe](https://github.com/waynebhayes/SpArcFiRe) which is written primarily in MatLab.
Like SpArcFiRe it can be used to detect spiral arcs in images, mostly for galaxy images however it perhaps may work in other contexts.

## Limitations

Note that this is currently a work in progress and the project may change greatly over time.

### Functionality

This port does not have all of the functionality and features of SpArcFiRe such as bar finding and fitting, automatic centering and deprojection, etc.

## Requirements

Currently this requires python 3.9 or above.
The packages used are:

- Pillow
- numpy
- scikit-image
- scipy
- matplotlib (only if using CLI)
- rich (only if using CLI)
- pytest (for testing)
- hypothesis (for testing)

## Interface

There are two main ways of using PyArcFiRe

1. As a python package to use in your own programs.
2. As a command line interface.

### Package

The main function to interface with is called `detect_spirals_in_image` which takes in a grayscale image and then performs the spiral finding algorithm.

### Command Line Interface

PyArcFiRe can also be interacted with through the command line interface via `python -m pyarcfire ...`. Currently this is a work in progress and is mainly
a way to drive debugging code.
