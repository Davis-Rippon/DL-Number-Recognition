[Link to website article](https://davisrippon.com/posts/making-a-libraryless-mlp/)

# DL-Number-Recognition
A Multi-Layer Perceptron project written in c++ without any AI/ML libraries!

## The Goal
I set out to make a replica of the MLP described in 3Blue1Brown's lectures with primarily just the C++ Standard Library. 

Originally, I thought I would have to use Eigen for matrix operations (in the interest of time), but I abandoned that idea. 
The usage of Eigen in this project is redundant (except for the random initial values in NetworkMLP's constructor).

## Build Instructions
Download:

```bash
git clone git@github.com:Davis-Rippon/DL-Number-Recognition.git
```

Build (make sure you're in "DL-Number-Recognition"!):
```bash
make
```


## Future Goals (TODOs)
1. Make it quicker (parallelism, better activation function)
2. Make it flexible (change # of nodes, layers, etc.)
3. Remove Eigen

### Sources
3Blue1Brown's lecture series:
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

Eigen:
https://eigen.tuxfamily.org/index.php?title=Main_Page
