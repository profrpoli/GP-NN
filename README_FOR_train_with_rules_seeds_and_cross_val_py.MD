The purpose of the code in train_with_rules_seeds_and_cross_val.py is to train a
network on a problem with a number of different seeds, with multiple
evolved learning rules including the SBP

Upon completion it saves the losses and accuracies obtained throughout
training with the corresponding learning rule.

In the code:

  problem_name: It is a variable that indicates the problem to be
    solved. It must match the name of a problem in the data folder. In
    the code provided it is set to 'Iris'.

  number_of_seeds: is an integer variable set by user to define the
    number of independent training runs for each problem.

  epoch: number of training epochs

  lr:  learning rate of the network

  inp_dim: number of input features of the problem set

  target_dim:  number of classes of the problem set

  hidden: It is a list that defines the number of hidden layers and
    number of neurons in the hidden layers

  batch_size: It is an integer variable and defines the number of
    samples that are shown to the networks in each iteration

  test_split: It is a rate used to define how many percents of the
    samples in the data are split for testing

  activation: the name of the activation used in hidden layers. 

  cv_fold_numb:  number of cross validation folds. 

  data_norm: a Boolean variable to control the whether a min-max
    normalization is applied to the input data or not

  ruleexps: is a list that includes the expression of the learning
    rules intended to be tested

  rules: is a list that includes labels for corresponding rule in
    ruleexps

  def calc_rule(x, o, d, e, rule): is function that applies the
  learning rule in rule variable to the corresponding network weight
  and returns the resulting value.
