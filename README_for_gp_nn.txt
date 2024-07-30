Definition of the variables at the top, before the main program:

cluster: Gets the system time to save the related results in.

min_seed: An integer value that is used for initializing the first
  random seed.

max_seed: An integer value that is used for initializing the the last
  random seed.  As many networks are initialized as the differences
  between max_seed and min_seed.

normalisation: A Boolean variable that is used to decide whether
  min-max normalization is apply to to the input data.

penalise: is a Boolean variable that is used to decide the GP program
  is penalized considering behaviour of the learning throughout the
  epochs.

penalty_coef: A float penalty coefficient. It is applied to the error
  difference between the current epoch and the previous epoch in the
  case of penalize variable is True and the error in the current epoch
  is higher than the previous epoch.

reward_coef: A float reward coefficient. It is applied to the error
  difference between the current epoch and the previous epoch in the
  case of penalize variable is True and the error in the current epoch
  is lower than the previous epoch.

mainfolder: A string variable that is used to define the main folder
  of the results.


The for loop : for seed in range(min_seed, max_seed) fills two lists (N, C).

  N consists of objects of class NeuralNetwork (one for each seed and
    problem used in the fitness function).

    Parameters include: filename for the dataset, epochs, number of
    inputs, number of outputs, the number and size of the hidden
    layers, the learning rate and fraction of the data to be used for
    training.  The file name must be matches the file name of the data
    set in csv format in the subfolder "data"

  C consists of objects of class interpreter (one for each seed and probem).

    Here only parameter required is the register length for the
    interpreter, which needs to match the number of weights used in
    the corresponding object of in N list.

In the main program, 

   g is an object of class gp

   Parameters include: population size, program length, number of
   instructions in the primitive set, mutation probability and
   tournament size.


   sbp, np.array([23]), is an
   array containing only the opcode that copies register rSBP into the output 
   register r0 thus producing the standard backpropagation (SBP) learning rule.

   If the instruction list changes by adding new instructions,
   removing some instructions, and/or changing the order of the
   instructions, the array may require to be re-arranged to produce
   the SBP learning rule.
