/*
 * GENETIC ALGORITHM TO TRAIN ARTIFICIAL NEURAL NETWORK 
 *
 * Copyright (c) 2019-2020 Midhath Ahmed
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgement in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 */

#ifndef ANN_H
#define ANN_H

#include <stdio.h>

#ifndef ANN_RANDOM
/* Function for generating random weights, called by randomise() */
#define ANN_RANDOM() (((double)rand())/RAND_MAX)
#endif

struct ann;

typedef double (*actfun)(const struct ann *an, double a);

typedef struct ann {
    int inputs;                 /* Number of input neurones      */
    int hidden_layers;          /* Number of hidden layers       */
    int hidden;                 /* Number of hidden neurones     */
    int outputs;                /* Number of output neurons.     */
    int weights;                /* Total nof weigths(chromosomes)*/
    int neurons;                /* Total Number of neurones      */
    double *weight;             /* The weights(genotype)         */
    double *output;             /* Output                        */
    double fitness;              /* Total fitness of the network    */
    double *delta;
    actfun activation_hidden;   /* Hidden layer activation func  */
    actfun activation_output;   /* Output layer activation func  */
} ann;

ann *create(int inputs, int hidden_layers, int hidden, int outputs);
void detroy(ann *individual);
ann *load(FILE *in); /* Creates ANN from file saved with gaann_write. */
void save(ann const *an, FILE *out); /* Saves the ann. */
ann *ann_copy(ann const *an); /* Returns a new copy of ann. */
void ann_randomize(ann *an);
double *feedforward(ann const *an, double const *inputs);
void ann_free( ann *an);
void ann_train(ann const *an, double const *inputs, double const *desired_outputs, double learning_rate);
double double_rand( double min, double max );

/* Activation Functions*/

void ann_init_sigmoid_lookup(const ann *an);
double ann_act_sigmoid(const ann *an, double a);
double ann_act_sigmoid_cached(const ann *, double a);
double ann_act_threshold(const ann *an, double a);
double ann_act_linear(const ann *an, double a);

/* ==============================================================================
    DATA SET 
=================================================================================*/

typedef struct data_member {
	double*			inputs;			/* The input data */
	double*			targets;		/* The target outputs */
} data_member;

typedef struct dataset{
	data_member*	members;		/* The members of the dataset */
	int 			num_members;	/* The number of members in the set */
	int 			num_inputs;		/* The number of inputs in the set */
	int 			num_outputs;	/* The number of outputs in the set */
} dataset;

dataset * load_dataset(char *filename, double ratio, dataset **testdataset);

#endif

