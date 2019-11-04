/*
 * GENETIC ALGORITHM TO TRAIN ARTIFICIAL NEURAL NETWORK 
 *
 * Copyright (c) 2019 Midhath Ahmed
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

#ifndef GAANN_H
#define GAANN_H

#include <stdio.h>



typedef struct individual {
    int inputs;             /* Number of input neurones      */
    int hidden_layers;      /* Number of hidden layers       */
    int hidden;             /* Number of hidden neurones     */
    int outputs;            /* Number of output neurons.     */
    int weights;            /* Total of weigths(chromosomes) */
    int neurons;            /* Total Number of neurones      */
    double *weight;         /* The weights(genotype)         */
    double *output;         /* Output                        */
} individual;

typedef struct population {
    individual **individuals;
    double **fitness;
    // dataset* input;
    // dataset* output;

} population;

individual *create   ( int inputs, int hidden_layers, int hidden, int outputs );
double     *fitness  ( individual const *ann, double const *inputs );
individual *mutate   ( individual *individual );
void       *crossover( individual *mom, individual *dad );
void       kill      ( individual *individual );

/* ===============================================================================
    DATA SET 
=================================================================================*/


typedef struct dataMember {
	double*			inputs;			/* The input data */
	double*			targets;		/* The target outputs */
} dataMember;

typedef struct dataset{
	dataMember*		members;		/* The members of the dataset */
	int 			numMembers;		/* The number of members in the set */
	int 			numInputs;		/* The number of inputs in the set */
	int 			numOutputs;		/* The number of outputs in the set */
} dataset;

dataset * loadData(char* filename);

#endif

