#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ann.h"


#ifndef ann_act
#define ann_act_hidden ann_act_hidden_indirect
#define ann_act_output ann_act_output_indirect
#else
#define ann_act_hidden ann_act
#define ann_act_output ann_act
#endif

#define LOOKUP_SIZE 4096

double ann_act_hidden_indirect(const struct ann *an, double a) {
    return an->activation_hidden(an, a);
}

double ann_act_output_indirect(const struct ann *an, double a) {
    return an->activation_output(an, a);
}

const double sigmoid_dom_min = -15.0;
const double sigmoid_dom_max = 15.0;
double interval;
double lookup[LOOKUP_SIZE];

#ifdef __GNUC__
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#define unused          __attribute__((unused))
#else
#define likely(x)       x
#define unlikely(x)     x
#define unused
#pragma warning(disable : 4996) /* For fscanf */
#endif



double ann_act_sigmoid(const ann *ann unused, double a) {
    if (a < -45.0) return 0;
    if (a > 45.0) return 1;
    return 1.0 / (1 + exp(-a));
}

void ann_init_sigmoid_lookup(const ann *ann) {
        const double f = (sigmoid_dom_max - sigmoid_dom_min) / LOOKUP_SIZE;
        int i;

        interval = LOOKUP_SIZE / (sigmoid_dom_max - sigmoid_dom_min);
        for (i = 0; i < LOOKUP_SIZE; ++i) {
            lookup[i] = ann_act_sigmoid(ann, sigmoid_dom_min + f * i);
        }
}

double ann_act_sigmoid_cached(const ann *ann unused, double a) {
    assert(!isnan(a));

    if (a < sigmoid_dom_min) return lookup[0];
    if (a >= sigmoid_dom_max) return lookup[LOOKUP_SIZE - 1];

    size_t j = (size_t)((a-sigmoid_dom_min)*interval+0.5);

    /* Because floating point... */
    if (unlikely(j >= LOOKUP_SIZE)) return lookup[LOOKUP_SIZE - 1];

    return lookup[j];
}

double ann_act_linear(const struct ann *ann unused, double a) {
    return a;
}

double ann_act_threshold(const struct ann *ann unused, double a) {
    return a > 0;
}


double double_rand( double min, double max )
{
    double scale = rand() / (double) RAND_MAX; /* [0, 1.0] */
    return min + scale * ( max - min );      /* [min, max] */
}



dataset *load_dataset ( char *filename, double ratio, dataset **testset ){
    
    FILE *datafile;
    dataset *trainset;

    int num_inputs, num_outputs, num_members;
    int i, j;
    double temp=0.0;

    char trainset_check = 0x00;
    char testset_check = 0x00;

    /* Open filename */
    if ((datafile = fopen(filename, "r")) == NULL)
    {
        printf("Couldn't open file %s\n", filename);
        return (NULL);
    }

    /* Load first line which contain settings */
    fscanf(datafile, "%d, %d, %d\n", &num_members, &num_inputs, &num_outputs);

    /* Setup the dataset */

    /* Allocate memory for the dataset */
    if ((trainset = (dataset *)malloc(sizeof(dataset))) == NULL)
    {
        perror("Couldn't allocate the trainest\n");
        return (NULL);
    }
    /* Allocate memory for the dataset */
    if (( *testset = (dataset *)malloc(sizeof(dataset))) == NULL)
    {
        perror("Couldn't allocate the testset\n");
        return (NULL);
    }

    /* Set the variables */
    trainset->num_members = ceil( num_members * (1-ratio) ) ;
    trainset->num_inputs = num_inputs;
    trainset->num_outputs = num_outputs;

    /* Set the variables */
    (*testset)->num_members = floor( num_members * ratio) ;
    (*testset)->num_inputs = num_inputs;
    (*testset)->num_outputs = num_outputs;

    /* Allocate memory for the arrays in the dataset */
    if ((trainset->members = (data_member *)malloc(num_members * sizeof(data_member))) != NULL)
        trainset_check |= 0x01;

    if (trainset_check < 1)
    {
        printf("1.Couldn't allocate trainset\n");
        if (trainset_check & 0x01)
            free(trainset->members);
        free(trainset);
        return (NULL);
    }

    /* Allocate memory for the arrays in the dataset */
    if (((*testset)->members = (data_member *)malloc(num_members * sizeof(data_member))) != NULL)
        testset_check |= 0x01;

    if (testset_check < 1)
    {
        printf("1.Couldn't allocate testset\n");
        if (testset_check & 0x01)
            free((*testset)->members);
        free(testset);
        return (NULL);
    }

    /* Get the data */
    /* For each Member */
    for (i = 0; i < trainset->num_members; i++)
    {
        /* Allocate the memory for the member */
        trainset_check = 0x00;
        if (((trainset->members + i)->inputs = (double *)malloc(num_inputs * sizeof(double))) != NULL)
            trainset_check |= 0x01;
        if (((trainset->members + i)->targets = (double *)malloc(num_outputs * sizeof(double))) != NULL)
            trainset_check |= 0x03;

        if (trainset_check < 3)
        {
            printf("2.Couldn't allocate trainset\n");
            /* Deallocate the previous loops */
            for (j = 0; j < i; j++)
            {
                free((trainset->members + j)->inputs);
                free((trainset->members + j)->targets);
            }

            if (trainset_check & 0x01)
                free((trainset->members + i)->inputs);
            if (trainset_check & 0x02)
                free((trainset->members + i)->targets);

            free(trainset->members);
            free(trainset);
            return (NULL);
        }

        /* Read the inputs */
        for (j = 0; j < num_inputs; j++)
        {
            fscanf(datafile, "%lf, ", &temp);
            printf("%lf, ", temp);
            (trainset->members + i)->inputs[j] = temp;
        }

        /* Read the outputs */
        for (j = 0; j < num_outputs - 1; j++)
        {
            fscanf(datafile, "%lf, ", &temp);
            printf("%lf, ", temp);
            (trainset->members + i)->targets[j] = temp;
        }
        fscanf(datafile, "%lf\n", &temp);
        printf("%lf\n", temp);
        (trainset->members + i)->targets[j] = temp;
    }

    printf("done loading trainset...\n");

    /* Get the data */
    /* For each Member */
    
    for (i = 0; i < (*testset)->num_members; i++)
    {
        /* Allocate the memory for the member */
        testset_check = 0x00;
        if ((((*testset)->members + i)->inputs = (double *)malloc(num_inputs * sizeof(double))) != NULL)
            testset_check |= 0x01;
        if ((((*testset)->members + i)->targets = (double *)malloc(num_outputs * sizeof(double))) != NULL)
            testset_check |= 0x03;

        if (testset_check < 3)
        {
            printf("2.Couldn't allocate testset\n");
            /* Deallocate the previous loops */
            for (j = 0; j < i; j++)
            {
                free(((*testset)->members + j)->inputs);
                free(((*testset)->members + j)->targets);
            }

            if (testset_check & 0x01)
                free(((*testset)->members + i)->inputs);
            if (testset_check & 0x02)
                free(((*testset)->members + i)->targets);

            free((*testset)->members);
            free(testset);
            return (NULL);
        }

        /* Read the inputs */
        for (j = 0; j < num_inputs; j++)
        {
            fscanf(datafile, "%lf, ", &temp);
             printf("%lf, ", temp);
            ((*testset)->members + i)->inputs[j] = temp;
        }

        /* Read the outputs */
        for (j = 0; j < num_outputs - 1; j++)
        {
            fscanf(datafile, "%lf, ", &temp);
            printf("%lf, ", temp);
            ((*testset)->members + i)->targets[j] = temp;
        }
        fscanf(datafile, "%lf\n", &temp);
        printf("%lf\n", temp);
        ((*testset)->members + i)->targets[j] = temp;
    }

    printf("done loading testset...\n");

    /* Make sure the file is closed */
    fclose(datafile);

    /* Finally, return the pointer to the dataset */
    return (trainset); 
}


void ann_randomize(ann *ann) {
    int i;
    for (i = 0; i < ann->weights; ++i) {
        // double r = ANN_RANDOM();
        /* Sets weights from -0.5 to 0.5. */
        ann->weight[i] =  double_rand(-5.0,5.0);
    }
}

ann *create   ( int inputs, int hidden_layers, int hidden, int outputs ) {
    
    if (hidden_layers < 0) return 0;
    if (inputs < 1) return 0;
    if (outputs < 1) return 0;
    if (hidden_layers > 0 && hidden < 1) return 0;


    const int hidden_weights = hidden_layers ? (inputs+1) * hidden + (hidden_layers-1) * (hidden+1) * hidden : 0;
    const int output_weights = (hidden_layers ? (hidden+1) : (inputs+1)) * outputs;
    const int total_weights = (hidden_weights + output_weights);

    const int total_neurons = (inputs + hidden * hidden_layers + outputs);

    /* Allocate extra size for weights, outputs, and deltas. */
    const int size = sizeof(ann) + sizeof(double) * (total_weights + total_neurons + (total_neurons - inputs));
    ann *ret = malloc(size);
    if (!ret) return 0;

    ret->inputs = inputs;
    ret->hidden_layers = hidden_layers;
    ret->hidden = hidden;
    ret->outputs = outputs;

    ret->weights = total_weights;
    ret->neurons = total_neurons;

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(ann));
    ret->output = ret->weight + ret->weights;
    ret->delta = ret->output + ret->neurons;

    ann_randomize(ret);

    ret->activation_hidden = ann_act_sigmoid_cached;
    ret->activation_output = ann_act_sigmoid_cached;

    ann_init_sigmoid_lookup(ret);

    return ret;
}


double *feedforward(ann const *an, double const *inputs) {
    double const *w = an->weight;
    double *o = an->output + an->inputs;
    double const *i = an->output;

    /* Copy the inputs to the scratch area, where we also store each neuron's
     * output, for consistency. This way the first layer isn't a special case. */
    memcpy(an->output, inputs, sizeof(double) * an->inputs);

    int h, j, k;

    if (!an->hidden_layers) {
        double *ret = o;
        for (j = 0; j < an->outputs; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < an->inputs; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = ann_act_output(an, sum);
        }

        return ret;
    }

    /* Figure input layer */
    for (j = 0; j < an->hidden; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < an->inputs; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = ann_act_hidden(an, sum);
    }

    i += an->inputs;

    /* Figure hidden layers, if any. */
    for (h = 1; h < an->hidden_layers; ++h) {
        for (j = 0; j < an->hidden; ++j) {
            double sum = *w++ * -1.0;
            for (k = 0; k < an->hidden; ++k) {
                sum += *w++ * i[k];
            }
            *o++ = ann_act_hidden(an, sum);
        }

        i += an->hidden;
    }

    double *ret = o;

    /* Figure output layer. */
    for (j = 0; j < an->outputs; ++j) {
        double sum = *w++ * -1.0;
        for (k = 0; k < an->hidden; ++k) {
            sum += *w++ * i[k];
        }
        *o++ = ann_act_output(an, sum);
    }

    /* Sanity check that we used all weights and wrote all outputs. */
    assert(w - an->weight == an->weights);
    assert(o - an->output == an->neurons);

    return ret;
}

void ann_train(ann const *an, double const *inputs, double const *desired_outputs, double learning_rate) {
    /* To begin with, we must run the network forward. */
    feedforward(an, inputs);

    int h, j, k;

    /* First set the output layer deltas. */
    {
        double const *o = an->output + an->inputs + an->hidden * an->hidden_layers; /* First output. */
        double *d = an->delta + an->hidden * an->hidden_layers; /* First delta. */
        double const *t = desired_outputs; /* First desired output. */


        /* Set output layer deltas. */
        if (ann_act_output == ann_act_linear ||
                an->activation_output == ann_act_linear) {
            for (j = 0; j < an->outputs; ++j) {
                *d++ = *t++ - *o++;
            }
        } else {
            for (j = 0; j < an->outputs; ++j) {
                *d++ = (*t - *o) * *o * (1.0 - *o);
                ++o; ++t;
            }
        }
    }


    /* Set hidden layer deltas, start on last layer and work backwards. */
    /* Note that loop is skipped in the case of hidden_layers == 0. */
    for (h = an->hidden_layers - 1; h >= 0; --h) {

        /* Find first output and delta in this layer. */
        double const *o = an->output + an->inputs + (h * an->hidden);
        double *d = an->delta + (h * an->hidden);

        /* Find first delta in following layer (which may be hidden or output). */
        double const * const dd = an->delta + ((h+1) * an->hidden);

        /* Find first weight in following layer (which may be hidden or output). */
        double const * const ww = an->weight + ((an->inputs+1) * an->hidden) + ((an->hidden+1) * an->hidden * (h));

        for (j = 0; j < an->hidden; ++j) {

            double delta = 0;

            for (k = 0; k < (h == an->hidden_layers-1 ? an->outputs : an->hidden); ++k) {
                const double forward_delta = dd[k];
                const int windex = k * (an->hidden + 1) + (j + 1);
                const double forward_weight = ww[windex];
                delta += forward_delta * forward_weight;
            }

            *d = *o * (1.0-*o) * delta;
            ++d; ++o;
        }
    }


    /* Train the outputs. */
    {
        /* Find first output delta. */
        double const *d = an->delta + an->hidden * an->hidden_layers; /* First output delta. */

        /* Find first weight to first output delta. */
        double *w = an->weight + (an->hidden_layers
                ? ((an->inputs+1) * an->hidden + (an->hidden+1) * an->hidden * (an->hidden_layers-1))
                : (0));

        /* Find first output in previous layer. */
        double const * const i = an->output + (an->hidden_layers
                ? (an->inputs + (an->hidden) * (an->hidden_layers-1))
                : 0);

        /* Set output layer weights. */
        for (j = 0; j < an->outputs; ++j) {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (an->hidden_layers ? an->hidden : an->inputs) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }

            ++d;
        }

        assert(w - an->weight == an->weights);
    }


    /* Train the hidden layers. */
    for (h = an->hidden_layers - 1; h >= 0; --h) {

        /* Find first delta in this layer. */
        double const *d = an->delta + (h * an->hidden);

        /* Find first input to this layer. */
        double const *i = an->output + (h
                ? (an->inputs + an->hidden * (h-1))
                : 0);

        /* Find first weight to this layer. */
        double *w = an->weight + (h
                ? ((an->inputs+1) * an->hidden + (an->hidden+1) * (an->hidden) * (h-1))
                : 0);


        for (j = 0; j < an->hidden; ++j) {
            *w++ += *d * learning_rate * -1.0;
            for (k = 1; k < (h == 0 ? an->inputs : an->hidden) + 1; ++k) {
                *w++ += *d * learning_rate * i[k-1];
            }
            ++d;
        }

    }

}

void ann_write(ann const *an, FILE *out) {
    fprintf(out, "%d %d %d %d", an->inputs, an->hidden_layers, an->hidden, an->outputs);

    int i;
    for (i = 0; i < an->weights; ++i) {
        fprintf(out, " %.20e", an->weight[i]);
    }
}

void ann_free(ann *an) {
    free(an);
}

ann *ann_copy(ann const *an) {
    const int size = sizeof(an) + sizeof(double) * (an->weights + an->neurons + (an->neurons - an->inputs));
    ann *ret = malloc(size);
    if (!ret) return 0;

    memcpy(ret, an, size);

    /* Set pointers. */
    ret->weight = (double*)((char*)ret + sizeof(ann));
    ret->output = ret->weight + ret->weights;
    ret->delta = ret->output + ret->neurons;

    return ret;
}