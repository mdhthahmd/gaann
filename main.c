#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <time.h>

#include "ann.h"

#if 1
#define GA
#endif

double error_threshold = 0.25;

int cmp(const void *a, const void *b)
{
    const ann *left = *(const ann **)a;
    const ann *right = *(const ann **)b;

    return (right->fitness < left->fitness) - (left->fitness < right->fitness);
}

void swap(ann **A, ann **B)
{
    ann *temp = *A;
    *A = *B;
    *B = temp;
}

void display_usage()
{
    //-d data/dataset3.txt -s 0.8 -i 5 -h 1 5 -o 1 -p 1000 -m 0.1

    printf("\n");
    printf("FLAGS\n");
    printf("   -d <file>                Relative path to datafile\n");
    printf("   -s <double>              Ratio of spliting the dataset between train and test set\n");
    printf("                                0.0 no testset, 1.0 no trainset\n");
    printf("   -h <int>                 Number of hidden layers,\n");
    printf("   -p <int>                 Population size\n");
    printf("   -m <double>              Chance of mutation for the population_size\n");
}

int main(int argc, char *argv[])
{

    srand((unsigned int)time(NULL));

    int opt;
    opterr = 0;

    if (argc < 2)
    {
        printf("USAGE: %s -<flag1> <arg1> -<flag2> <arg2> ... -<flagN> <argN> ", argv[0]);
        display_usage();
    }
    else
    {

        char *file_name = NULL;
        double split_ratio = -1.0;
        int hidden = 0;
        int population_size = 0;
        double mutation = 0.0;

        while ((opt = getopt(argc, argv, "d:s:h:p:m:")) != -1)
            switch (opt)
            {
            case 'd':
                file_name = optarg;
                break;
            case 's':
                if (sscanf(optarg, "%lf", &split_ratio) != 1)
                {
                    printf("error: option -s, argument must be decimal\n");
                    exit(1);
                }
                break;
            case 'h':
                if (sscanf(optarg, "%d", &hidden) != 1)
                {
                    printf("error: option -h, argument must be an integer\n");
                    exit(1);
                }

                break;
            case 'p':
                if (sscanf(optarg, "%d", &population_size) != 1)
                {
                    printf("error: option -p, argument must be an integer\n");
                    exit(1);
                }
                break;
            case 'm':
                if (sscanf(optarg, "%lf", &mutation) != 1)
                {
                    printf("error: option -m, argument must be decimal\n");
                    exit(1);
                }
                break;
            case '?':
                if (optopt == 'd' ||
                    optopt == 's' ||
                    optopt == 'h' ||
                    optopt == 'p' ||
                    optopt == 'm')
                {
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                    printf("USAGE: %s -<flag1> <arg1> -<flag2> <arg2> ... -<flagN> <argN> ", argv[0]);
                    display_usage();
                }
                else if (isprint(optopt))
                {
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                    printf("USAGE: %s -<flag1> <arg1> -<flag2> <arg2> ... -<flagN> <argN> ", argv[0]);
                    display_usage();
                }
                else
                {
                    fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                    printf("USAGE: %s -<flag1> <arg1> -<flag2> <arg2> ... -<flagN> <argN> ", argv[0]);
                    display_usage();
                }
            }

        int index = 0;
        for (index = optind; index < argc; index++)
            printf("Non-option argument %s\n", argv[index]);

        /*     printf("Parsed Values\n");
    printf("-------------\n");
    printf("file_name = %s\n", file_name);
    printf("split_ratio = %lf\n", split_ratio);
    printf("inputs = %d\n", inputs);
    printf("hidden = %d, ", hidden);
    printf("\noutputs = %d\n", outputs);
    printf("population_size = %d\n", population_size);
    printf("mutation rate = %lf\n", mutation); */

        if (file_name == NULL)
        {
            fprintf(stderr, "No datafile found, use: -d <file>\n");
            exit(EXIT_FAILURE);
        }
        else
        {

            if (split_ratio == -1.0)
            {
                printf("Split ratio not found, use -s <decimal>\n");
                exit(1);
            };

            if (hidden == 0)
            {
                printf("Hidden layer neurones not found, use -h <int>\n");
                exit(1);
            };

            dataset *trainset;
            dataset *testset;

            trainset = load_dataset(file_name, split_ratio, &testset);

            if (trainset == NULL)
            {
                printf("Failed to parse data in file: %s to dataset\n", file_name);
                exit(1);
            }

            int i, j;

            printf("TRAINING SET\n");
            printf("--------------\n");
            for (i = 0; i < trainset->num_members; i++)
            {
                for (j = 0; j < trainset->num_inputs; j++)
                {
                    printf("%f ", (trainset->members + i)->inputs[j]);
                }
                for (j = 0; j < trainset->num_outputs; j++)
                {
                    printf("%f ", (trainset->members + i)->targets[j]);
                }
                printf("\n");
            }

            printf("\n\n");
            printf("TESTING SET\n");
            printf("--------------\n");
            for (i = 0; i < testset->num_members; i++)
            {
                for (j = 0; j < testset->num_inputs; j++)
                {
                    printf("%f ", (testset->members + i)->inputs[j]);
                }
                for (j = 0; j < testset->num_outputs; j++)
                {
                    printf("%f ", (testset->members + i)->targets[j]);
                }
                printf("\n");
            }

            printf("datasets loaded and ready\n");

            //=======================================================================================================

#ifdef GA

            // Get params, population_size size, and mutation rate

            if (population_size == 0)
            {
                printf("Population Size not found, use -p <int>\n");
                exit(1);
            }
            if (population_size <= 2)
            {
                printf("Population Size must be at least greater than 2.\n");
                exit(1);
            }
            if (population_size % 2 != 0)
            {
                printf("Population Size must be an even number.\n");
                exit(1);
            }

            if (mutation == 0.0)
            {
                printf("Mutation rate not found, use -m <double>\n");
                exit(1);
            }
            if (mutation <= 0.0)
            {
                printf("Mutation rate must be greater than 0.0\n");
                exit(1);
            }

            printf("Population: %d, Mutation: %f\n", population_size, mutation);

            // create population_size

            // make room for offsprings
            int p_size = population_size;             // number of parents
            population_size += (population_size / 2); // number of parent + offsprings

            // set mutation to chance
            int n = (mutation * 100);
            int d = 100;

            // create the first gen
            ann **population = malloc(population_size * sizeof(ann *));

            for (i = 0; i < population_size; i++)
            {
                population[i] = create(trainset->num_inputs, 1, hidden, trainset->num_outputs);
            }

            double ind_fitness;
            double *prediction;
            printf("calc fitness:");
            for (i = 0; i < population_size; i++)
            {

                ind_fitness = 0;
                // calculate the fitness over the train set data
                for (j = 0; j < trainset->num_members; j++)
                {
                    prediction = feedforward(population[i], (trainset->members + j)->inputs);
                    ind_fitness += fabs((trainset->members + j)->targets[0] - prediction[0]);
                }
                population[i]->fitness = ind_fitness;
                printf("%f ", population[i]->fitness);
            }
            printf("\n");

            int gen = 0;

            while (gen < 10000)
            {

                for (i = 0; i < population_size; i++)
                { // loop n times - 1 per element
                    for (j = 0; j < population_size - i - 1; j++)
                    { // last i elements are sorted already
                        if (population[j]->fitness > population[j + 1]->fitness)
                        { // swop if order is broken

                            swap(&population[j], &population[j + 1]);
                        }
                    }
                }

                printf("Generation: %d\n", gen);
                printf("------------------\n");
                printf("fitness: ");
                for (i = 0; i < population_size; i++)
                {
                    printf(" %f ", population[i]->fitness);
                }
                printf("\n");

                for (i = 0; i < p_size / 2; i++)
                {
                    for (j = 0; j < population[(2 * i)]->weights; j++)
                    {
                        if (rand() % 2 == 0)
                            population[p_size + i]->weight[j] = population[(2 * i)]->weight[j];
                        else
                            population[p_size + i]->weight[j] = population[(2 * i) + 1]->weight[j];
                        // population[p_size + i]->weight[j] = (population[(2 * i)]->weight[j] + population[(2 * i) + 1]->weight[j]) / 2.0;
                    }
                }

                for (i = 0; i < p_size / 2; i++)
                {
                    for (j = 0; j < population[p_size + i]->weights; j++)
                    {
                        if (n > (rand() % d))
                        {
                            population[p_size + i]->weight[j] = double_rand(-5.0, 5.0); // ANN_RANDOM() - 0.5;
                        }
                    }
                }

                double ind_fitness;
                double *prediction;
                for (i = 0; i < population_size; i++)
                {

                    ind_fitness = 0;
                    // calculate the fitness over the train set data
                    for (j = 0; j < trainset->num_members; j++)
                    {
                        prediction = feedforward(population[i], (trainset->members + j)->inputs);
                        ind_fitness += fabs((trainset->members + j)->targets[0] - prediction[0]);
                    }
                    population[i]->fitness = ind_fitness;
                }

                gen++;
            }

            int u;

            for (u = 0; u < 1; u++)
            {

                printf("Generation: %d\n", gen);
                printf("------------------\n");
                printf("fitness: ");

                printf(" %f ", population[u]->fitness);
                printf(" | W: ");
                for (j = 0; j < population[u]->weights; j++)
                {
                    printf("%f ", population[u]->weight[j]);
                }
                printf("\n");

                int matchs = 0;
                double delta = 0;

                printf("Validate Training inputs\n");
                for (i = 0; i < trainset->num_members; i++)
                {
                    prediction = (double *)feedforward(population[u], (trainset->members + i)->inputs);
                    delta = fabs(prediction[0] - (trainset->members + i)->targets[0]);

                    for (j = 0; j < trainset->num_inputs; j++)
                    {
                        printf("%.2f ", (trainset->members + i)->inputs[j]);
                    }

                    printf("T: %.2f O: %.2f\n", (trainset->members + i)->targets[0], prediction[0]);

                    if (delta < error_threshold)
                        matchs++;
                }

                printf("Accuracy on training set: %.2f%%\n", (matchs / (double)trainset->num_members * 100));

                matchs = 0;

                printf("Validate Testset inputs\n");
                for (i = 0; i < testset->num_members; i++)
                {
                    prediction = (double *)feedforward(population[u], (testset->members + i)->inputs);
                    delta = fabs(prediction[0] - (testset->members + i)->targets[0]);

                    for (j = 0; j < testset->num_inputs; j++)
                    {
                        printf("%.2f ", (testset->members + i)->inputs[j]);
                    }

                    printf("T: %.2f O: %.2f\n", (testset->members + i)->targets[0], prediction[0]);

                    if (delta < error_threshold)
                        matchs++;
                }

                printf("Accuracy on testing set: %.2f%%\n", (matchs / (double)testset->num_members * 100));
            }

            for (i = 0; i < population_size; i++)
            {
                ann_free(population[i]);
            }
            free(population);

#else

            ann *nn = create(trainset->num_inputs, 1, hidden, trainset->num_outputs);
            printf("start bp training...");

            double error;
            double *prediction;

            for (i = 0; i < 10000; i++)
            {

                for (j = 0; j < trainset->num_members; j++)
                {
                    ann_train(nn, (trainset->members + j)->inputs, (trainset->members + j)->targets, 0.1);
                }

                // Training Error
                error = 0;
                for (j = 0; j < trainset->num_members; j++)
                {
                    prediction = feedforward(nn, (trainset->members + j)->inputs);
                    error += fabs(((trainset->members + j)->targets[0] - prediction[0]));
                }
                error /= trainset->num_members;
                printf("epoch %d: Error %f\n", i, error);
            }

            printf("trained...\n");

            int matchs = 0;
            double delta = 0;

            printf("Validate Training inputs\n");
            for (i = 0; i < trainset->num_members; i++)
            {
                prediction = (double *)feedforward(nn, (trainset->members + i)->inputs);
                delta = fabs(prediction[0] - (trainset->members + i)->targets[0]);

                for (j = 0; j < trainset->num_inputs; j++)
                {
                    printf("%.2f ", (trainset->members + i)->inputs[j]);
                }

                printf("T: %.2f O: %.2f\n", (trainset->members + i)->targets[0], prediction[0]);

                if (delta < error_threshold)
                    matchs++;
            }

            printf("Accuracy on training set: %.2f%%\n", (matchs / (double)trainset->num_members * 100));

            matchs = 0;

            printf("Validate Testset inputs\n");
            for (i = 0; i < testset->num_members; i++)
            {
                prediction = (double *)feedforward(nn, (testset->members + i)->inputs);
                delta = fabs(prediction[0] - (testset->members + i)->targets[0]);

                for (j = 0; j < testset->num_inputs; j++)
                {
                    printf("%.2f ", (testset->members + i)->inputs[j]);
                }

                printf("T: %.2f O: %.2f\n", (testset->members + i)->targets[0], prediction[0]);

                if (delta < error_threshold)
                    matchs++;
            }

            printf("Accuracy on testing set: %.2f%%\n", (matchs / (double)testset->num_members * 100));

#endif
        }
    }
    return 0;
}