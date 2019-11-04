#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "gaann.h"

dataset *loadData(char *filename)
{
    FILE *ptrDataFile;
    dataset *ptrDataset;
    int numInputs, numOutputs, numMembers;
    int i, j;
    double temp=0.0;
    char check = 0x00;

    /* Open filename */
    if ((ptrDataFile = fopen(filename, "r")) == NULL)
    {
        perror(NULL);
        return (NULL);
    }

    /* Load first line which contain settings */
    fscanf(ptrDataFile, "%d, %d, %d\n", &numMembers, &numInputs, &numOutputs);

    /* Setup the dataset */
    /* Allocate memory for the dataset */
    if ((ptrDataset = (dataset *)malloc(sizeof(dataset))) == NULL)
    {
        perror("Couldn't allocate the dataset\n");
        return (NULL);
    }

    /* Set the variables */
    ptrDataset->numMembers = numMembers;
    ptrDataset->numInputs = numInputs;
    ptrDataset->numOutputs = numOutputs;

    /* Allocate memory for the arrays in the dataset */
    if ((ptrDataset->members = (dataMember *)malloc(numMembers * sizeof(dataMember))) != NULL)
        check |= 0x01;

    if (check < 1)
    {
        printf("1.Couldn't allocate dataset\n");
        if (check & 0x01)
            free(ptrDataset->members);
        free(ptrDataset);
        return (NULL);
    }

    /* Get the data */
    /* For each Member */
    for (i = 0; i < numMembers; i++)
    {
        /* Allocate the memory for the member */
        check = 0x00;
        if (((ptrDataset->members + i)->inputs = (double *)malloc(numInputs * sizeof(double))) != NULL)
            check |= 0x01;
        if (((ptrDataset->members + i)->targets = (double *)malloc(numOutputs * sizeof(double))) != NULL)
            check |= 0x03;

        if (check < 3)
        {
            printf("2.Couldn't allocate dataset\n");
            /* Deallocate the previous loops */
            for (j = 0; j < i; j++)
            {
                free((ptrDataset->members + j)->inputs);
                free((ptrDataset->members + j)->targets);
            }

            if (check & 0x01)
                free((ptrDataset->members + i)->inputs);
            if (check & 0x02)
                free((ptrDataset->members + i)->targets);

            free(ptrDataset->members);
            free(ptrDataset);
            return (NULL);
        }

        /* Read the inputs */
        for (j = 0; j < numInputs; j++)
        {
            fscanf(ptrDataFile, "%lf, ", &temp);
            // printf("%lf, ", temp);
            (ptrDataset->members + i)->inputs[j] = temp;
        }

        /* Read the outputs */
        for (j = 0; j < numOutputs - 1; j++)
        {
            fscanf(ptrDataFile, "%lf, ", &temp);
            // printf("%lf, ", temp);
            (ptrDataset->members + i)->targets[j] = temp;
        }
        fscanf(ptrDataFile, "%lf\n", &temp);
        // printf("%lf\n", temp);
        (ptrDataset->members + i)->targets[j] = temp;
    }

    /* Make sure the file is closed */
    fclose(ptrDataFile);

    /* Finally, return the pointer to the dataset */
    return (ptrDataset);
}