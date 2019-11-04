#include <stdio.h>

#include "gaann.h"

int main () {

    dataset *dataset = loadData("data/data3.txt");
    printf("%d, %d, %d\n" , dataset->numMembers, dataset->numInputs, dataset->numOutputs);

    int i, j, k;
    for( i = 0; i < dataset->numMembers; i++ ){
        for(j = 0; j < dataset->numInputs; j++ ){
             printf("%.1f, ", (dataset->members+i)->inputs[j]);
        }
        for(j = 0; j < dataset->numOutputs-1; j++ ){
            printf("%.1f, ", (dataset->members+i)->targets[j]);
        }
        printf("%.1f\n", (dataset->members+i)->targets[j]);
    }

    // print the dataset

}