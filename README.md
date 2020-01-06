# Artificial Nerual Network (ANN) trained with Genetic Algorithm

This project is my attempt in finiding and implementing a general optimisation method using genetic algorithm.

## Getting Started

### Build Environment

This project is made and build on ubutntu 18, more specifically on windows 10's WSL environment.

### Data File Format

The program can accept any file that has data in the following format

```
<no of lines> <inputs> <output>
<input1> <input2> ... <inputN> <output1>
<input1> <input2> ... <inputN> <output1>
<input1> <input2> ... <inputN> <output1>
<input1> <input2> ... <inputN> <output1>
<input1> <input2> ... <inputN> <output1>
```

***Note: Program can parse more outputs but the main.c have hardcoded output and prediction to 1, change the code accordingly if you wish***


### Running it
```
git clone https://github.com/mdhthahmd/gaann.git
cd gaann
make
```

This will output a runnable called main, and to run this enter
```
./main
```
The nessesary help doc will appear, Happy demoing...

### Extra
As a reference point to comapre my optimisation method i have also implemented the ANN training with the conventional back propagation, to activate this feature: you need to modify the **12th** line in **main.c** to
```
#if 0
```
and recompile by issuing th command ```make``` 
and run ```./main``

### Thanks
The Network is implemented by refering the following repo, and i would like to extend thanks to its author as it helped me a lot in this project
```
https://github.com/codeplea/genann.git
```

