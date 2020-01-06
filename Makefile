# Defines
CC         = gcc
OBJ_DIR    = ./obj
SRC_DIR    = ./src
INCL_DIR   = ./include
OBJECTS    = $(addprefix $(OBJ_DIR)/, ann.o)
INCLUDES   = $(addprefix $(INCL_DIR)/, ann.h)
CFLAGS     = -g -Wall
EXECUTABLE = main

# Generate the executable file
$(EXECUTABLE): main.c $(OBJECTS)
	$(CC) $(CFLAGS) $< $(OBJECTS) -o $(EXECUTABLE) -I $(INCL_DIR) -lm

# Compile and Assemble C source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c $(INCLUDES)
	$(CC) $(CFLAGS) -I $(INCL_DIR) -c $< -o $@

# Clean the generated executable file and object files
clean:
	rm -f $(OBJECTS)
	rm -rf $(EXECUTABLE)*

