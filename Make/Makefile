CC      = mpicc
# -ffp-contract=off: no FMA fusing, so every P runs the exact same flops (bitwise check)
CFLAGS  = -O3 -march=native -std=c11 -ffp-contract=off -Wall -Wextra
LDLIBS  = -lm

SRC   = src
BUILD = build
OBJS  = $(BUILD)/main.o $(BUILD)/psor.o $(BUILD)/grid.o $(BUILD)/black_scholes.o

psor: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD):
	mkdir -p $@

$(BUILD)/main.o: $(SRC)/main.c $(SRC)/params.h $(SRC)/black_scholes.h $(SRC)/grid.h $(SRC)/psor.h
$(BUILD)/psor.o: $(SRC)/psor.c $(SRC)/psor.h $(SRC)/grid.h $(SRC)/params.h
$(BUILD)/grid.o: $(SRC)/grid.c $(SRC)/grid.h $(SRC)/black_scholes.h $(SRC)/params.h
$(BUILD)/black_scholes.o: $(SRC)/black_scholes.c $(SRC)/black_scholes.h $(SRC)/params.h

clean:
	rm -rf psor $(BUILD)

.PHONY: clean
