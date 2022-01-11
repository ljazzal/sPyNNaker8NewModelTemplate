MODELS = QIF_curr_delta \
	QIF_curr_alpha \
	QIF_curr_exp

all:
	for d in $(MODELS); do $(MAKE) -C $$d || exit $$?; done

clean:
	for d in $(MODELS); do $(MAKE) -C $$d clean || exit $$?; done
