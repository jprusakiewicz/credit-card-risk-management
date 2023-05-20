.PHONY: download-bash


URL := $(shell cat data/data_url.txt)
download-bash:
	wget -O data/credit_cards.xls $(URL)