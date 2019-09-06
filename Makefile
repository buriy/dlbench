cpu: .venv/bin/python data/sample .venv/bin/all_set
	@echo "Running multithread tests"
	sync; sudo su -c "echo 3 > /proc/sys/vm/drop_caches"
	@.venv/bin/python -m tests.cpu_preprocess
	@echo "CPU `grep 'model name' /proc/cpuinfo|head -n1`"
	@echo -n "Versions: `.venv/bin/python --version`, "
	@echo -n "`.venv/bin/pip freeze|grep -i torch`, "
	@echo -n "`.venv/bin/pip opencv|grep -i opencv`, "
	@echo "`.venv/bin/pip freeze|grep -i albumentations`"
cpu-single: .venv/bin/python
	@echo "Running simplethread tests"
	sync; sudo su -c "echo 3 > /proc/sys/vm/drop_caches"
	@.venv/bin/python -m tests.cpu_preprocess --ncore 1 --start 5 --finish 20
	@sleep 5
	sync; sudo su -c "echo 3 > /proc/sys/vm/drop_caches"
	@.venv/bin/python -m test.pystone 5000000
data/sample.zip: .venv/bin/python
	mkdir -p data
	.venv/bin/pip install gdown
	.venv/bin/gdown https://drive.google.com/uc?id=1OXr16vkpXVlAuUovYPsIxhohhp-mgnah -O data/sample.zip
data/sample/: data/sample.zip
	cd data; unzip sample.zip
.venv/bin/python:
	virtualenv --python=python3 .venv
.venv/bin/all_set:
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m tests.cpu_preprocess --start 0 --finish 1 --ncore 0 --ntimes 1
	@echo "All set">.venv/bin/all_set