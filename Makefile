cpu: .venv/bin/python
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m tests.cpu_preprocess
data/sample.zip:
	mkdir -p data
	gdown https://drive.google.com/uc?id=1OXr16vkpXVlAuUovYPsIxhohhp-mgnah -O data/sample.zip
sample: data/sample.zip
	unzip data/sample.zip
.venv/bin/python: 
	virtualenv --python=python3 .venv
	.venv/bin/pip install -r requirements.txt
