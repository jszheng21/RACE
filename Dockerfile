# Better use newer Python as generated code can use new features
FROM python:3.10-slim

# install git
RUN apt-get update && apt-get install -y git

# upgrade to latest pip
RUN pip install --upgrade pip

COPY . /race

RUN cd /race && pip install .[eval]

# Pre-install the dataset
RUN python3 -c "from race.dataloader import get_human_eval_plus, get_mbpp_plus; get_human_eval_plus(); get_mbpp_plus()"

WORKDIR /race

ENTRYPOINT ["python3", "-m"]
