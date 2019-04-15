# gcamp-proccessor
Automatically process gcamp videos

## Installation (Windows)
Install python 3.7 from [here](https://www.python.org/downloads/)
- When installing make sure you check "Add Python 3.7 to Path". All other defaults should be fine.

Install git from [here](https://git-scm.com/download/win)

Open cmd prompt and run the following commands:
- Clone the repo
```
git clone https://github.com/tclarke104/gcamp-processor.git
```
- Change directory into project
```
cd gcamp-processor
```
- Install requirements
```
pip3 install -r requirements.txt
```

## Using program
- Use python to call main.py (make sure you changed directory into the project)
```
python main.py
```
- A window will pop up prompting you to select a file to process
- The program will run and then save an image with labels and an excel sheet with the data same directory as the video
