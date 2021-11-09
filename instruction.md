# DOES instructions

## ROPIS Dataset acquisition

To download the ROPIS dataset please follow the link "" or send an email to the authors.
The ROPIS directory will contain 3 folders (Train, Validation and Test) and *needs to be placed in the main DOES directory*. 

## Training and testing phase

You can directly run DOES using the following line:

```Python
python does_main.py --bright 0.5 --hue 0.5 --contrast 5 --saturation 5 --pitch 1 --roll 1 --model resnet18
```

This will run DOES with the default setting used by the authors
