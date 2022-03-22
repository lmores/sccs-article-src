# Input

## Data set files

The `datasets/` folder contains the files that describe each instance.
Since data set files from different sources may have different formats, the `datasets/` folder is organised in various subfolders: each of them is named after the place/source where the instances it contains have been retrieved.

Names of top level folders inside `datasets/` are used to detect the format to read the data set files they contain, therefore they should not be changed without accordingly modifying the source code.
Each of these folders may be further organised into subfolders without restrictions.

Currently the only folder inside `datasets/` is the `beasley` folder that contains a modified version of the data set files available at [Beasley's web page](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html) that have originally been created for the classical set covering problem.
The new data set files have the same format described on Beasley's web page and they have been obtained merging each consecutive triple of columns (each representing a subset) into a single one with cost equal to the sum of the three original columns.
For this reason the name of each data set files describing these new instances has been appended with '-3'.
For details about how conflict costs are chosen please refer to our paper.

```txt
datasets/
    |-- beasley/
        |-- ...
        |-- scpa1-3.txt
        |-- scpa2-3.txt
        |-- scpa3-3.txt
        |-- ...
```

Note that inside Beasley's original data set files both the elements and the subsets are represented by integer numbers starting from 1 and the same holds true for the modified data set files we generated starting from the original ones.
However, when the data set files are loaded into the program the indexes of both the elements and the subsets are decreased by one, so that they start at 0 (for the sake of simplicity).
All the indexes inside the output files follow this last convention.
