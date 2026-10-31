# Raw data

`uci_dropout_success.csv` is the UCI Machine Learning Repository dataset
**"Predict Students' Dropout and Academic Success"** (dataset 697).

- 4,424 students from several undergraduate degrees at the Polytechnic Institute of Portalegre (Portugal)
- Authors: M.V. Martins, D. Tolledo, J. Machado, L.M.T. Baptista, V. Realinho (2021);
  paper: V. Realinho, J. Machado, L. Baptista, M.V. Martins, *Predicting Student Dropout and
  Academic Success*, Data 7(11):146, 2022
- Licence: CC BY 4.0 - https://archive.ics.uci.edu/dataset/697
- The file is unchanged except for removing the byte-order mark and stray whitespace in column names.
  Two independent GitHub copies were compared and are identical.

`python app/data/prepare_real_data.py` turns it into the app's `sample_students.csv`.
