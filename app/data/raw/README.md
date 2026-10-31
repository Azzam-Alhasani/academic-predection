# Raw data

`student-mat.csv` (395 students, Mathematics) and `student-por.csv` (649 students, Portuguese language)

**Source:** P. Cortez and A. Silva, *Using Data Mining to Predict Secondary School Student Performance*,
in A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008),
pp. 5-12, Porto, Portugal, April 2008. UCI Machine Learning Repository, dataset 320.
Licence: CC BY 4.0.

Two Portuguese secondary schools, school year 2005-2006. The files hold many survey questions
(family, free time, alcohol ...). **This project deliberately uses only four things:**

| Used | Original column | Meaning |
|---|---|---|
| Term 1 mark | `G1` | first-period grade, 0-20 |
| Term 2 mark | `G2` | second-period grade, 0-20 |
| Absences | `absences` | number of school absences |
| Final mark | `G3` | final grade, 0-20 (the thing to predict) |

plus the course (Mathematics / Portuguese). Everything else is ignored. `age`, `sex`, `school`, `address`,
`famsize`, `Pstatus`, `Medu`, `Fedu`, `Mjob`, `Fjob`, `reason`, `nursery` and `internet` are used only to
recognise that a student appears in both files (382 students do), so that both rows are kept on the same
side of every train/test split.
