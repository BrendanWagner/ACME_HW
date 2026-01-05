"""Volume 1: SQL 2.
<Name>
<Class>
<Date>
"""

import sqlite3 as sql


# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    data = cur.execute("SELECT SI.StudentName " # All we want is the student names
                       "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG " # We want it based on grades
                       "ON SI.StudentID == SG.StudentID " # Attach it by Student ID
                       "WHERE SG.Grade == 'B'").fetchall() # Where it's a B
    return [i[0] for i in data] # Un-tuple data

# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    data = cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade " # The tuple we want
                       "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI " # This is how we'll get the major names
                       "ON MI.MajorID == SI.MajorID " 
                       "INNER JOIN StudentGrades AS SG "  # Attach grades
                       "ON SI.StudentID == SG.StudentID "
                       "INNER JOIN CourseInfo as CI " # Attach course names so we can find 'Calculus'
                       "ON CI.CourseID == SG.CourseID " 
                       "WHERE CI.CourseName == 'Calculus'").fetchall()
    return data


# Problem 3
def prob3(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    data = cur.execute("SELECT MI.MajorName, COUNT(*) as student_count "
                       "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                       "ON MI.MajorID == SI.MajorID "
                       "GROUP BY MI.MajorID "
                       "ORDER BY student_count DESC, MI.MajorName ASC").fetchall()
    return data


# Problem 4
def prob4(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    conn = sql.connect(db_file)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS GPA")
    cur.execute("CREATE TABLE GPA (Grade TEXT, Points REAL)")
    rows = [("A+",4.0), ("A",4.0), ("A-",3.7), ("B+",3.3), ("B",3.0), ("B-",2.7), ("C+",2.3), ("C",2.0), ("C-",1.7), ("D+",1.3), ("D",1.0), ("D-",0.7), ("F",0.0)]
    cur.executemany("INSERT INTO GPA VALUES(?,?);", rows)
    data = cur.execute("""
    SELECT 
        new_GPA.StudentName, 
        COUNT(*) AS CourseCount,
        AVG(GPA.Points) AS AvgPoints
        
    FROM (
        SELECT *
        FROM StudentInfo AS SI
        INNER JOIN StudentGrades AS SG
        ON SG.StudentID = SI.StudentID
    ) AS new_GPA
    INNER JOIN GPA
    ON new_GPA.Grade = GPA.Grade
    GROUP BY new_GPA.StudentName
    ORDER BY AvgPoints DESC
    """).fetchall() # Gonna be transparent this was formatted by ChatGPT and I like it way more than whatever the heck I was gonna do
    cur.execute("DROP TABLE GPA")
    return data

# Problem 5
def prob5(db_file="mystery_database.db"):
    """Use what you've learned about SQL to identify the outlier in the mystery
    database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): outlier's name, outlier's ID number, outlier's eye color, outlier's height
    """
    # I imagine that my exploring this db is gonna be super hard to document so what you'll find here is just the end product.
    conn = sql.connect(db_file)
    cur = conn.cursor()
    # data = cur.execute("PRAGMA table_info(Table_3)").fetchall()
    # I'm gonna guess he's from Earth. So we'll just find anybody whose home planet is Earth
    # data = cur.execute("SELECT * FROM Table_2 WHERE description LIKE '%Earth%'").fetchall() # Found out his name is William T. Riker, ID number 830744
    # data = cur.execute("SELECT * FROM Table_1 WHERE name LIKE 'William%'").fetchall() # His eyes are 'Hazel-blue'
    # data = cur.execute("SELECT * FROM Table_3 WHERE eye_color LIKE 'Hazel-blue'").fetchall() # Found out ('Male', '1.93', 'Hazel-blue', 'Light', 'Dark brown', '87 kilograms')
    return ["William T. Riker", "830744", "Hazel-blue", "1.93"]