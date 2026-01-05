# regular_expressions.py
"""Volume 3: Regular Expressions.
<Name>
<Class>
<Date>
"""

import re

# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile("python") # Easy

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$") # Pretty much just a backslash before every key character

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$") # Search for pairs of B/M/G with store/supplier

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python parameter definition.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^[A-Za-z_][\w_]*\s*((=\s*)(\d+(\.\d+)?)|(=\s*)('.*')|(=\s*)([A-Za-z_][\w_]*))?$") # Start with letter or _, then \w, then an optional space with = then another valid Python entry

# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    my_pattern = re.compile("(if)|(elif)|(else)|(for)|(while)|(try)|(except)|(finally)|(with)|(def)|(class)\s+") # Gross but it works
    my_list = re.split("\n", code) # Just run through line by line
    for i in range(len(my_list)):
        if bool(my_pattern.search(my_list[i])): # If it starts with a key word:
            my_list[i] = my_list[i] + ":" # Then add a colon to the end of the line
    return "\n".join(my_list) # Stitch lines back together

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """

    my_dict = {}
    name_pattern = re.compile(r"^[A-Z][A-Za-z]+\s([A-Z]\.\s)?[A-Z][A-Za-z]+") # The mid-name capitals got me
    email_pattern = re.compile(r"[A-Za-z][\w._]+@[\w.]+\.[\w]+") # Pretty much \w @ \w \. com
    birthday_pattern = re.compile(r"[\d\/]{6,10}") # This one is super loosey goosey: 6-10 char of digits and /
    phone_pattern = re.compile(r"(\+?\s?1[-\s])?(\(\d{3}\)|\d{3})-?\d{3}-?\d{4}") # I hate the +1 at the beginning

    phone_prefix = re.compile(r"^(1-?)?[-\s]?(\(\d{3}\)|\d{3})[\s-]?") # Used to parse the phone list out
    phone_body = re.compile(r"\d{3}-\d{4}") # Consistent phone body

    with open(filename) as db:
        for line in db:
            name = name_pattern.search(line).group() # Search for name
            email_match = email_pattern.search(line) # Search for email
            birthday_match = birthday_pattern.search(line) # Search for birthday
            phone_match = phone_pattern.search(line) # Search for phone

            if email_match:
                email = email_match.group() # No cleaning to do here :)
            else:
                email = None

            if birthday_match:
                month, day, year = birthday_match.group().split("/") # Gonna parse this by mm/dd/yyyy to get consistency
                if len(month) == 1:
                    month = "0" + month
                if len(day) == 1:
                    day = "0" + day
                if len(year) == 2:
                    year = "20" + year
                birthday = month + "/" + day + "/" + year # Stitch all back together
            else:
                birthday = None

            if phone_match:
                phone_temp = phone_match.group() # Initialize temporary phone number
                phone_temp = re.sub(r"\s", "", phone_temp)
                prefix = phone_prefix.search(phone_temp).group() # Find prefix
                prefix = "(" + re.sub(r'\D', '', prefix) + ")" # Gotta get prefix ready for output
                if len(prefix) == 6: # Get rid of extra 1 at the beginning
                    prefix = "(" + prefix[2:]

                body = phone_body.search(phone_temp).group() # Straightforward body search

                phone = prefix + body # Stitch back together
            else:
                phone = None

            my_dict[name] = {"birthday":birthday, "email":email, "phone":phone} # Make dictionary

    return my_dict
