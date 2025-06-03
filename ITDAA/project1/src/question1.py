#  1.1      Use Python to perform the appropriate data cleaning and preprocessing steps 
# needed to get your data into a more usable format.


import sqlite3 # for the .db file reading 
import pandas as pd
from pathlib import Path

db_path = Path(r"..\Project\data\Github.db")

# Open the connection to the database 
def open_file():
 return sqlite3.connect(db_path)

db_pool = open_file()

columns = ["YearsCode", "MainBranch", "Country", "EdLevel", "LanguageHaveWorkedWith", "LanguageWantToWorkWith",
    "DatabaseHaveWorkedWith", "DatabaseWantToWorkWith", "Age"]

cols_str = ",".join(columns)

table2021 = pd.read_sql_query(f"SELECT {cols_str} FROM data_2021", db_pool)
table2022 = pd.read_sql_query(f"SELECT {cols_str} FROM data_2022", db_pool)
table2023 = pd.read_sql_query(f"SELECT {cols_str} FROM data_2023", db_pool)

table2021["Year"] = 2021
table2022["Year"] = 2022
table2023["Year"] = 2023

#Close connection
db_pool.close() 

        # 1.1.1 ) Combine the data from all three years into a single dataset by selecting only the relevant 
# columns and adding an indicator column to specify the year of each survey

combinedTables = pd.concat([table2021, table2022, table2023], ignore_index=True)
print("Print first answer")


        # 1.1.2 )  Standardize the Country column, ensuring consistency in naming for countries with 
# variations

country_mapping = {
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom", "United States of America": "United States",
    "Iran, Islamic Republic of...": "Iran", "Russian Federation": "Russia","Venezuela, Bolivarian Republic of..." :"Venezuela"
    , "The former Yugoslav Republic of Macedonia": "North Macedonia","Micronesia, Federated States of...": "Micronesia", "Republic of Korea": "Korea", "Hong Kong (S.A.R.)"
    :"Hong Kong" , "Congo, Republic of the...": "Congo" ,"United Republic of Tanzania" : "Tanzania" , "Lao People's Democratic Republic":"Laos", "Republic of Moldova" :"Moldova" ,
    "Syrian Arab Republic" : "Syria" , "Czech Republic": "Czechia" , "United Arab Emirates": "UAE", "United KingdomUnited Kingdom" : "UK", "Viet Nam": "Vietnam"
}

combinedTables["Country"] = combinedTables["Country"].replace(country_mapping)


        # 1.1.3)  Handle missing data and subsetting the dataset to include a manageable selection of countries to facilitate clear visualizations and analysis.
#  Justify your choices for missing value treatment and country selection

    # Filter the countries 
selected_countries = ["United States", "India", "Germany", "United Kingdom", "Canada", 
                    "France", "Brazil", "Netherlands", "Australia", "Italy", 
                    "Spain", "Russia", "Mexico", "South Africa", "Vietnam"]

ctables_filtered = combinedTables[combinedTables["Country"].isin(selected_countries)]


    # Handle missing data 
ess_coloumns = ["Country" , "MainBranch", "YearsCode" , "EdLevel" , "Age"] # essential coloumns 

# drop the rows that have missing values in the essential coloumns
droped_rtable = ctables_filtered.dropna( how='any')
print("Droped and filtered table :")
print(droped_rtable)



        # 1.1.4)  Categorize the values in the MainBranch column into meaningful groups to 
    # provide a clear understanding of respondent roles.

replace_mb = { 
    "I code primarily as a hobby": "Hobbyist",
    "I am a developer by profession": "Professional Developer",
    "I am not primarily a developer, but I write code sometimes as part of my work": "Occasional Coder",
    "I am a student who is learning to code": "Student",
    "I used to be a developer by profession, but no longer am": "Former Developer",
    "I am not primarily a developer, but I write code sometimes as part of my work/studies": "Occasional Coder",
    "I am learning to code": "Self-taught"
}

droped_rtable = droped_rtable.replace(replace_mb)


        # 1.1.5)  Reclassify the EdLevel column into broader educational 
    # categories to create concise, interpretable groupings.

replace_el = {
    "Primary/elementary school": "Primary Education",
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": "Secondary Education",
    "Some college/university study without earning a degree": "Tertiary Education",
    "Associate degree (A.A., A.S., etc.)": "Associate Degree",
    "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": "Bachelor’s Degree",
    "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": "Master’s Degree",
    "Other doctoral degree (Ph.D., Ed.D., etc.)": "Doctorate / PhD",
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": "Professional Degree",
    "Professional degree (JD, MD, etc.)" : "Professional Degree",
    "Something else": "Other"
}

clean_data = droped_rtable.replace(replace_el)
    
####
# Display output 
####

# create clean data .db file for visualization 
conn = sqlite3.connect('..\Project\data\Q1_Output.db')
clean_data.to_sql('clean_data', conn , if_exists='replace', index=False)
conn.close()

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(droped_rtable.to_string())  

print("Final Result:")
print(clean_data.head(20))

