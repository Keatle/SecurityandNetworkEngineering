import pandas as pd
import sqlite3 
#import vader  
import matplotlib.pyplot as plt
from pathlib import Path

#2.1 Use visualizations and descriptive statistics to explore the dataset and answer the following questions. 
# Provide clear insights to support your analysis.

path = Path(r"..\Project\data\Q1_Output.db")

# 1. Demographics and Roles
            # 1. How has the number of users in each MainBranch category changed over three years?
            # Users in MainBranch 

conn = sqlite3.connect(path)

    # Display X : num users in MainBranch , Y : Years ( 2022 , 2021 , 2023)

df = pd.read_sql_query("SELECT * FROM clean_data", conn)
df['Year']= df['Year'].astype(str)

df_graph = pd.crosstab(df['MainBranch'], df['Year'])
df_graph.plot(kind='bar', figsize=(10,7))

plt.title("MainBranch Count per Year")
plt.xlabel("Main Branch")
plt.ylabel("Count")
plt.legend(title="Year")
plt.tight_layout()
plt.show()


        # 2. What are the trends in users' Age, Country, EdLevel, and YearsCode 
        # across different MainBranch categories?

#          Age

# Ensure Year is treated as a string or int (based on your DB structure)
df['Age'] = df['Age'].astype(str)

age_graph = pd.crosstab(df['MainBranch'], df['Age'])
age_graph.plot(kind='bar', figsize=(10, 7))

plt.title("Age groups across MainBranch categories")
plt.xlabel('Users in MainBranch')
plt.ylabel('Count')
plt.legend(title="Age")
plt.tight_layout()
plt.show()


#          Country 
country_graph =pd.crosstab(df['MainBranch'], df['Country'])
country_graph.plot(kind= 'bar', figsize=(10, 7))

plt.title("Countries across MainBranch categories")
plt.xlabel('Users in MainBranch')
plt.ylabel('Country')
plt.legend(title ='Age')
plt.tight_layout()
plt.show()


#          EdLevel
edlevel_graph = pd.crosstab(df['MainBranch'], df['EdLevel'])
edlevel_graph.plot(kind='bar', figsize=(10,7))

plt.title("EdLevel indiduals across MainBranch categories")
plt.xlabel('Users in MainBranch')
plt.ylabel('Count')
plt.legend(title ='EdLevel')
plt.tight_layout()
plt.show()


#         YearsCode

                         # Filter the years into categories of 10s

def clean_years_code(val):
    if val == 'Less than 1 year':
        return 0
    elif val == 'More than 50 years':
        return 51
    try:
        return float(val)
    except:
        return None

                        #clean the data set 
df['YearsCodeCleaned'] = df['YearsCode'].apply(clean_years_code)

bins = [0, 10, 20, 30, 40, 50, float('inf')]
ranges = ['0-9' , '10-19', '20-29','30-39','40-49','50+']

df['YearsCodeGroup'] = pd.cut(df['YearsCodeCleaned'], bins=bins , labels=ranges, right=False)


yearscode_graph =pd.crosstab(df['MainBranch'], df['YearsCodeGroup'])
yearscode_graph.plot(kind='bar', figsize=(10,7))


plt.title("YearsCode of indiduals across MainBranch categories")
plt.xlabel('MainBranch Category')
plt.ylabel('Count')
plt.legend(title='Years of coding experience')
plt.show()

                #  2. Technology Trends and Preferences

# 1. What are the top 5 most popular databases and programming languages 
# that GitHub users currently use and want to use in the future

       ### Databases ###

# top 5 currently used 
df['DatabaseHaveWorkedWith'] =(
                                df['DatabaseHaveWorkedWith']
                               .astype(str)
                               .str.replace(r"[\[\],]", "" , regex=True)
                               .str.split(";")
                            )

dbww_count = df.explode('DatabaseHaveWorkedWith')
dbww_count['DatabaseHaveWorkedWith'] = dbww_count['DatabaseHaveWorkedWith'].str.strip()

dbww_count = dbww_count['DatabaseHaveWorkedWith'].head(5).value_counts()
dbww_count.plot( kind='bar' , figsize= (10,7), color=('yellow'))

plt.title('Top 5 Databases worked with')
plt.xlabel('Database')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# top 5 want to use in future

             # regex for [ , ]
df['DatabaseWantToWorkWith'] =(
                                df['DatabaseWantToWorkWith']
                               .astype(str)
                               .str.replace(r"[\[\],]", "" , regex=True)
                               .str.split(";")
                            )

db_count = df.explode('DatabaseWantToWorkWith')
db_count['DatabaseWantToWorkWith'] = db_count['DatabaseWantToWorkWith'].str.strip()

db_count = db_count['DatabaseWantToWorkWith'].value_counts().head(5)
db_count.plot(kind='bar', figsize=(10,7), color='red')

plt.title('Top 5 Databases desrired')
plt.xlabel('Database')
plt.ylabel('Count')
plt.xticks(rotation= 45, ha= 'right')
plt.tight_layout()
plt.show()
 
       ### LANGUAGES ###

# top 5 currently used 

# ----> 1. Retrieve the count of LanguageHaveWorkedWith coloumn
df['LanguageHaveWorkedWith'] = df['LanguageHaveWorkedWith'].astype(str).str.split(';')

countTable = df.explode('LanguageHaveWorkedWith') # explode lists into seperate rows
countTable['LanguageHaveWorkedWith'] = countTable['LanguageHaveWorkedWith'].str.strip() # remove white space

language_counts = countTable['LanguageHaveWorkedWith'].head(5).value_counts()
language_counts.plot(kind="bar", figsize=(10,7) , color="skyblue")

plt.title("Popular language worked with ")
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation= 45, ha='right')
plt.tight_layout()
plt.show()

# top 5 want to use in future
df['LanguageWantToWorkWith'] = df['LanguageWantToWorkWith'].str.split(';')

countTable = df.explode('LanguageWantToWorkWith')
countTable['LanguageWantToWorkWith'] = countTable['LanguageWantToWorkWith'].str.strip()

lang_count = countTable['LanguageWantToWorkWith'].head(5).value_counts()
lang_count.plot(kind="bar" , figsize=(10,7), color="red")

plt.title("Popular language desired")
plt.xlabel('Lanuage')
plt.ylabel('Count')
plt.xticks(rotation= 45, ha='right')
plt.tight_layout()
plt.show()


# 2. How have the usage trends of the top 5 databases and programming languages 
# changed over three years?


       ### Databases ###
# top 5 currently used 
dbww = df.explode('DatabaseHaveWorkedWith')
dbww['DatabaseHaveWorkedWith'] = dbww['DatabaseHaveWorkedWith'].astype(str).str.strip()

dbww_top5_set = dbww['DatabaseHaveWorkedWith'].value_counts().head(5).index.tolist()

dbww_top5 = dbww[dbww['DatabaseHaveWorkedWith'].isin(dbww_top5_set)]
dbww_graph = pd.crosstab(dbww_top5['DatabaseHaveWorkedWith'], dbww_top5['Year'].astype(str))

dbww_graph.plot( kind='bar' , figsize= (10,7))

plt.title('Top 5 Databases worked with')
plt.xlabel('Database')
plt.ylabel('Count')
plt.xticks(rotation= 45, ha='right')
plt.tight_layout()
plt.show()

       ### Languages ###
lww = df.explode('LanguageHaveWorkedWith')
lww['LanguageHaveWorkedWith'] = lww['LanguageHaveWorkedWith'].astype(str).str.strip()

lww_top5_set = lww['LanguageHaveWorkedWith'].value_counts().head(5).index.tolist()

lww_top5 = lww[lww['LanguageHaveWorkedWith'].isin(lww_top5_set)]
lww_graph = pd.crosstab(lww_top5['LanguageHaveWorkedWith'], lww_top5['Year'].astype(str))


lww_graph.plot( kind='bar' , figsize= (10,7))

plt.title('Top 5 Languages worked with')
plt.xlabel('Language')
plt.ylabel('Count')
plt.xticks(rotation= 45, ha='right')
plt.tight_layout()
plt.show()


#    3. Relationship Analysis
# 1. How does YearsCode correlate with the use of the top 5 databases and programming languages currently in use?
# Focus on clarity, relevance, and insightful analysis in your visualizations and interpretations

# YearsCode x Languages 

yclang_graph = pd.crosstab(dbww_top5['DatabaseHaveWorkedWith'],dbww_top5['YearsCodeGroup'])

yclang_graph.plot( kind='bar', figsize= (10,7))

plt.title('Correlation of Yearscode with Programming languages used')
plt.xlabel('Language')
plt.ylabel('Years Code')
plt.xticks(rotation= 45, ha= 'right')
plt.tight_layout()
plt.show()



#  2.2. Based on your above analysis, provide a short report detailing your findings / insights derived from the data that may be useful to GitHub to better understand their users.
#  Additionally, provide recommendations on
# improvements or ideas to maintain their current users and attract new ones. Ensure that your findings are relevant based on your analysis and visualizations, and that they are articulated well. Your recommendations should be
# innovative, detailed and actionable





