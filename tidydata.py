# Initial imports and environment setting.

# matplotlib inline
import csv;
import numpy as np;
import pandas as pd;

import matplotlib.pyplot as plt;
plt.rcParams.update({'font.size': 16});

#import seaborn as sns;
#sns.set(context='talk', style='white');

import warnings;
warnings.filterwarnings('ignore');

def parse_response(s):
    try:
        return s[~s.isnull()][0];
    except IndexError:
        return np.nan;

###############################################################################
# AMW's approach

# Load in the data. This spreadsheet is (almost) direct from SurveyMonkey,
# though IP addresses, access dates, and free-text responses were scrubbed
# to anonymize respondents.

df = pd.read_csv('public_survey_data.csv', sep=',');
# import pandas as pd;df = pd.read_csv('test_survey.csv',sep=',');
# SurveyMonkey provides an odd nesting of responses when exporting results.
# We'd like to convert this structure to a pandas MultiIndex data frame.
# First, let's find question indices -- adapted from https://stackoverflow.com/a/49584888

indices   = [ i for i, c in enumerate(df.columns) if not c.startswith('Unnamed') ];
questions = [ c for c in df.columns if not c.startswith('Unnamed') ];
slices    = [ slice (i, j) for i, j in zip(indices, indices[1:] + [None])];
responses = [];

# check if header rows and (first) answers are aligned
# for q in slices:
#    print(df.iloc[:, q])  # Use `display` if using Jupyter

# merge extra multi-answer columns
df.replace ( np.nan,';',regex=True, inplace=True );
for i in range( 1, df.shape[1] ):
    if ( i in indices ):
        # print ('{} is a question'.format(i));
        lastq = i;
        responses.append ( [ df.iloc[ 0, i ] ] );
    else:
        # print ('attaching {} to question {}'.format(i,lastq));
        df.iloc[ :, lastq ] = df.iloc[ :, lastq ].astype(str) + ';' + df.iloc [ :, i ].astype(str);
        responses[ -1 ].append ( df.iloc[ 0, i ] );

# drop the copied columns
for i in reversed ( range ( 1, df.shape[1] ) ):
    if ( not i in indices ): 
        del df[df.columns[i]];

df.replace ( ';nan',';',regex=True, inplace=True ); 
df.replace ( ';;','',regex=True, inplace=True ); 
df.replace ( ';','',regex=False, inplace=True ); 
df.replace ( '"','',regex=True, inplace=True ); 
df.to_csv  ( 'tidydata.csv', quoting = csv.QUOTE_NONE, quotechar="",  escapechar="\\" );

# There are examples on StackOverflow etc. that deal with SurveyMonkey data.
# Problem for us with those is that none of them use multi-answer questions.
# (our data still has extra columns for those, causing column misalignment)
# 
# The code above replaces separators between non-question columns 
# with a semicolon, so that a CSV parser keeps them in one 'cell'
#
# Now loading tidydata.csv at least aligns the question/response columns, but
# CSV/Excel datasets cannot distinguish between levels: extra parsing is now 
# required for the semicolon-separated responses, without breaking alignment.
#
# Next: separate reponses, store in nested structure (e.g., JSON?) 

df = df.applymap( lambda x: x.split( ';' ) if isinstance( x, str ) else x );
# for row in range ( df.shape [ 0 ] ):
#     for col in range ( df.shape [ 1 ] ):
#         value = df.iloc[row][col];        
#         if ( type ( value ) == str ):
#             value = value.split( ';' );
#         df.iloc[row, col] = value; 
df.to_json ( 'tidydata.json', indent = 4 );
    
# gather the options for each question
options = df.iloc[:1].values[0].tolist(); 
alloptions = [ ];
for i in range ( 1, len ( options ) ):
    
    alloptions.append ( list ( set ( options [ i ] + [ item for sublist in df.iloc[ 1:, i ] for item in sublist ] ) ) );
    if ( '' in alloptions [ -1 ] ):
        alloptions [ -1 ].remove ( '' );
    if ( 'Response' in alloptions [ -1 ] ):
        alloptions [ -1 ].remove ( 'Response' );
    
# so now there's a list (of the same length as df's #columns) with questions
# and a list of lists ( of strings ) of the same length with all possible/given responses

# In [1]: runfile('.../2022-community-survey/tidydata.py', wdir='.../2022-community-survey')
# In [2]: len(questions)
# Out[2]: 24
# In [3]: len(alloptions)
# Out[3]: 24
# In [4]: questions[1]
# Out[4]: 'What geographic region are you currently located in?'
# In [5]: alloptions[1]
# Out[5]: 
# ['North America (Including Canada and Mexico)',
#  'Europe',
#  'Oceania (Australia and New Zealand)',
#  'Middle East',
#  'South America (Including Central America)',
#  'Asia']

# and now we're back to Elizabeth's code
matched_questions = pd.MultiIndex.from_arrays ( [tuple(questions), tuple(tuple(sub) for sub in alloptions)], names = ( 'question', 'options' ) );






###############################################################################
# Simon's approach
survey_data = pd.read_csv('public_survey_data.csv', sep=',')
survey_clean = survey_data.copy()

ID_cols = ['Unnamed: 0', 'Are you a member of OHBM?', 'What geographic region are you currently located in?', 'What is your current career status?']

multi_Q1 = 'Which of the following platforms do you use to access OHBM content? When applicable, a direct link to the platform is provided next to each option. Please check all options that apply.'
multi_Q2 = 'Do you currently follow any of the following OHBM Special Interest Groups (SIG) platforms? When applicable, a direct link to the platform is provided next to each option. Please check all options that apply.'
multi_Q3 = 'How important is each of these types of content to you?'

multi_Qs = [multi_Q1, multi_Q2, multi_Q3]
unnamed_range = [(13, 21), (41, 47), (49, 55)]

orig_name = []
new_name = []

# Rename unnamed columns to question + response option
for q, ur in zip(multi_Qs, unnamed_range):
    unnamed = [q] + [f'Unnamed: {i}' for i in range(ur[0], ur[1] + 1)]
    renamed = [q + '@@' + survey_data.loc[0, un] for un in unnamed]
    orig_name.extend(unnamed)
    new_name.extend(renamed)


# Rename columns
rename_dict = {i : j for i, j in zip(orig_name, new_name)}
survey_clean.rename(columns=rename_dict, inplace=True)

# Drop response description (i.e. first row)
survey_clean.drop(0, inplace=True)

# Make long format:
survey_long = survey_clean.melt(id_vars=ID_cols, var_name='questions', value_name='response')
survey_long.sort_values('Unnamed: 0', inplace=True)

# This removes the response options from the concatenated question string, but makes pivoting really complicated.
survey_long['questions'] = survey_long['questions'].str.split('@@').str[0]
group_plot = survey_long.query('questions == @multi_Q1').groupby([ID_cols[2], 'response']).size().unstack(fill_value=0)
group_plot.transpose().plot.bar()
plt.title(multi_Q1)
plt.show()

survey_wide = survey_long.copy()
survey_wide['Unnamed: 0'] = survey_wide['Unnamed: 0'].astype("string")
# Drop nans
survey_wide.dropna(inplace=True, subset=['response'])
survey_wide = survey_wide.groupby(ID_cols + ['questions']).agg({'response': lambda x: list(x)}).reset_index().copy()
survey_wide['new_index'] = survey_wide[ID_cols].apply(lambda row: '@@'.join(row.values.astype(str)), axis=1)
survey_wide.drop(columns=ID_cols, inplace=True)
survey_wide = survey_wide.pivot('new_index', columns=['questions'], values='response')
survey_wide.reset_index(inplace=True)
for n, id in enumerate(ID_cols):
    survey_wide[id] = survey_wide['new_index'].str.split('@@').str[n]

survey_wide['Unnamed: 0'] = survey_wide['Unnamed: 0'].astype("int")

survey_wide.drop(columns='new_index', inplace=True)
survey_wide.sort_values('Unnamed: 0', inplace=True)


