# Initial imports and environment setting.

# matplotlib inline
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

import seaborn as sns
sns.set(context='talk', style='white')

import warnings
warnings.filterwarnings('ignore')

def parse_response(s):
    try:
        return s[~s.isnull()][0]
    except IndexError:
        return np.nan

# Load in the data. This spreadsheet is (almost) direct from SurveyMonkey,
# though IP addresses, access dates, and free-text responses were scrubbed
# to anonymize respondents.

df = pd.read_csv('public_survey_data.csv', sep=',')
# import pandas as pd;df = pd.read_csv('test_survey.csv',sep=',');
# SurveyMonkey provides an odd nesting of responses when exporting results.
# We'd like to convert this structure to a pandas MultiIndex data frame.
# First, let's find question indices -- adapted from https://stackoverflow.com/a/49584888

indices   = [ i for i, c in enumerate(df.columns) if not c.startswith('Unnamed') ]
questions = [ c for c in df.columns if not c.startswith('Unnamed') ]
slices    = [ slice (i, j) for i, j in zip(indices, indices[1:] + [None])]

# check if header rows and (first) answers are aligned
# for q in slices:
#    print(df.iloc[:, q])  # Use `display` if using Jupyter

# merge extra multi-answer columns
df.replace ( np.nan,';',regex=True, inplace=True );
for i in range( 1, df.shape[1] ):
    if ( i in indices ):
        # print ('{} is a question'.format(i));
        lastq = i;
    else:
        # print ('attaching {} to question {}'.format(i,lastq));
        df.iloc[ :, lastq ] = df.iloc[ :, lastq ].astype(str) + ';' + df.iloc [ :, i ].astype(str);

# drop the copied columns
for i in reversed ( range ( 1, df.shape[1] ) ):
    if ( not i in indices ): 
        del df[df.columns[i]];

df.replace ( ';nan',';',regex=True, inplace=True ); 
df.replace ( ';;','',regex=True, inplace=True ); 
df.replace ( ';','',regex=False, inplace=True ); 
df.to_csv  ( 'tidydata.csv' );

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

df = df.applymap( lambda x: x.split( ';' ) if isinstance( x, str ) else x )
# for row in range ( df.shape [ 0 ] ):
#     for col in range ( df.shape [ 1 ] ):
#         value = df.iloc[row][col];        
#         if ( type ( value ) == str ):
#             value = value.split( ';' );
#         df.iloc[row, col] = value; 

df.to_json ( 'tidydata.json' );
    