import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import zip_longest
from patsy.contrasts import Treatment
from pandas.api.types import CategoricalDtype
from statsmodels.miscmodels.ordinal_model import OrderedModel

plt.rcParams.update({'font.size': 16})
sns.set(context='talk', style='white')
warnings.filterwarnings('ignore')

demographics = {
    'Are you a member of OHBM?': 'is_member',
    'What geographic region are you currently located in?': 'geographic_region',
    'What is your current career status?': 'career_stage'
}
rename_questions = {
    'How would you describe your access to the OHBM job board?': 'job_board_access',
    'How would you describe the content on the OHBM job board?': 'job_board_content',
    'How would you describe your access to emails from OHBM?': 'email_access',
    'How would you describe the content in OHBM emails?': 'email_content',
    'How would you describe your access to the OHBM blog?': 'blog_access',
    'How would you describe the content in the OHBM blog?': 'blog_content',
    'How would you describe your access to OHBM Twitter?': 'twitter_access',
    'How would you describe the content on OHBM Twitter?': 'twitter_content',
    'How would you describe your access to the NeuroSalience podcast?': 'podcast_access',
    'How would you describe the content in the NeuroSalience podcast?': 'podcast_content',
    'How would you describe your access to OHBM Facebook?': 'facebook_acces',
    'How would you describe the content in OHBM Facebook?': 'facebook_content',
    'How would you describe your access to OHBM YouTube?': 'youtube_access',
    'How would you describe the content in OHBM YouTube?': 'youtube_content',
    'How would you describe your access to OHBM LinkedIn?': 'linkedin_access',
    'How would you describe the content in OHBM LinkedIn?': 'linkedin_content',
    'How would you describe your access to the OHBM OnDemand?': 'ondemand_access',
    'How would you describe the content in OHBM OnDemand?': 'ondemand_content',
    'Which of the following platforms do you use to access OHBM content? When applicable, a direct link to the platform is provided next to each option. Please check all options that apply.': 'content_platform',
    'Do you currently follow any of the following OHBM Special Interest Groups (SIG) platforms? When applicable, a direct link to the platform is provided next to each option. Please check all options that apply.': 'sig_platform',
    'How important is each of these types of content to you?': 'content_importance'
}
content_platform_opts = {
    'OHBM website (https://www.humanbrainmapping.org)',
    'Official OHBM Emails (*@humanbrainmapping.org emails)',
    'Blog (https://www.ohbmbrainmappingblog.com)',
    'Twitter (https://twitter.com/OHBM)',
    'NeuroSalience podcast (https://anchor.fm/ohbm)',
    'Facebook (https://www.facebook.com/humanbrainmapping.org)',
    'Youtube (https://www.youtube.com/channel/UCwMM4wEFi_hx2_6wVVoPBJA)',
    'LinkedIn (https://www.linkedin.com/company/organization-for-human-brain-mapping)',
    'OHBM OnDemand (https://www.pathlms.com/ohbm)',
    'OHBM Job Board'
}
sig_platform_opts = {
    'BrainArt Twitter (https://twitter.com/OHBM_BrainArt)',
    'BrainArt Instagram (https://www.instagram.com/ohbm_basig)',
    'Open Science Twitter (https://twitter.com/OhbmOpen)',
    'Student-Postdoc Blog (https://www.ohbmtrainees.com/sig-blog)',
    'Student-Postdoc Facebook (https://www.facebook.com/OHBMStudentandPostdocSection)',
    'Student-Postdoc Twitter (https://twitter.com/OHBM_Trainees)',
    'Sustainability Blog (https://ohbm-environment.org/blog)',
    'Sustainability Twitter (https://twitter.com/OhbmEnvironment)'
}
content_importance_options = {
    'Information on the annual meeting',
    'Information on OHBM behind-the-scenes (e.g. governance, financing)',
    'Updates on OHBM committee / SIGs activities throughout the year',
    'Reviews on OHBM activities',
    'Interviews with human brain mappers',
    'Tutorials on different topics related to human brain mapping',
    'Controversial topics in the field',
    'Exciting new science or developments in the brain mapping field'
}

# Load in the data. This spreadsheet is (almost) direct from SurveyMonkey,
# though IP addresses, access dates, and free-text responses were scrubbed
# to anonymize respondents.

df = pd.read_csv('public_survey_data.csv', sep=',', index_col='Unnamed: 0')

# SurveyMonkey provides an odd nesting of responses when exporting results.
# We'd like to convert this structure to a pandas MultiIndex data frame.
# First, let's find question indices -- adapted from https://stackoverflow.com/a/49584888

indices = [i for i, c in enumerate(df.columns) if not c.startswith('Unnamed')]
repeats = [len(df.columns[i:j]) for i, j in zip_longest(indices, indices[1:])]

# We can pair each possible response with its associated question...
matched_questions = []
for question, n_rep in zip(df.columns[indices], repeats):
    matched_questions += [question] * n_rep

df.columns = matched_questions
df.index.name = 'participant_id'

responses = pd.melt(
    df.iloc[1:].reset_index().rename({**demographics, **rename_questions}, axis=1), 
    id_vars=['participant_id'] + list(demographics.values()), 
    value_name='response', 
    var_name='question'
)

# We'll need to set up a coding scheme for our demographic data
responses['is_member'] = responses.is_member.astype(bool)
responses['geographic_region'] = responses.geographic_region.astype('category')
responses['career_stage'] = responses.career_stage.astype('category')

# now let's run some basic analyses to check against SurveyMonkey outputs
responses['is_member'].value_counts(normalize=True)
responses['geographic_region'].value_counts(normalize=True)
responses['career_stage'].value_counts(normalize=True)

# first, look at twitter access
tw = responses.query('question == "twitter_access"')
tw.dropna(inplace=True)

levels = [
    'I don’t use Twitter / NA',
    'I use Twitter but didn’t know that the OHBM Twitter account exists',
    'I use Twitter and know about OHBM Twitter, but I don’t follow the account',
    'I use Twitter, follow the OHBM Twitter account, and occasionally see their tweets',
    'I use Twitter, follow the OHBM Twitter account, and regularly see their tweets'
    ]
cat = CategoricalDtype(categories=levels, ordered=True)
tw['twitter_use'] = tw.response.astype(cat, copy=False)

mod = OrderedModel.from_formula(
    'twitter_use ~ C(geographic_region, Treatment) + C(career_stage, Treatment)',
    data=tw)
res = mod.fit(method='bfgs')
print(res.summary())
print(np.exp(res.params))  # odds ratios
