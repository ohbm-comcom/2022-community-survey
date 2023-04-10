import json
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


def fit_model(responses, query, levels):
    """
    Fit an ordinal logistic model to the Likert response data.

    You can find more information on this model in the StatsModels documentation:
    https://www.statsmodels.org/dev/generated/statsmodels.miscmodels.ordinal_model.OrderedModel.html

    Parameters
    ----------
    responses : pd.DataFrame
        Survey responses data, loaded as a Pandas DataFrame and coerced
        into the appropriate format.
    query : str
        The query for which to fit a model. Possible query values are defined
        in the `rename_questions` mapping.
    levels : list
        A query specific list of possible participant Likert responses, listed
        in ordinal ranking.
    """
    # first, subset the full data frame
    data = responses.query(f'question == \"{query}\"')
    data.dropna(inplace=True)

    # then, define the response levels as ordered
    cat = CategoricalDtype(categories=levels, ordered=True)
    data[f'{query}'] = data.response.astype(cat, copy=False)

    # and fit and return the model
    mod = OrderedModel.from_formula(
        f'{query} ~ C(geographic_region, Treatment) + C(career_stage, Treatment)',
        data=data)
    res = mod.fit(method='bfgs')
    return res


def format_data(survey_data_path):
    """
    """
    # Load in the data. This spreadsheet is (almost) direct from SurveyMonkey,
    # though IP addresses, access dates, and free-text responses were scrubbed
    # to anonymize respondents.

    df = pd.read_csv(survey_data_path, sep=',', index_col='Unnamed: 0')

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

    # Now we'll rename our columns to ease analysis--
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
        'How would you describe your access to OHBM Facebook?': 'facebook_access',
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

    return responses


if __name__ == "__main__":
    responses = format_data('public_survey_data.csv')

    # now let's run some basic analyses to check against SurveyMonkey outputs
    responses['is_member'].value_counts(normalize=True)
    responses['geographic_region'].value_counts(normalize=True)
    responses['career_stage'].value_counts(normalize=True)

    # load in our JSON with question-level metadata for available responses
    with open('levels.json') as f:
        queries = json.load(f)

    for q in queries:
        levels = queries[q]
        print(f'Now fitting a model for {q}....')
        res = fit_model(responses, q, levels)
        print(res.summary())
        print(np.exp(res.params))  # odds ratios
        print()


# FIXME : blog_access
# ValueError: shapes (293,12) and (11,) not aligned: 12 (dim 1) != 11 (dim 0)
    # "blog_access" : [
    #     "I didn't know that OHBM had a blog",
    #     "I know about the blog, but I don’t read it",
    #     "I know about the blog and occasionally read the posts",
    #     "I know about the blog and regularly read the post"
    # ],


# FIXME: facebook_access
# ValueError: zero-size array to reduction operation maximum which has no identity
    # "facebook_access" : [
    #     "I don’t use Facebook / NA",
    #     "I use Facebook but didn’t know that the OHBM Facebook page exists",
    #     "I use Facebook and know about the OHBM Facebook page, but I don’t follow/like the page",
    #     "I use Facebook, follow/like the page, and occasionally see their posts",
    #     "I use Facebook, follow/like the page, and regularly see their posts"
    # ],