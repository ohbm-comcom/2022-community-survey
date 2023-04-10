---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import json

from survey_analysis import format_data, fit_model
```

```{code-cell} ipython3
responses = format_data('public_survey_data.csv')
```

```{code-cell} ipython3
# now let's run some basic analyses to check against SurveyMonkey outputs
responses['is_member'].value_counts(normalize=True)
```

```{code-cell} ipython3
responses['geographic_region'].value_counts(normalize=True)
```

```{code-cell} ipython3
responses['career_stage'].value_counts(normalize=True)
```

```{code-cell} ipython3
# for the other questions, we can load in our sidecar JSON
# with question-level metadata
with open('levels.json') as f:
    queries = json.load(f)
```

```{code-cell} ipython3
print(f'There are {len(queries.keys())} questions to consider.') 
print(queries.keys())
```

```{code-cell} ipython3
question = 'job_board_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'job_board_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'email_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'email_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'blog_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'blog_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'twitter_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'twitter_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'podcast_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'podcast_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'facebook_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'youtube_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'youtube_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'linkedin_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'linkedin_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'ondemand_access'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'ondemand_content'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'content_platform'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'sig_platform'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```

```{code-cell} ipython3
question = 'content_importance'
scale_values = queries[question]

res = fit_model(responses, question, scale_values)
print(res.summary())
```
